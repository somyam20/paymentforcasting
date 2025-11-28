from fastapi import APIRouter, HTTPException, Form
import litellm
import asyncio
import requests
from pydantic import BaseModel
from src.utils.db import get_project_url, init_db, get_alias
from src.utils.s3_utils import download_to_bytes, extract_filename_from_url
from src.utils.normalize import read_input_file
from src.utils.aliaser import make_alias
from config.settings import MOVING_AVG_WINDOW
from config.config import get_model_config
import pandas as pd
import io
import logging
import traceback
import re
import os
import json

logging.basicConfig(
    level=logging.INFO,
    filemode="a",
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    project_name: str
    query: str


def extract_clean_answer(text: str) -> str:
    """
    Extract only the answer portion from the LLM response.
    Removes markdown formatting like ** and extracts text between **Answer:** and **Forecast:**
    """
    # Try to find the answer section
    answer_match = re.search(r'\*\*Answer:\*\*\s*(.*?)(?:\*\*Forecast:\*\*|\*\*Reasoning:\*\*|$)', text, re.DOTALL | re.IGNORECASE)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # If no structured format found, return the entire text
        answer = text.strip()
    
    # Remove all ** markdown formatting
    answer = re.sub(r'\*\*', '', answer)
    
    # Clean up extra whitespace
    answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
    answer = answer.strip()
    
    return answer


@router.post("/query")
async def query_project(
    req: QueryRequest,
    user_metadata: str | None = Form(None)  
    ):
    user_metadata = json.loads(user_metadata) if user_metadata else {}
    team_id = user_metadata.get("team_id")
    try:
        async with get_model_config() as config:
            # Get the team's model configuration
            team_config = await config.get_team_model_config(team_id)
            model = team_config["selected_model"]
            provider = team_config["provider"]
            provider_model = f"{provider}/{model}"
            model_config = team_config["config"]
 
            # Create LLM instance with the team's configuration
            llm_params = {
                "model": provider_model,
                **model_config  
            }
            auth_token = requests.headers.get("Authorization")
            if auth_token:
                llm_params.update({"auth_token": auth_token})
    except Exception as e:
        logging.error(f"Error extracting attendees: {str(e)}")
        return []

    await init_db()
    logger.info("=" * 80)
    logger.info("QUERY REQUEST START")
    logger.info("Project: %s", req.project_name)
    logger.info("Query: %s", req.query)
    logger.info("=" * 80)

    # 1. Get S3 URL
    s3_url = await get_project_url(req.project_name)
    if not s3_url:
        logger.error("Project not found in DB: %s", req.project_name)
        raise HTTPException(status_code=404, detail="project not found")
    logger.info("✓ Found S3 URL: %s", s3_url)

    # 2. Download file
    try:
        data_bytes = await download_to_bytes(s3_url)
        logger.info("✓ Downloaded %d bytes from S3", len(data_bytes) if data_bytes else 0)
    except Exception as e:
        logger.exception("✗ Failed to download from S3: %s", e)
        raise HTTPException(status_code=500, detail=f"failed to download from s3: {e}")

    # FIX: Use the extract_filename_from_url function to properly handle pre-signed URLs
    filename = extract_filename_from_url(s3_url)
    logger.info("✓ Filename: %s", filename)

    # 3. Parse file
    try:
        df = await read_input_file(io.BytesIO(data_bytes), filename)
        logger.info("✓ Parsed dataframe: shape=%s", df.shape)
        logger.info("✓ Columns: %s", list(df.columns))
        logger.info("✓ First 3 rows:\n%s", df.head(3).to_string(index=False))
    except Exception as e:
        logger.exception("✗ Failed to parse file: %s", e)
        raise HTTPException(status_code=500, detail=f"failed to parse file: {e}")

    # 4. Build alias mapping
    name_col = "customer_name"
    mapping = {}  # real_name -> alias
    
    logger.info("-" * 80)
    logger.info("ALIASING PROCESS")
    logger.info("-" * 80)
    
    if name_col not in df.columns:
        logger.error("✗ Column '%s' NOT FOUND in dataframe!", name_col)
        logger.error("Available columns: %s", list(df.columns))
        raise HTTPException(status_code=500, detail=f"Expected column '{name_col}' not found in data")
    
    logger.info("✓ Column '%s' exists in dataframe", name_col)
    
    try:
        # Get unique customer names
        raw_customers = df[name_col].dropna().unique()
        logger.info("✓ Found %d unique customer values (before processing)", len(raw_customers))
        
        # Clean and sort
        unique_customers = [str(x).strip() for x in raw_customers if str(x).strip()]
        unique_customers.sort(key=lambda x: -len(x))  # longest first
        
        logger.info("✓ After cleaning: %d valid customers", len(unique_customers))
        logger.info("✓ Sample customers (top 5): %s", unique_customers[:5])
        
        # Generate aliases for each customer
        for idx, cust in enumerate(unique_customers):
            logger.info("")
            logger.info("Processing customer %d/%d: '%s'", idx + 1, len(unique_customers), cust)
            
            # Check if alias already exists
            alias = await get_alias(req.project_name, cust)
            
            if alias:
                logger.info("  ✓ Found existing alias: '%s' -> '%s'", cust, alias)
                mapping[cust] = alias
            else:
                logger.info("  ⚙ No existing alias, generating new one...")
                try:
                    alias = await make_alias(req.project_name, cust)
                    logger.info("  ✓ Generated new alias: '%s' -> '%s'", cust, alias)
                    mapping[cust] = alias
                except Exception as ex_alias:
                    logger.exception("  ✗ Failed to create alias for '%s': %s", cust, ex_alias)
                    # Continue with other customers
        
        logger.info("")
        logger.info("-" * 80)
        logger.info("FINAL MAPPING (%d entries):", len(mapping))
        for real, alias in list(mapping.items())[:10]:  # Show first 10
            logger.info("  '%s' -> '%s'", real, alias)
        if len(mapping) > 10:
            logger.info("  ... and %d more", len(mapping) - 10)
        logger.info("-" * 80)
        
    except Exception as e:
        logger.exception("✗ Error building alias mapping: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to build aliases: {e}")

    if not mapping:
        logger.warning("⚠ WARNING: No aliases created! This means no customer names will be anonymized.")

    # 5. Apply aliases to dataframe
    logger.info("")
    logger.info("APPLYING ALIASES TO DATAFRAME")
    logger.info("-" * 80)
    
    df_alias = df.copy()
    
    if mapping:
        try:
            # Show before aliasing
            logger.info("Before aliasing (sample):")
            logger.info("%s", df_alias[name_col].head(5).tolist())
            
            # Apply mapping
            df_alias[name_col] = df_alias[name_col].astype(str).str.strip().map(mapping)
            
            # Show after aliasing
            logger.info("After aliasing (sample):")
            logger.info("%s", df_alias[name_col].head(5).tolist())
            
            # Check for unmapped values (will be NaN)
            unmapped_count = df_alias[name_col].isna().sum()
            if unmapped_count > 0:
                logger.warning("⚠ WARNING: %d rows have unmapped customer names (will be NaN)", unmapped_count)
            else:
                logger.info("✓ All customer names successfully aliased")
                
        except Exception as e:
            logger.exception("✗ Failed to apply aliases to dataframe: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to apply aliases: {e}")
    else:
        logger.warning("⚠ Skipping dataframe aliasing (no mapping available)")

    # 6. Transform query with aliases
    logger.info("")
    logger.info("TRANSFORMING QUERY")
    logger.info("-" * 80)
    logger.info("Original query: %s", req.query)
    
    transformed_query = req.query
    replacements_made = []
    
    if mapping:
        for cust, alias in mapping.items():
            if cust and cust in transformed_query:
                transformed_query = transformed_query.replace(cust, alias)
                replacements_made.append(f"'{cust}' -> '{alias}'")
        
        if replacements_made:
            logger.info("✓ Query transformations:")
            for repl in replacements_made:
                logger.info("  %s", repl)
            logger.info("Transformed query: %s", transformed_query)
        else:
            logger.info("⚠ No customer names found in query to replace")
    else:
        logger.warning("⚠ Skipping query transformation (no mapping available)")

    # 7. Precompute statistics for each customer
    logger.info("")
    logger.info("PRECOMPUTING CUSTOMER STATISTICS")
    logger.info("-" * 80)
    
    # Moving average window size (number of recent invoices to consider)
    MOVING_AVG_WINDOW = int(os.getenv("MOVING_AVG_WINDOW", "3"))
    logger.info("Using moving average window: %d invoices", MOVING_AVG_WINDOW)
    
    customer_stats = {}
    
    try:
        # Identify amount column (could be net_amount or total_invoice_amount)
        amount_col = None
        for col in ['total_invoice_amount', 'net_amount', 'amount']:
            if col in df_alias.columns:
                amount_col = col
                break
        
        if not amount_col:
            logger.warning("⚠ No amount column found, skipping statistics computation")
        
        # Identify date columns
        payment_date_col = 'payment_date' if 'payment_date' in df_alias.columns else None
        invoice_date_col = 'invoice_date' if 'invoice_date' in df_alias.columns else None
        due_date_col = 'due_date' if 'due_date' in df_alias.columns else None
        
        logger.info("Using columns: amount=%s, payment_date=%s, invoice_date=%s", 
                   amount_col, payment_date_col, invoice_date_col)
        
        # Get unique customers from aliased dataframe
        unique_aliases = df_alias[name_col].dropna().unique()
        
        for alias in unique_aliases:
            # Filter data for this customer
            customer_df = df_alias[df_alias[name_col] == alias].copy()
            
            # Filter only paid invoices (where payment_date is not null)
            if payment_date_col:
                paid_df = customer_df[customer_df[payment_date_col].notna()].copy()
            else:
                paid_df = customer_df.copy()
            
            if len(paid_df) == 0:
                logger.info("Customer '%s': No paid invoices found", alias)
                continue
            
            stats = {
                'total_invoices': len(paid_df),
                'mean_amount': 0,
                'moving_avg_amount': 0,
                'moving_avg_window': 0,
                'total_paid': 0,
                'avg_delay_days': 0,
                'last_payment_date': None
            }
            
            # Calculate amount statistics
            if amount_col and amount_col in paid_df.columns:
                # Convert to numeric, handling any string formatting
                amounts = pd.to_numeric(paid_df[amount_col], errors='coerce')
                amounts = amounts.dropna()
                
                if len(amounts) > 0:
                    # Overall mean
                    stats['mean_amount'] = float(amounts.mean())
                    stats['total_paid'] = float(amounts.sum())
                    
                    # Moving average: take last N invoices (sorted by date if possible)
                    if payment_date_col and payment_date_col in paid_df.columns:
                        # Sort by payment date to get most recent invoices
                        paid_df_sorted = paid_df.sort_values(by=payment_date_col, ascending=False)
                        recent_amounts = pd.to_numeric(paid_df_sorted[amount_col].head(MOVING_AVG_WINDOW), errors='coerce')
                        recent_amounts = recent_amounts.dropna()
                    else:
                        # If no date column, just take last N rows
                        recent_amounts = amounts.tail(MOVING_AVG_WINDOW)
                    
                    if len(recent_amounts) > 0:
                        stats['moving_avg_amount'] = float(recent_amounts.mean())
                        stats['moving_avg_window'] = len(recent_amounts)
                        logger.info("  Moving avg calculated from %d most recent invoices", len(recent_amounts))
            
            # Calculate delay statistics
            if payment_date_col and invoice_date_col:
                try:
                    paid_df[payment_date_col] = pd.to_datetime(paid_df[payment_date_col], errors='coerce')
                    paid_df[invoice_date_col] = pd.to_datetime(paid_df[invoice_date_col], errors='coerce')
                    
                    # Calculate delays
                    delays = (paid_df[payment_date_col] - paid_df[invoice_date_col]).dt.days
                    delays = delays.dropna()
                    
                    if len(delays) > 0:
                        stats['avg_delay_days'] = float(delays.mean())
                    
                    # Get last payment date
                    last_payment = paid_df[payment_date_col].max()
                    if pd.notna(last_payment):
                        stats['last_payment_date'] = last_payment.strftime('%Y-%m-%d')
                        
                except Exception as e:
                    logger.warning("Failed to calculate dates for '%s': %s", alias, e)
            
            customer_stats[alias] = stats
            logger.info("Customer '%s': %d paid invoices, mean=%.2f, moving_avg=%.2f (window=%d), total=%.2f, avg_delay=%.1f days", 
                       alias, stats['total_invoices'], stats['mean_amount'], 
                       stats['moving_avg_amount'], stats['moving_avg_window'],
                       stats['total_paid'], stats['avg_delay_days'])
        
        logger.info("-" * 80)
        logger.info("Computed statistics for %d customers", len(customer_stats))
        
    except Exception as e:
        logger.exception("✗ Failed to compute customer statistics: %s", e)
        # Continue without stats, LLM will work with raw data
        customer_stats = {}
    
    # 8. Convert to CSV for LLM
    try:
        data_text = df_alias.to_csv(index=False)
        logger.info("✓ Converted dataframe to CSV (%d characters)", len(data_text))
        logger.info("CSV preview (first 500 chars):\n%s", data_text[:500])
    except Exception as e:
        logger.exception("✗ Failed to convert dataframe to CSV: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to convert to CSV: {e}")

    # 9. Build LLM prompt with precomputed statistics
    
    # Format statistics for prompt
    stats_text = ""
    if customer_stats:
        stats_text = "\n\nPRECOMPUTED CUSTOMER STATISTICS:\n"
        stats_text += "-" * 60 + "\n"
        for alias, stats in customer_stats.items():
            stats_text += f"Customer: {alias}\n"
            stats_text += f"  - Total paid invoices: {stats['total_invoices']}\n"
            stats_text += f"  - Overall mean amount: ₹{stats['mean_amount']:.2f}\n"
            stats_text += f"  - Moving average amount (last {stats['moving_avg_window']} invoices): ₹{stats['moving_avg_amount']:.2f}\n"
            stats_text += f"  - Total amount paid: ₹{stats['total_paid']:.2f}\n"
            stats_text += f"  - Average payment delay: {stats['avg_delay_days']:.1f} days\n"
            if stats['last_payment_date']:
                stats_text += f"  - Last payment date: {stats['last_payment_date']}\n"
            stats_text += "\n"
        stats_text += "-" * 60 + "\n"
    
    prompt = f"""
You are an expert financial data analyst having a conversation with a business stakeholder. 
Provide detailed, conversational, and easy-to-understand responses that tell the complete story.

Below is the dataset with CUSTOMER NAMES replaced by ALIASES:
------------------------------------------------------------
{data_text}
------------------------------------------------------------
{stats_text}

USER QUERY:
{transformed_query}

CRITICAL INSTRUCTIONS:
The statistics above have been PRECOMPUTED for you. USE THESE EXACT NUMBERS in your response.
Write in a natural, conversational tone as if you're explaining this to a colleague over coffee.

YOUR APPROACH:
1. Start with a direct answer to their question
2. Then elaborate with context and details
3. Explain what the numbers mean in practical terms
4. Compare trends to give perspective
5. Describe patterns you observe in the data
6. Be specific with numbers, dates, and percentages
7. Help them understand the "why" behind the numbers

CONVERSATIONAL GUIDELINES:
- Use natural language like "Looking at their payment history..." or "What's interesting here is..."
- Explain trends: "Their recent invoices have been higher..." or "They've been pretty consistent..."
- Give context: "Out of X invoices paid..." or "Over the past period..."
- Make comparisons: "compared to their usual average of..." or "which is about X% higher than before"
- Describe patterns: "They typically pay within X days..." or "There's a clear upward trend..."
- Be conversational but professional
- Avoid bullet points, use flowing paragraphs
- Don't give recommendations, just describe what you see in the data

AMOUNT PREDICTIONS:
- Use the precomputed **moving_avg_amount** 
- Explain it naturally: "Based on their last [window] invoices, they've been averaging around ₹[amount]..."
- Compare to overall history: "This is actually [higher/lower] than their overall average of ₹[mean], suggesting their invoice amounts are [trending up/down/staying stable]"
- Mention the percentage difference if significant
- Explain what this trend might indicate about the business relationship

DATE PREDICTIONS:
- Calculate: next_payment_date = last_payment_date + avg_delay_days
- Describe their payment timing naturally: "They usually take about [X] days to pay after receiving an invoice..."
- Give the specific expected date: "Since their last payment was on [date], I'd expect the next one around [predicted_date]"
- Describe their payment behavior: "They're fairly reliable" or "They tend to pay a bit late" etc.

RESPONSE FORMAT:
Write 3-4 detailed paragraphs that flow naturally, covering:

Paragraph 1: Direct answer with immediate context
- Answer their specific question right away
- Give the key numbers (amount and/or date)
- Mention how many invoices and total paid amount for context

Paragraph 2: Historical context and patterns
- Describe their overall payment history
- Compare recent behavior to historical average
- Explain any trends you notice (amounts increasing/decreasing, payment timing patterns)
- Use specific numbers and percentages

Paragraph 3: Detailed forecast explanation
- Explain the predicted amount using moving average
- Give the reasoning behind the date prediction
- Describe their typical payment behavior
- Mention data quality (how many invoices you're basing this on)

Paragraph 4 (if needed): Additional observations
- Any interesting patterns in their payment behavior
- Confidence level in the prediction based on data consistency
- What the trend direction tells us about the relationship

IMPORTANT: 
- Write in complete, flowing paragraphs - NO bullet points or lists
- Be conversational and natural
- Use the precomputed statistics
- Include specific numbers, dates, and percentages
- Make it feel like you're having a conversation, not writing a report

Now provide a detailed, conversational response.
"""

    logger.info("✓ Built LLM prompt (%d characters)", len(prompt))

    # 10. Call LLM
    logger.info("")
    logger.info("CALLING LLM")
    logger.info("-" * 80)
    
    try:
        llm_response = await asyncio.to_thread(lite_client.generate, prompt)
        logger.info("✓ LLM responded (%d characters)", len(llm_response))
        logger.info("LLM response preview:\n%s", llm_response[:500])
    except Exception as e:
        logger.exception("✗ LLM call failed: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 11. DE-ALIAS the response
    logger.info("")
    logger.info("DE-ALIASING RESPONSE")
    logger.info("-" * 80)
    
    dealiased_response = llm_response
    dealias_replacements = []
    
    if mapping:
        # Create reverse mapping: alias -> real_name
        reverse_mapping = {alias: real_name for real_name, alias in mapping.items()}
        logger.info("Created reverse mapping (%d entries)", len(reverse_mapping))
        
        # Sort by length (longest first) to avoid substring issues
        sorted_aliases = sorted(reverse_mapping.keys(), key=lambda x: -len(x))
        
        for alias in sorted_aliases:
            real_name = reverse_mapping[alias]
            if alias in dealiased_response:
                dealiased_response = dealiased_response.replace(alias, real_name)
                dealias_replacements.append(f"'{alias}' -> '{real_name}'")
        
        if dealias_replacements:
            logger.info("✓ De-aliasing transformations:")
            for repl in dealias_replacements[:10]:  # Show first 10
                logger.info("  %s", repl)
            if len(dealias_replacements) > 10:
                logger.info("  ... and %d more", len(dealias_replacements) - 10)
        else:
            logger.info("⚠ No aliases found in LLM response to replace")
            
        logger.info("De-aliased response preview:\n%s", dealiased_response[:500])
    else:
        logger.warning("⚠ Skipping de-aliasing (no mapping available)")

    # 12. Extract clean answer
    clean_answer = extract_clean_answer(dealiased_response)
    logger.info("")
    logger.info("EXTRACTED CLEAN ANSWER:")
    logger.info("%s", clean_answer[:300])

    logger.info("")
    logger.info("=" * 80)
    logger.info("QUERY REQUEST COMPLETE")
    logger.info("=" * 80)

    # Return only the clean answer
    return {
        "answer": clean_answer
    }
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.db import get_project_url, init_db, get_alias
from src.utils.s3_utils import download_to_bytes
from src.utils.normalize import read_input_file
from src.utils.aliaser import make_alias
from src.llm.lite_client import lite_client
import pandas as pd
import io
import logging
import traceback
import re

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
def query_project(req: QueryRequest):
    init_db()
    logger.info("=" * 80)
    logger.info("QUERY REQUEST START")
    logger.info("Project: %s", req.project_name)
    logger.info("Query: %s", req.query)
    logger.info("=" * 80)

    # 1. Get S3 URL
    s3_url = get_project_url(req.project_name)
    if not s3_url:
        logger.error("Project not found in DB: %s", req.project_name)
        raise HTTPException(status_code=404, detail="project not found")
    logger.info("✓ Found S3 URL: %s", s3_url)

    # 2. Download file
    try:
        data_bytes = download_to_bytes(s3_url)
        logger.info("✓ Downloaded %d bytes from S3", len(data_bytes) if data_bytes else 0)
    except Exception as e:
        logger.exception("✗ Failed to download from S3: %s", e)
        raise HTTPException(status_code=500, detail=f"failed to download from s3: {e}")

    filename = s3_url.split("/")[-1]
    logger.info("✓ Filename: %s", filename)

    # 3. Parse file
    try:
        df = read_input_file(io.BytesIO(data_bytes), filename)
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
            alias = get_alias(req.project_name, cust)
            
            if alias:
                logger.info("  ✓ Found existing alias: '%s' -> '%s'", cust, alias)
                mapping[cust] = alias
            else:
                logger.info("  → No existing alias, generating new one...")
                try:
                    alias = make_alias(req.project_name, cust)
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
                    stats['mean_amount'] = float(amounts.mean())
                    stats['total_paid'] = float(amounts.sum())
            
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
            logger.info("Customer '%s': %d paid invoices, mean=%.2f, total=%.2f, avg_delay=%.1f days", 
                       alias, stats['total_invoices'], stats['mean_amount'], 
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
            stats_text += f"  - Mean invoice amount: ₹{stats['mean_amount']:.2f}\n"
            stats_text += f"  - Total amount paid: ₹{stats['total_paid']:.2f}\n"
            stats_text += f"  - Average payment delay: {stats['avg_delay_days']:.1f} days\n"
            if stats['last_payment_date']:
                stats_text += f"  - Last payment date: {stats['last_payment_date']}\n"
            stats_text += "\n"
        stats_text += "-" * 60 + "\n"
    
    prompt = f"""
You are an expert financial data analyst. 

Below is the dataset with CUSTOMER NAMES replaced by ALIASES:
------------------------------------------------------------
{data_text}
------------------------------------------------------------
{stats_text}

USER QUERY:
{transformed_query}

CRITICAL INSTRUCTIONS:
The statistics above have been PRECOMPUTED for you. USE THESE EXACT NUMBERS in your response.

YOUR TASKS:
1. Identify which alias(es) the question refers to from the query.

2. FOR AMOUNT PREDICTIONS:
   - Use the PRECOMPUTED mean_amount from the statistics above
   - DO NOT recalculate - use the exact number provided
   - Format: "Expected next invoice amount: ₹[use precomputed mean] (based on [N] paid invoices)"

3. FOR DATE PREDICTIONS:
   - Use the precomputed average payment delay
   - Calculate: next_payment_date = last_payment_date + avg_delay_days
   - Provide specific date prediction

4. If the customer is mentioned in the query:
   - Extract their statistics from the PRECOMPUTED section above
   - Use those exact numbers in your answer
   - DO NOT attempt to recalculate from the raw data

5. Provide final output in this format:
   **Answer:**
   Based on [N] paid invoices (from precomputed stats), [customer] has paid a total of ₹[total_paid]. 
   The average invoice amount is ₹[mean_amount]. 
   Expected next invoice amount: ₹[mean_amount]
   
   **Forecast:**
   - Next invoice amount: ₹[precomputed mean_amount]
   - Next payment date: [calculated from last_payment + avg_delay]
   - Based on: [N] historical invoices, [X] day average delay
   
   **Reasoning:**
   Using precomputed statistics: mean amount ₹[X] from [N] invoices, average delay [Y] days.

IMPORTANT: Always use the precomputed statistics provided above. Do not recalculate.

Now produce the answer using the precomputed statistics.
"""

    logger.info("✓ Built LLM prompt (%d characters)", len(prompt))

    # 10. Call LLM
    logger.info("")
    logger.info("CALLING LLM")
    logger.info("-" * 80)
    
    try:
        llm_response = lite_client.generate(prompt)
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
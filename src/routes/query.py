# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from src.utils.db import get_project_url, init_db, get_alias
# from src.utils.s3_utils import download_to_bytes, extract_filename_from_url
# from src.utils.normalize import read_input_file
# from src.utils.aliaser import make_alias
# from src.utils.prompts_loader import format_prompt
# from src.llm.lite_client import lite_client
# from config.settings import MOVING_AVG_WINDOW
# import pandas as pd
# import io
# import logging
# import traceback
# import re
# import os
# import asyncio


# logging.basicConfig(
#     level=logging.INFO,
#     filemode="a",
#     filename="app.log",
#     format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# router = APIRouter()


# class QueryRequest(BaseModel):
#     project_name: str
#     query: str
#     metadata:str

# def extract_clean_answer(text: str) -> str:
#     """
#     Extract only the answer portion from the LLM response.
#     Removes markdown formatting like ** and extracts text between **Answer:** and **Forecast:**
#     """
#     # Try to find the answer section
#     answer_match = re.search(r'\*\*Answer:\*\*\s*(.*?)(?:\*\*Forecast:\*\*|\*\*Reasoning:\*\*|$)', text, re.DOTALL | re.IGNORECASE)
    
#     if answer_match:
#         answer = answer_match.group(1).strip()
#     else:
#         # If no structured format found, return the entire text
#         answer = text.strip()
    
#     # Remove all ** markdown formatting
#     answer = re.sub(r'\*\*', '', answer)
    
#     # Clean up extra whitespace
#     answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
#     answer = answer.strip()
    
#     return answer


# async def process_customer_async(project_name: str, cust: str, idx: int, total: int) -> tuple:
#     """Helper to process single customer asynchronously"""
#     logger.info("")
#     logger.info("Processing customer %d/%d: '%s'", idx, total, cust)
    
#     # Check if alias already exists (sync -> thread)
#     alias = await asyncio.to_thread(get_alias, project_name, cust)
    
#     if alias:
#         logger.info("  ✓ Found existing alias: '%s' -> '%s'", cust, alias)
#         return (cust, alias)
#     else:
#         logger.info("  ⚙ No existing alias, generating new one...")
#         try:
#             # FIXED: await the make_alias coroutine
#             alias = await make_alias(project_name, cust)
#             logger.info("  ✓ Generated new alias: '%s' -> '%s'", cust, alias)
#             return (cust, alias)
#         except Exception as ex_alias:
#             logger.exception("  ✗ Failed to create alias for '%s': %s", cust, ex_alias)
#             return None


# async def build_alias_mapping(project_name: str, df: pd.DataFrame, name_col: str) -> dict:
#     """Async helper to build alias mapping"""
#     mapping = {}  # real_name -> alias
    
#     logger.info("-" * 80)
#     logger.info("ALIASING PROCESS")
#     logger.info("-" * 80)
    
#     if name_col not in df.columns:
#         logger.error("✗ Column '%s' NOT FOUND in dataframe!", name_col)
#         logger.error("Available columns: %s", list(df.columns))
#         raise HTTPException(status_code=500, detail=f"Expected column '{name_col}' not found in data")
    
#     logger.info("✓ Column '%s' exists in dataframe", name_col)
    
#     try:
#         # Get unique customer names
#         raw_customers = df[name_col].dropna().unique()
#         logger.info("✓ Found %d unique customer values (before processing)", len(raw_customers))
        
#         # Clean and sort
#         unique_customers = [str(x).strip() for x in raw_customers if str(x).strip()]
#         unique_customers.sort(key=lambda x: -len(x))  # longest first
        
#         logger.info("✓ After cleaning: %d valid customers", len(unique_customers))
#         logger.info("✓ Sample customers (top 5): %s", unique_customers[:5])
        
#         # Generate aliases for each customer (run in parallel where possible)
#         tasks = []
#         for idx, cust in enumerate(unique_customers):
#             tasks.append(process_customer_async(project_name, cust, idx + 1, len(unique_customers)))
        
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         for result in results:
#             if isinstance(result, Exception):
#                 logger.exception("Error processing customer: %s", result)
#                 continue
#             if result:
#                 cust, alias = result
#                 mapping[cust] = alias
        
#         logger.info("")
#         logger.info("-" * 80)
#         logger.info("FINAL MAPPING (%d entries):", len(mapping))
#         for real, alias in list(mapping.items())[:10]:  # Show first 10
#             logger.info("  '%s' -> '%s'", real, alias)
#         if len(mapping) > 10:
#             logger.info("  ... and %d more", len(mapping) - 10)
#         logger.info("-" * 80)
        
#     except Exception as e:
#         logger.exception("✗ Error building alias mapping: %s", e)
#         raise HTTPException(status_code=500, detail=f"Failed to build aliases: {e}")
    
#     return mapping


# async def compute_single_customer_stats(df_alias: pd.DataFrame, name_col: str, alias: str, 
#                                        amount_col: str, payment_date_col: str, invoice_date_col: str, 
#                                        moving_avg_window: int) -> dict:
#     """Compute stats for single customer (CPU intensive -> thread)"""
#     def _compute():
#         # Filter data for this customer
#         customer_df = df_alias[df_alias[name_col] == alias].copy()
        
#         # Filter only paid invoices (where payment_date is not null)
#         if payment_date_col:
#             paid_df = customer_df[customer_df[payment_date_col].notna()].copy()
#         else:
#             paid_df = customer_df.copy()
        
#         if len(paid_df) == 0:
#             logger.info("Customer '%s': No paid invoices found", alias)
#             return {}
        
#         stats = {
#             'total_invoices': len(paid_df),
#             'mean_amount': 0,
#             'moving_avg_amount': 0,
#             'moving_avg_window': 0,
#             'total_paid': 0,
#             'avg_delay_days': 0,
#             'last_payment_date': None
#         }
        
#         # Calculate amount statistics
#         if amount_col and amount_col in paid_df.columns:
#             # Convert to numeric, handling any string formatting
#             amounts = pd.to_numeric(paid_df[amount_col], errors='coerce')
#             amounts = amounts.dropna()
            
#             if len(amounts) > 0:
#                 # Overall mean
#                 stats['mean_amount'] = float(amounts.mean())
#                 stats['total_paid'] = float(amounts.sum())
                
#                 # Moving average: take last N invoices (sorted by date if possible)
#                 if payment_date_col and payment_date_col in paid_df.columns:
#                     # Sort by payment date to get most recent invoices
#                     paid_df_sorted = paid_df.sort_values(by=payment_date_col, ascending=False)
#                     recent_amounts = pd.to_numeric(paid_df_sorted[amount_col].head(moving_avg_window), errors='coerce')
#                     recent_amounts = recent_amounts.dropna()
#                 else:
#                     # If no date column, just take last N rows
#                     recent_amounts = amounts.tail(moving_avg_window)
                
#                 if len(recent_amounts) > 0:
#                     stats['moving_avg_amount'] = float(recent_amounts.mean())
#                     stats['moving_avg_window'] = len(recent_amounts)
        
#         # Calculate delay statistics
#         if payment_date_col and invoice_date_col:
#             try:
#                 paid_df[payment_date_col] = pd.to_datetime(paid_df[payment_date_col], errors='coerce')
#                 paid_df[invoice_date_col] = pd.to_datetime(paid_df[invoice_date_col], errors='coerce')
                
#                 # Calculate delays
#                 delays = (paid_df[payment_date_col] - paid_df[invoice_date_col]).dt.days
#                 delays = delays.dropna()
                
#                 if len(delays) > 0:
#                     stats['avg_delay_days'] = float(delays.mean())
                
#                 # Get last payment date
#                 last_payment = paid_df[payment_date_col].max()
#                 if pd.notna(last_payment):
#                     stats['last_payment_date'] = last_payment.strftime('%Y-%m-%d')
                    
#             except Exception as e:
#                 logger.warning("Failed to calculate dates for '%s': %s", alias, e)
        
#         return {'alias': alias, 'stats': stats}
    
#     return await asyncio.to_thread(_compute)


# async def compute_customer_stats(df_alias: pd.DataFrame, name_col: str) -> dict:
#     """Async helper to compute customer statistics"""
#     logger.info("")
#     logger.info("PRECOMPUTING CUSTOMER STATISTICS")
#     logger.info("-" * 80)
    
#     # Moving average window size (number of recent invoices to consider)
#     MOVING_AVG_WINDOW = int(os.getenv("MOVING_AVG_WINDOW", "3"))
#     logger.info("Using moving average window: %d invoices", MOVING_AVG_WINDOW)
    
#     customer_stats = {}
    
#     try:
#         # Identify amount column (could be net_amount or total_invoice_amount)
#         amount_col = None
#         for col in ['total_invoice_amount', 'net_amount', 'amount']:
#             if col in df_alias.columns:
#                 amount_col = col
#                 break
        
#         if not amount_col:
#             logger.warning("⚠ No amount column found, skipping statistics computation")
        
#         # Identify date columns
#         payment_date_col = 'payment_date' if 'payment_date' in df_alias.columns else None
#         invoice_date_col = 'invoice_date' if 'invoice_date' in df_alias.columns else None
#         due_date_col = 'due_date' if 'due_date' in df_alias.columns else None
        
#         logger.info("Using columns: amount=%s, payment_date=%s, invoice_date=%s", 
#                    amount_col, payment_date_col, invoice_date_col)
        
#         # Get unique customers from aliased dataframe
#         unique_aliases = df_alias[name_col].dropna().unique()
        
#         # Process customers in parallel
#         tasks = []
#         for alias in unique_aliases:
#             tasks.append(compute_single_customer_stats(df_alias, name_col, alias, amount_col, payment_date_col, invoice_date_col, MOVING_AVG_WINDOW))
        
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         for result in results:
#             if isinstance(result, dict) and 'alias' in result:
#                 alias = result['alias']
#                 stats = result['stats']
#                 customer_stats[alias] = stats
#                 logger.info("Customer '%s': %d paid invoices, mean=%.2f, moving_avg=%.2f (window=%d), total=%.2f, avg_delay=%.1f days", 
#                            alias, stats['total_invoices'], stats['mean_amount'], 
#                            stats['moving_avg_amount'], stats['moving_avg_window'],
#                            stats['total_paid'], stats['avg_delay_days'])
        
#         logger.info("-" * 80)
#         logger.info("Computed statistics for %d customers", len(customer_stats))
        
#     except Exception as e:
#         logger.exception("✗ Failed to compute customer statistics: %s", e)
#         customer_stats = {}
    
#     return customer_stats


# def format_customer_stats(customer_stats: dict) -> str:
#     """Format customer statistics for prompt inclusion"""
#     if not customer_stats:
#         return ""
    
#     stats_lines = []
#     for alias, stats in customer_stats.items():
#         stats_lines.append(f"Customer: {alias}")
#         stats_lines.append(f"  - Total paid invoices: {stats['total_invoices']}")
#         stats_lines.append(f"  - Overall mean amount: ₹{stats['mean_amount']:.2f}")
#         stats_lines.append(f"  - Moving average amount (last {stats['moving_avg_window']} invoices): ₹{stats['moving_avg_amount']:.2f}")
#         stats_lines.append(f"  - Total amount paid: ₹{stats['total_paid']:.2f}")
#         stats_lines.append(f"  - Average payment delay: {stats['avg_delay_days']:.1f} days")
#         if stats['last_payment_date']:
#             stats_lines.append(f"  - Last payment date: {stats['last_payment_date']}")
#         stats_lines.append("")
    
#     return "\n".join(stats_lines)


# @router.post("/query")
# async def query_project(req: QueryRequest):
#     # Run sync init_db in thread pool
#     await asyncio.to_thread(init_db)
#     logger.info("=" * 80)
#     logger.info("QUERY REQUEST START")
#     logger.info("Project: %s", req.project_name)
#     logger.info("Query: %s", req.query)
#     logger.info("=" * 80)

#     # 1. Get S3 URL (sync -> thread)
#     s3_url = await asyncio.to_thread(get_project_url, req.project_name)
#     if not s3_url:
#         logger.error("Project not found in DB: %s", req.project_name)
#         raise HTTPException(status_code=404, detail="project not found")
#     logger.info("✓ Found S3 URL: %s", s3_url)

#     # 2. Download file (async)
#     try:
#         data_bytes = await asyncio.to_thread(download_to_bytes, s3_url)
#         logger.info("✓ Downloaded %d bytes from S3", len(data_bytes) if data_bytes else 0)
#     except Exception as e:
#         logger.exception("✗ Failed to download from S3: %s", e)
#         raise HTTPException(status_code=500, detail=f"failed to download from s3: {e}")

#     # FIX: Use the extract_filename_from_url function to properly handle pre-signed URLs
#     filename = extract_filename_from_url(s3_url)
#     logger.info("✓ Filename: %s", filename)

#     # 3. Parse file (async)
#     try:
#         df = await asyncio.to_thread(read_input_file, io.BytesIO(data_bytes), filename)
#         logger.info("✓ Parsed dataframe: shape=%s", df.shape)
#         logger.info("✓ Columns: %s", list(df.columns))
#         logger.info("✓ First 3 rows:\n%s", df.head(3).to_string(index=False))
#     except Exception as e:
#         logger.exception("✗ Failed to parse file: %s", e)
#         raise HTTPException(status_code=500, detail=f"failed to parse file: {e}")

#     # 4. Build alias mapping (async)
#     name_col = "customer_name"
#     mapping = await build_alias_mapping(req.project_name, df, name_col)

#     if not mapping:
#         logger.warning("⚠ WARNING: No aliases created! This means no customer names will be anonymized.")

#     # 5. Apply aliases to dataframe
#     logger.info("")
#     logger.info("APPLYING ALIASES TO DATAFRAME")
#     logger.info("-" * 80)
    
#     df_alias = df.copy()
    
#     if mapping:
#         try:
#             # Show before aliasing
#             logger.info("Before aliasing (sample):")
#             logger.info("%s", df_alias[name_col].head(5).tolist())
            
#             # Apply mapping
#             df_alias[name_col] = df_alias[name_col].astype(str).str.strip().map(mapping)
            
#             # Show after aliasing
#             logger.info("After aliasing (sample):")
#             logger.info("%s", df_alias[name_col].head(5).tolist())
            
#             # Check for unmapped values (will be NaN)
#             unmapped_count = df_alias[name_col].isna().sum()
#             if unmapped_count > 0:
#                 logger.warning("⚠ WARNING: %d rows have unmapped customer names (will be NaN)", unmapped_count)
#             else:
#                 logger.info("✓ All customer names successfully aliased")
                
#         except Exception as e:
#             logger.exception("✗ Failed to apply aliases to dataframe: %s", e)
#             raise HTTPException(status_code=500, detail=f"Failed to apply aliases: {e}")
#     else:
#         logger.warning("⚠ Skipping dataframe aliasing (no mapping available)")

#     # 6. Transform query with aliases
#     logger.info("")
#     logger.info("TRANSFORMING QUERY")
#     logger.info("-" * 80)
#     logger.info("Original query: %s", req.query)
    
#     transformed_query = req.query
#     replacements_made = []
    
#     if mapping:
#         for cust, alias in mapping.items():
#             if cust and cust in transformed_query:
#                 transformed_query = transformed_query.replace(cust, alias)
#                 replacements_made.append(f"'{cust}' -> '{alias}'")
        
#         if replacements_made:
#             logger.info("✓ Query transformations:")
#             for repl in replacements_made:
#                 logger.info("  %s", repl)
#             logger.info("Transformed query: %s", transformed_query)
#         else:
#             logger.info("⚠ No customer names found in query to replace")
#     else:
#         logger.warning("⚠ Skipping query transformation (no mapping available)")

#     # 7. Precompute statistics for each customer (async)
#     customer_stats = await compute_customer_stats(df_alias, name_col)

#     # 8. Convert to CSV for LLM
#     try:
#         data_text = await asyncio.to_thread(df_alias.to_csv, index=False)
#         logger.info("✓ Converted dataframe to CSV (%d characters)", len(data_text))
#         logger.info("CSV preview (first 500 chars):\n%s", data_text[:500])
#     except Exception as e:
#         logger.exception("✗ Failed to convert dataframe to CSV: %s", e)
#         raise HTTPException(status_code=500, detail=f"Failed to convert to CSV: {e}")

#     # 9. Build LLM prompt using prompts_loader
#     logger.info("")
#     logger.info("BUILDING LLM PROMPT FROM TEMPLATE")
#     logger.info("-" * 80)
    
#     try:
#         # Format customer statistics
#         formatted_stats = format_customer_stats(customer_stats)
        
#         # Use statistics template to wrap the stats
#         stats_text = format_prompt(
#             "statistics_template",
#             customer_stats=formatted_stats
#         )
        
#         # Build the full prompt using the main query template
#         prompt = format_prompt(
#             "llm_query_prompt",
#             data_text=data_text,
#             stats_text=stats_text,
#             transformed_query=transformed_query
#         )
        
#         logger.info("✓ Built LLM prompt from template (%d characters)", len(prompt))
        
#     except Exception as e:
#         logger.exception("✗ Failed to build prompt from template: %s", e)
#         raise HTTPException(status_code=500, detail=f"Failed to build prompt: {e}")

#     # 10. Call LLM (async)
#     logger.info("")
#     logger.info("CALLING LLM")
#     logger.info("-" * 80)
    
#     try:
#         llm_response = await lite_client.async_generate(prompt)
#         logger.info("✓ LLM responded (%d characters)", len(llm_response))
#         logger.info("LLM response preview:\n%s", llm_response[:500])
#     except Exception as e:
#         logger.exception("✗ LLM call failed: %s", e)
#         raise HTTPException(status_code=500, detail=f"LLM error: {e}")

#     # 11. DE-ALIAS the response
#     logger.info("")
#     logger.info("DE-ALIASING RESPONSE")
#     logger.info("-" * 80)
    
#     dealiased_response = llm_response
#     dealias_replacements = []
    
#     if mapping:
#         # Create reverse mapping: alias -> real_name
#         reverse_mapping = {alias: real_name for real_name, alias in mapping.items()}
#         logger.info("Created reverse mapping (%d entries)", len(reverse_mapping))
        
#         # Sort by length (longest first) to avoid substring issues
#         sorted_aliases = sorted(reverse_mapping.keys(), key=lambda x: -len(x))
        
#         for alias in sorted_aliases:
#             real_name = reverse_mapping[alias]
#             if alias in dealiased_response:
#                 dealiased_response = dealiased_response.replace(alias, real_name)
#                 dealias_replacements.append(f"'{alias}' -> '{real_name}'")
        
#         if dealias_replacements:
#             logger.info("✓ De-aliasing transformations:")
#             for repl in dealias_replacements[:10]:  # Show first 10
#                 logger.info("  %s", repl)
#             if len(dealias_replacements) > 10:
#                 logger.info("  ... and %d more", len(dealias_replacements) - 10)
#         else:
#             logger.info("⚠ No aliases found in LLM response to replace")
            
#         logger.info("De-aliased response preview:\n%s", dealiased_response[:500])
#     else:
#         logger.warning("⚠ Skipping de-aliasing (no mapping available)")

#     # 12. Extract clean answer
#     clean_answer = extract_clean_answer(dealiased_response)
#     logger.info("")
#     logger.info("EXTRACTED CLEAN ANSWER:")
#     logger.info("%s", clean_answer[:300])

#     logger.info("")
#     logger.info("=" * 80)
#     logger.info("QUERY REQUEST COMPLETE")
#     logger.info("=" * 80)

#     # Return only the clean answer
#     return {
#         "answer": clean_answer
#     }




from fastapi import APIRouter, HTTPException, Form, Request
from src.utils.db import get_project_url, init_db, get_alias
from src.utils.s3_utils import download_to_bytes, extract_filename_from_url
from src.utils.normalize import read_input_file
from src.utils.aliaser import make_alias
from src.utils.prompts_loader import format_prompt
from src.llm.lite_client import lite_client
from config.settings import MOVING_AVG_WINDOW
import pandas as pd
import io
import logging
import traceback
import re
import os
import asyncio
from config.config import get_model_config
import json
import requests
import litellm
from src.utils.obs import LLMUsageTracker




logging.basicConfig(
    level=logging.INFO,
    filemode="a",
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)


router = APIRouter()


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


async def process_customer_async(project_name: str, cust: str, idx: int, total: int) -> tuple:
    """Helper to process single customer asynchronously"""
    logger.info("")
    logger.info("Processing customer %d/%d: '%s'", idx, total, cust)
    
    # Check if alias already exists (sync -> thread)
    alias = await asyncio.to_thread(get_alias, project_name, cust)
    
    if alias:
        logger.info("  ✓ Found existing alias: '%s' -> '%s'", cust, alias)
        return (cust, alias)
    else:
        logger.info("  ⚙ No existing alias, generating new one...")
        try:
            # FIXED: await the make_alias coroutine
            alias = await make_alias(project_name, cust)
            logger.info("  ✓ Generated new alias: '%s' -> '%s'", cust, alias)
            return (cust, alias)
        except Exception as ex_alias:
            logger.exception("  ✗ Failed to create alias for '%s': %s", cust, ex_alias)
            return None


async def build_alias_mapping(project_name: str, df: pd.DataFrame, name_col: str) -> dict:
    """Async helper to build alias mapping"""
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
        
        # Generate aliases for each customer (run in parallel where possible)
        tasks = []
        for idx, cust in enumerate(unique_customers):
            tasks.append(process_customer_async(project_name, cust, idx + 1, len(unique_customers)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.exception("Error processing customer: %s", result)
                continue
            if result:
                cust, alias = result
                mapping[cust] = alias
        
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
    
    return mapping


async def compute_single_customer_stats(df_alias: pd.DataFrame, name_col: str, alias: str, 
                                       amount_col: str, payment_date_col: str, invoice_date_col: str, 
                                       moving_avg_window: int) -> dict:
    """Compute stats for single customer (CPU intensive -> thread)"""
    def _compute():
        # Filter data for this customer
        customer_df = df_alias[df_alias[name_col] == alias].copy()
        
        # Filter only paid invoices (where payment_date is not null)
        if payment_date_col:
            paid_df = customer_df[customer_df[payment_date_col].notna()].copy()
        else:
            paid_df = customer_df.copy()
        
        if len(paid_df) == 0:
            logger.info("Customer '%s': No paid invoices found", alias)
            return {}
        
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
                    recent_amounts = pd.to_numeric(paid_df_sorted[amount_col].head(moving_avg_window), errors='coerce')
                    recent_amounts = recent_amounts.dropna()
                else:
                    # If no date column, just take last N rows
                    recent_amounts = amounts.tail(moving_avg_window)
                
                if len(recent_amounts) > 0:
                    stats['moving_avg_amount'] = float(recent_amounts.mean())
                    stats['moving_avg_window'] = len(recent_amounts)
        
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
        
        return {'alias': alias, 'stats': stats}
    
    return await asyncio.to_thread(_compute)


async def compute_customer_stats(df_alias: pd.DataFrame, name_col: str) -> dict:
    """Async helper to compute customer statistics"""
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
        
        # Process customers in parallel
        tasks = []
        for alias in unique_aliases:
            tasks.append(compute_single_customer_stats(df_alias, name_col, alias, amount_col, payment_date_col, invoice_date_col, MOVING_AVG_WINDOW))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and 'alias' in result:
                alias = result['alias']
                stats = result['stats']
                customer_stats[alias] = stats
                logger.info("Customer '%s': %d paid invoices, mean=%.2f, moving_avg=%.2f (window=%d), total=%.2f, avg_delay=%.1f days", 
                           alias, stats['total_invoices'], stats['mean_amount'], 
                           stats['moving_avg_amount'], stats['moving_avg_window'],
                           stats['total_paid'], stats['avg_delay_days'])
        
        logger.info("-" * 80)
        logger.info("Computed statistics for %d customers", len(customer_stats))
        
    except Exception as e:
        logger.exception("✗ Failed to compute customer statistics: %s", e)
        customer_stats = {}
    
    return customer_stats


def format_customer_stats(customer_stats: dict) -> str:
    """Format customer statistics for prompt inclusion"""
    if not customer_stats:
        return ""
    
    stats_lines = []
    for alias, stats in customer_stats.items():
        stats_lines.append(f"Customer: {alias}")
        stats_lines.append(f"  - Total paid invoices: {stats['total_invoices']}")
        stats_lines.append(f"  - Overall mean amount: ₹{stats['mean_amount']:.2f}")
        stats_lines.append(f"  - Moving average amount (last {stats['moving_avg_window']} invoices): ₹{stats['moving_avg_amount']:.2f}")
        stats_lines.append(f"  - Total amount paid: ₹{stats['total_paid']:.2f}")
        stats_lines.append(f"  - Average payment delay: {stats['avg_delay_days']:.1f} days")
        if stats['last_payment_date']:
            stats_lines.append(f"  - Last payment date: {stats['last_payment_date']}")
        stats_lines.append("")
    
    return "\n".join(stats_lines)


@router.post("/query")
async def query_project(
    req: Request,
    project_name: str = Form(...),
    query: str = Form(...),
    user_metadata: str = Form(...)
):
    """
    Query endpoint that accepts form data instead of JSON.
    
    Usage with curl:
    curl -X POST "http://localhost:8000/query" \
         -F "project_name=my_project" \
         -F "query=What is the total revenue?" \
         -F "metadata=some_metadata"
    
    Usage with Python requests:
    import requests
    response = requests.post(
        "http://localhost:8000/query",
        data={
            "project_name": "my_project",
            "query": "What is the total revenue?",
            "metadata": "some_metadata"
        }
    )
    """
    
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
            auth_token = req.headers.get("Authorization")
            if auth_token:
                llm_params.update({"auth_token": auth_token})
    except Exception as e:
        logging.error(f"Error extracting attendees: {str(e)}")
        return []
    # Run sync init_db in thread pool
    await asyncio.to_thread(init_db)
    logger.info("=" * 80)
    logger.info("QUERY REQUEST START")
    logger.info("Project: %s", project_name)
    logger.info("Query: %s", query)
    logger.info("=" * 80)

    # 1. Get S3 URL (sync -> thread)
    s3_url = await asyncio.to_thread(get_project_url, project_name)
    if not s3_url:
        logger.error("Project not found in DB: %s", project_name)
        raise HTTPException(status_code=404, detail="project not found")
    logger.info("✓ Found S3 URL: %s", s3_url)

    # 2. Download file (async)
    try:
        data_bytes = await asyncio.to_thread(download_to_bytes, s3_url)
        logger.info("✓ Downloaded %d bytes from S3", len(data_bytes) if data_bytes else 0)
    except Exception as e:
        logger.exception("✗ Failed to download from S3: %s", e)
        raise HTTPException(status_code=500, detail=f"failed to download from s3: {e}")

    # FIX: Use the extract_filename_from_url function to properly handle pre-signed URLs
    filename = extract_filename_from_url(s3_url)
    logger.info("✓ Filename: %s", filename)

    # 3. Parse file (async)
    try:
        df = await asyncio.to_thread(read_input_file, io.BytesIO(data_bytes), filename)
        logger.info("✓ Parsed dataframe: shape=%s", df.shape)
        logger.info("✓ Columns: %s", list(df.columns))
        logger.info("✓ First 3 rows:\n%s", df.head(3).to_string(index=False))
    except Exception as e:
        logger.exception("✗ Failed to parse file: %s", e)
        raise HTTPException(status_code=500, detail=f"failed to parse file: {e}")

    # 4. Build alias mapping (async)
    name_col = "customer_name"
    mapping = await build_alias_mapping(project_name, df, name_col)

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
    logger.info("Original query: %s", query)
    
    transformed_query = query
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

    # 7. Precompute statistics for each customer (async)
    customer_stats = await compute_customer_stats(df_alias, name_col)

    # 8. Convert to CSV for LLM
    try:
        data_text = await asyncio.to_thread(df_alias.to_csv, index=False)
        logger.info("✓ Converted dataframe to CSV (%d characters)", len(data_text))
        logger.info("CSV preview (first 500 chars):\n%s", data_text[:500])
    except Exception as e:
        logger.exception("✗ Failed to convert dataframe to CSV: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to convert to CSV: {e}")

    # 9. Build LLM prompt using prompts_loader
    logger.info("")
    logger.info("BUILDING LLM PROMPT FROM TEMPLATE")
    logger.info("-" * 80)
    
    try:
        # Format customer statistics
        formatted_stats = format_customer_stats(customer_stats)
        
        # Use statistics template to wrap the stats
        stats_text = format_prompt(
            "statistics_template",
            customer_stats=formatted_stats
        )
        
        # Build the full prompt using the main query template
        prompt = format_prompt(
            "llm_query_prompt",
            data_text=data_text,
            stats_text=stats_text,
            transformed_query=transformed_query
        )
        
        logger.info("✓ Built LLM prompt from template (%d characters)", len(prompt))
        
    except Exception as e:
        logger.exception("✗ Failed to build prompt from template: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to build prompt: {e}")

    # 10. Call LLM (async)
    logger.info("")
    logger.info("CALLING LLM")
    logger.info("-" * 80)
    
    try:
        auth_token = llm_params.pop("auth_token", "")
        llm_response = litellm.completion(
            **llm_params,
            messages=[{"role": "user", "content": prompt}],
        )

        # print("llm_response", llm_response)
        token_tracker = LLMUsageTracker()
        token_tracker.track_response(response=llm_response, auth_token=auth_token, model=llm_params.get("model", ""))
        response_text = llm_response.choices[0].message.content.strip()
        print("response_text", response_text)
        
        # token_tracker.track_response(response=response, auth_token=self.auth_token, model=llm_params.get("model", ""))
        
        # llm_response = await lite_client.async_generate(prompt)
        logger.info("✓ LLM responded (%d characters)", llm_response)
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
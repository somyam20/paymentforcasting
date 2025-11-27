import pandas as pd
from config.settings import EXPECTED_COLS
import logging

logger = logging.getLogger(__name__)

def normalize_columns(df):
    """
    Normalize column names in dataframe using EXPECTED_COLS mapping.
    """
    col_map = {}
    
    logger.info("=" * 80)
    logger.info("NORMALIZING COLUMNS")
    logger.info("Raw columns from file: %s", list(df.columns))
    logger.info("=" * 80)
    
    for c in df.columns:
        lc = str(c).lower().strip()
        c_upper = str(c).strip().upper()
        c_stripped = str(c).strip()
        
        # CUSTOMER NAME mapping (exact match)
        if c_stripped == "CUSTOMER NAME":
            col_map[c] = EXPECTED_COLS["name"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["name"])
        
        # ENTITY LEGAL NAME
        elif c_stripped == "ENTITY LEGAL NAME":
            col_map[c] = EXPECTED_COLS["entity_legal"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["entity_legal"])
        
        # HOLDING COMPANY NAME
        elif c_stripped == "HOLDING COMPANY NAME":
            col_map[c] = EXPECTED_COLS["holding"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["holding"])
        
        # INVOICE DATE
        elif "INVOICE" in c_upper and "DATE" in c_upper:
            col_map[c] = EXPECTED_COLS["date"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["date"])
        
        # SALES ORDER NUMBER
        elif "SALES ORDER" in c_upper and "NUMBER" in c_upper:
            col_map[c] = EXPECTED_COLS["sales_order"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["sales_order"])
        
        # INVOICE NUMBER
        elif c_stripped == "INVOICE NUMBER":
            col_map[c] = EXPECTED_COLS["invoice"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["invoice"])

        # PAY TERMS
        elif c_stripped == "PAY TERMS":
            col_map[c] = EXPECTED_COLS["pat_terms"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["pat_terms"])

        # DUE DATE
        elif c_stripped == "DUE DATE":
            col_map[c] = EXPECTED_COLS["due_date"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["due_date"])

        # SALES MANAGER NAME
        elif "SALES" in c_upper and "MANAGER" in c_upper:
            col_map[c] = EXPECTED_COLS["salesperson"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["salesperson"])

        # SERVICE TYPE
        elif c_stripped == "SERVICE TYPE":
            col_map[c] = EXPECTED_COLS["service_type"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["service_type"])

        # LOCATION
        elif c_stripped == "LOCATION":
            col_map[c] = EXPECTED_COLS["location"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["location"])

        # NET AMOUNT
        elif "NET" in c_upper and "AMOUNT" in c_upper:
            col_map[c] = EXPECTED_COLS["amt"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["amt"])

        # VAT
        elif c_stripped == "VAT (5%)":
            col_map[c] = EXPECTED_COLS["vat"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["vat"])

        # TOTAL INVOICE AMT
        elif "TOTAL" in c_upper and "INVOICE" in c_upper and "AMT" in c_upper:
            col_map[c] = EXPECTED_COLS["total_invoice"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["total_invoice"])

        # PAYMENT DATE
        elif c_stripped == "PAYMENT DATE":
            col_map[c] = EXPECTED_COLS["receipt"]
            logger.info("✓ Mapped '%s' -> '%s'", c, EXPECTED_COLS["receipt"])

    logger.info("-" * 80)
    logger.info("Total mappings created: %d", len(col_map))
    logger.info("-" * 80)
    
    df_renamed = df.rename(columns=col_map)

    # Fallback: If customer_name doesn't exist or is mostly null, try ENTITY LEGAL NAME
    if EXPECTED_COLS["name"] not in df_renamed.columns:
        logger.warning("⚠ 'customer_name' column not found after mapping!")
        if EXPECTED_COLS["entity_legal"] in df_renamed.columns:
            logger.info("→ Falling back to 'entity_legal_name' as customer_name")
            df_renamed[EXPECTED_COLS["name"]] = df_renamed[EXPECTED_COLS["entity_legal"]]
        else:
            logger.error("✗ No suitable column found for customer names!")
    else:
        # Check if customer_name is mostly null
        customer_null_count = df_renamed[EXPECTED_COLS["name"]].isna().sum()
        null_percentage = (customer_null_count / len(df_renamed)) * 100
        
        if null_percentage > 50:
            logger.warning("⚠ 'customer_name' is %.1f%% null", null_percentage)
            if EXPECTED_COLS["entity_legal"] in df_renamed.columns:
                logger.info("→ Falling back to 'entity_legal_name'")
                df_renamed[EXPECTED_COLS["name"]] = df_renamed[EXPECTED_COLS["entity_legal"]]

    logger.info("Final normalized columns: %s", list(df_renamed.columns))
    logger.info("=" * 80)
    
    return df_renamed

def read_input_file(file_obj, filename: str):
    """
    Read Excel or CSV file uploaded by user.
    Excel: header on row 3 -> header=2, usecols from column B onward (we drop the first column if unnamed)
    CSV: read normally and then normalize columns
    Returns: pandas.DataFrame
    """
    logger.info("Reading file: %s", filename)
    
    lower = filename.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        # read header from row 3 (0-based index 2)
        df = pd.read_excel(file_obj, header=2)
        logger.info("✓ Read Excel file with shape: %s", df.shape)
        
        # If there is an unnamed left-most column (common when sheet has offset), drop it
        if df.columns[0].startswith("Unnamed"):
            logger.info("→ Dropping unnamed first column")
            df = df.iloc[:, 1:]
    elif lower.endswith(".csv"):
        df = pd.read_csv(file_obj)
        logger.info("✓ Read CSV file with shape: %s", df.shape)
    else:
        raise ValueError("Unsupported file type")

    logger.info("Raw columns: %s", list(df.columns))
    df = normalize_columns(df)
    
    return df
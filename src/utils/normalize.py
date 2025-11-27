import pandas as pd
from config.settings import EXPECTED_COLS

def normalize_columns(df):
    """
    Normalize column names in dataframe using EXPECTED_COLS mapping.
    Assumes EXPECTED_COLS is defined elsewhere in the project.
    """
    col_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if str(c).strip().upper() == "INVOICE NUMBER":
            col_map[c] = EXPECTED_COLS["invoice"]

        elif str(c).strip().upper() == "PAY TERMS":
            col_map[c] = EXPECTED_COLS["pat_terms"]

        elif str(c).strip().upper() == "DUE DATE":
            col_map[c] = EXPECTED_COLS["due_date"]

        elif "sales manager" in lc:
            col_map[c] = EXPECTED_COLS["salesperson"]

        elif str(c).strip().upper() == "SERVICE TYPE":
            col_map[c] = EXPECTED_COLS["service_type"]

        elif str(c).strip().upper() == "LOCATION":
            col_map[c] = EXPECTED_COLS["location"]

        elif "net amount" in lc:
            col_map[c] = EXPECTED_COLS["amt"]

        elif "vat" in lc:
            col_map[c] = EXPECTED_COLS["vat"]

        elif "total invoice" in lc:
            col_map[c] = EXPECTED_COLS["total_invoice"]

        elif str(c).strip().upper() == "PAYMENT DATE":
            col_map[c] = EXPECTED_COLS["receipt"]

    df_renamed = df.rename(columns=col_map)

    # If customer_name exists but mostly null, fallback to ENTITY column if present
    if EXPECTED_COLS["name"] in df_renamed.columns and "ENTITY" in df.columns:
        customer_null_count = df_renamed[EXPECTED_COLS["name"]].isna().sum()
        if customer_null_count > len(df_renamed) * 0.5:
            if "ENTITY" in df.columns and EXPECTED_COLS["name"] not in df.columns:
                df_renamed[EXPECTED_COLS["name"]] = df["ENTITY"]

    return df_renamed

def read_input_file(file_obj, filename: str):
    """
    Read Excel or CSV file uploaded by user.
    Excel: header on row 3 -> header=2, usecols from column B onward (we drop the first column if unnamed)
    CSV: read normally and then normalize columns
    Returns: pandas.DataFrame
    """
    lower = filename.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        # read header from row 3 (0-based index 2)
        df = pd.read_excel(file_obj, header=2)
        # If there is an unnamed left-most column (common when sheet has offset), drop it
        if df.columns[0].startswith("Unnamed"):
            df = df.iloc[:, 1:]
    elif lower.endswith(".csv"):
        df = pd.read_csv(file_obj)
    else:
        raise ValueError("Unsupported file type")

    df = normalize_columns(df)
    return df
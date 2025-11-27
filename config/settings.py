import os
from dotenv import load_dotenv
load_dotenv()

# Expected canonical column names (must match your Excel structure)
EXPECTED_COLS = {
    "name": "customer_name",                    # CUSTOMER NAME
    "entity_legal": "entity_legal_name",        # ENTITY LEGAL NAME
    "holding": "holding_company_name",          # HOLDING COMPANY NAME
    "date": "invoice_date",                     # INVOICE DATE
    "sales_order": "sales_order_number",        # SALES ORDER NUMBER
    "invoice": "invoice_number",                # INVOICE NUMBER
    "pat_terms": "pay_terms",                   # PAY TERMS
    "due_date": "due_date",                     # DUE DATE
    "salesperson": "sales_manager_name",        # SALES MANAGER NAME
    "service_type": "service_type",             # SERVICE TYPE
    "location": "location",                     # LOCATION
    "amt": "net_amount",                        # NET AMOUNT (...)
    "vat": "vat",                               # VAT (5%)
    "total_invoice": "total_invoice_amount",    # TOTAL INVOICE AMT (...)
    "receipt": "payment_date",                  # PAYMENT DATE
}


# DB and AWS configuration (pull from env)
DATABASE_URL = os.getenv("DATABASE_URL")  # example: postgres://user:pass@host:5432/db
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# Lite LLM / model configuration
LITE_LLM_API_KEY = os.getenv("LITE_LLM_API_KEY")
LITE_LLM_MODEL = os.getenv("LITE_LLM_MODEL", "gemini/gemini-2.5-flash-lite")


# Upload settings
S3_UPLOAD_PREFIX = os.getenv("S3_UPLOAD_PREFIX", "projects/")


# Sanitization / alias config
ALIAS_SALT = os.getenv("ALIAS_SALT", "change_this_in_prod")
ALIAS_LENGTH = int(os.getenv("ALIAS_LENGTH", 8))
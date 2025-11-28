Financial Data Analytics Platform
A privacy-focused financial analytics platform that provides intelligent insights on invoice data using AI, with built-in customer name anonymization and secure data handling.
🌟 Features

📊 Intelligent Data Analysis: AI-powered insights on payment patterns, invoice forecasting, and customer behavior
🔒 Privacy-First Design: Automatic customer name aliasing to protect sensitive information
📈 Smart Forecasting: Moving average-based predictions for invoice amounts and payment dates
☁️ S3 Integration: Seamless file storage and retrieval from AWS S3
🤖 LLM-Powered Insights: Natural language responses using LiteLLM with support for multiple AI models
📁 Multi-Format Support: Handles both Excel (.xlsx, .xls) and CSV files
🗃️ PostgreSQL Backend: Robust data persistence for projects and customer aliases

🏗️ Architecture
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Routes │
    ├─────────┤
    │ upload  │  ← Upload S3 URLs for projects
    │ query   │  ← Ask questions about data
    └────┬────┘
         │
    ┌────┴──────────────────┐
    │                       │
┌───┴────┐          ┌──────┴──────┐
│   S3   │          │  PostgreSQL │
│ Storage│          │  Database   │
└────────┘          └─────────────┘
                           │
                    ┌──────┴──────┐
                    │   Tables    │
                    ├─────────────┤
                    │  projects   │
                    │   aliases   │
                    └─────────────┘
🚀 Getting Started
Prerequisites

Python 3.9+
PostgreSQL database
AWS account with S3 access
LiteLLM API key (supports various AI providers)

Installation

Clone the repository

bash   git clone <repository-url>
   cd generative_ai_project

Create virtual environment

bash   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bash   pip install -r requirements.txt

Set up environment variables
Create a .env file in the root directory:

env   # Database Configuration
   DB_HOST=localhost
   DB_NAME=financial_analytics
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_PORT=5432
   
   # AWS Configuration
   AWS_S3_BUCKET=your-bucket-name
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   
   # LiteLLM Configuration
   LITE_LLM_API_KEY=your_api_key
   LITE_LLM_MODEL=gemini/gemini-2.5-flash-lite
   
   # Aliasing Configuration
   ALIAS_SALT=your_random_salt_string
   ALIAS_LENGTH=8
   
   # Analytics Configuration
   MOVING_AVG_WINDOW=3
   
   # S3 Upload Configuration
   S3_UPLOAD_PREFIX=projects/

Initialize the database

bash   # The database tables will be auto-created on first run
   # Or manually run:
   python -c "from src.utils.db import init_db; init_db()"
Running the Application
bashuvicorn main:app --reload --host 0.0.0.0 --port 8000
The API will be available at http://localhost:8000
API documentation at http://localhost:8000/docs
📝 API Usage
1. Upload Project
Register an S3 URL with a project name:
bashcurl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "s3_url=https://your-bucket.s3.amazonaws.com/path/to/file.xlsx" \
  -d "project_name=my_project"
Response:
json{
  "message": "uploaded",
  "project_name": "my_project",
  "s3_url": "https://your-bucket.s3.amazonaws.com/path/to/file.xlsx"
}
2. Query Project
Ask questions about your data:
bashcurl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my_project",
    "query": "What is the expected payment amount and date for ABC Company?"
  }'
Response:
json{
  "answer": "Looking at ABC Company's payment history, they have 15 paid invoices totaling ₹2,450,000. Based on their last 3 invoices, they've been averaging around ₹185,000 per payment, which is about 12% higher than their overall average of ₹163,333. This upward trend suggests their business relationship is growing.\n\nTheir payment behavior shows they typically take about 28 days to pay after receiving an invoice. Since their last payment was on 2024-11-15, I'd expect the next payment around December 13, 2024. They're fairly consistent with their payment timing, rarely varying by more than a few days from their average..."
}
📊 Data Format
Expected Excel/CSV Columns
The application expects the following columns (column names are normalized automatically):
Expected ColumnAlternative NamesDescriptionCUSTOMER NAMEEntity Legal NameCustomer identifierINVOICE DATE-Date of invoiceINVOICE NUMBER-Unique invoice IDSALES ORDER NUMBER-Sales order referencePAY TERMS-Payment termsDUE DATE-Payment due dateNET AMOUNT-Invoice amount before taxVAT (5%)-Tax amountTOTAL INVOICE AMT-Total amount including taxPAYMENT DATE-Actual payment dateSALES MANAGER NAME-Sales person nameSERVICE TYPE-Type of serviceLOCATION-Location/region
Excel File Requirements

Header must be on row 3 (index 2)
First column can be unnamed (will be auto-dropped)
Supported formats: .xlsx, .xls

CSV File Requirements

Standard CSV format
Header in first row
Comma-separated values

🔐 Privacy & Security
Customer Name Aliasing
The system automatically anonymizes customer names:

Deterministic Hashing: Uses HMAC-SHA256 with a secret salt
Project-Scoped: Aliases are unique per project
Reversible: Original names stored securely in database
Consistent: Same customer always gets the same alias within a project

Example:

Input: "ABC Corporation Ltd."
Alias: "x7Km9pQl"
Stored in database for de-aliasing responses

Data Flow
User Data → S3 Upload → Download → Parse → Alias → LLM Analysis → De-alias → User
                                     ↓
                              PostgreSQL
                           (aliases stored)
🤖 AI-Powered Features
Predictive Analytics

Amount Forecasting

Uses moving average of recent invoices (configurable window)
Compares to historical average
Identifies trends (increasing/decreasing/stable)


Payment Date Prediction

Calculates average payment delay
Predicts next payment: last_payment_date + avg_delay_days
Describes payment reliability


Pattern Recognition

Identifies seasonal trends
Detects anomalies in payment behavior
Provides contextual insights



Conversational Responses
The LLM generates natural, detailed responses that:

Answer questions directly
Provide context and comparisons
Explain trends and patterns
Use specific numbers and percentages
Maintain professional yet conversational tone

🛠️ Configuration
Environment Variables
MOVING_AVG_WINDOW-3
Number of recent invoices for moving averageALIAS_LENGTH8Length of generated aliasesALIAS_SALT-Secret salt for HMAC hashing (required)LITE_LLM_MODELgemini/gemini-2.5-flash-liteAI model to use
Supported LLM Providers
Via LiteLLM, supports:

Google Gemini
OpenAI GPT
Anthropic Claude
Azure OpenAI
Cohere
And many more...

Change model by updating LITE_LLM_MODEL in .env
📁 Project Structure
generative_ai_project/
├── config/
│   ├── logging_config.yaml    # Logging configuration
│   └── settings.py             # Environment variables & constants
├── src/
│   ├── llm/
│   │   └── lite_client.py      # LiteLLM integration
│   ├── routes/
│   │   ├── upload.py           # Upload endpoint
│   │   └── query.py            # Query endpoint
│   └── utils/
│       ├── db.py               # Database operations
│       ├── s3_utils.py         # S3 file operations
│       ├── normalize.py        # Column normalization
│       └── aliaser.py          # Customer name aliasing
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # This file
🔍 Logging
Comprehensive logging to app.log:

All API requests and responses
S3 operations
Database queries
Aliasing operations
LLM interactions
Error traces

Log format:
2024-11-28 15:24:16,738 - INFO - query.py:45 - query_project - Query request received
🐛 Troubleshooting
Common Issues

"Unsupported file type" error

Ensure S3 URL doesn't have query parameters interfering with extension detection
Fixed in latest version using extract_filename_from_url()


"Column 'customer_name' not found"

Check that your Excel/CSV has "CUSTOMER NAME" column
System will fallback to "ENTITY LEGAL NAME" if available


Database connection fails

Verify PostgreSQL is running
Check DB credentials in .env
Ensure database exists: createdb financial_analytics


S3 download fails

Verify AWS credentials have S3 read permissions
Check bucket name and region are correct
Ensure files exist at specified S3 paths


LLM errors

Verify LITE_LLM_API_KEY is set correctly
Check model name is valid for your provider
Review LiteLLM documentation for provider-specific setup



🧪 Testing
bash# Test upload endpoint
curl -X POST "http://localhost:8000/upload" \
  -F "s3_url=https://bucket.s3.amazonaws.com/test.xlsx" \
  -F "project_name=test_project"

# Test query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_name": "test_project", "query": "What are the total invoices?"}'
📈 Performance Optimization

Database Indexing: Primary keys on project_name and customer_key
Connection Pooling: Consider adding for high-traffic scenarios
Caching: Pre-computed statistics reduce LLM processing time
Batch Operations: Aliases generated in bulk on first query

🔮 Future Enhancements

 Support for multiple file formats (JSON, Parquet)
 Real-time data streaming
 Dashboard UI for visualizations
 Export reports to PDF/Excel
 Role-based access control
 Multi-tenancy support
 Webhook notifications for payment predictions
 Integration with accounting software (QuickBooks, Xero)


🙏 Acknowledgments

Built with FastAPI
Powered by LiteLLM
Data storage by AWS S3 and PostgreSQL
AI capabilities via various LLM providers


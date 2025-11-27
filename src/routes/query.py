from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.db import get_project_url, init_db, get_alias
from src.utils.s3_utils import download_to_bytes
from src.utils.normalize import read_input_file
from src.utils.aliaser import make_alias
from src.llm.lite_client import lite_client
import io

router = APIRouter()

class QueryRequest(BaseModel):
    project_name: str
    query: str

@router.post("/query")
def query_project(req: QueryRequest):
    init_db()

    # 1. Get S3 URL
    s3_url = get_project_url(req.project_name)
    if not s3_url:
        raise HTTPException(status_code=404, detail="project not found")

    # 2. Download file
    try:
        data_bytes = download_to_bytes(s3_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to download from s3: {e}")

    # Extract filename from S3 URL
    filename = s3_url.split("/")[-1]

    # 3. Parse file
    try:
        df = read_input_file(io.BytesIO(data_bytes), filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to parse file: {e}")

    # 4. Alias creation
    name_col = "customer_name"
    mapping = {}

    if name_col in df.columns:
        unique_customers = [str(x).strip() for x in df[name_col].dropna().unique()]
        unique_customers.sort(key=lambda x: -len(x))

        for cust in unique_customers:
            alias = get_alias(req.project_name, cust)
            if alias:
                mapping[cust] = alias
            else:
                mapping[cust] = make_alias(req.project_name, cust)

    # 5. ALWAYS apply alias mapping to dataframe
    df_alias = df.copy()
    if name_col in df.columns:
        df_alias[name_col] = df_alias[name_col].astype(str).str.strip().map(mapping)

    # 6. Replace customer names in query
    transformed_query = req.query
    for cust, alias in mapping.items():
        transformed_query = transformed_query.replace(cust, alias)

    # 7. Convert full aliased dataset to CSV
    data_text = df_alias.to_csv(index=False)

    # 8. Build prompt for LLM
    prompt = (
        "You are given the following dataset (customer names have been replaced with aliases):\n\n"
        f"{data_text}\n\n"
        f"User question (alias-aware): {transformed_query}\n\n"
        "Answer concisely. If forecasting is needed, perform it with reasoning."
    )

    # 9. Call LLM
    resp = lite_client.generate(prompt)

    return {
        "model_response": resp,
        "transformed_query": transformed_query,
        "aliases_used": mapping
    }

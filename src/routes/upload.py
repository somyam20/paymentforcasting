from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.utils.s3_utils import upload_fileobj
from src.utils.db import save_project, init_db
from src.utils.normalize import read_input_file
from src.utils.aliaser import make_alias
import io

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), project_name: str = Form(...)):
    # initialize tables (idempotent)
    init_db()

    contents = await file.read()
    filelike_for_s3 = io.BytesIO(contents)
    filelike_for_pd = io.BytesIO(contents)

    key = f"{project_name}/{file.filename}"

    # upload to S3
    try:
        s3_url = upload_fileobj(filelike_for_s3, key, content_type=file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    # Save project record
    try:
        save_project(project_name, s3_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB save failed: {e}")

    # parse and precompute aliases
    try:
        df = read_input_file(filelike_for_pd, file.filename)
        name_col = "customer_name"
        if name_col in df.columns:
            for c in df[name_col].fillna(""):
                key_val = str(c).strip()
                if key_val:
                    make_alias(project_name, key_val)
    except Exception as e:
        # non-fatal — we've already stored the S3 url, but provide warning
        return {"message": "uploaded", "s3_url": s3_url, "warning": f"parsing/aliasing failed: {e}"}

    return {"message": "uploaded", "s3_url": s3_url}

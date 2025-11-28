from fastapi import APIRouter, Form, HTTPException
from src.utils.db import save_project, init_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_file(s3_url: str = Form(...), project_name: str = Form(...)):
    """
    Save S3 URL and project name mapping to database.
    
    Args:
        s3_url: The S3 URL where the file is stored (required)
        project_name: The project identifier (required)
    
    Returns:
        Success message with the S3 URL
    """
    # Validate inputs
    if not s3_url or not project_name:
        logger.error("Missing required fields: s3_url=%s, project_name=%s", s3_url, project_name)
        raise HTTPException(
            status_code=400, 
            detail="Both s3_url and project_name are required"
        )
    
    # Validate S3 URL format
    if not s3_url.startswith(("https://", "s3://")):
        logger.error("Invalid S3 URL format: %s", s3_url)
        raise HTTPException(
            status_code=400,
            detail="s3_url must be a valid S3 URL (starting with https:// or s3://)"
        )
    
    # Initialize database
    await init_db()
    
    try:
        # Save project to database
        await save_project(project_name, s3_url)
        logger.info("✓ Successfully saved project '%s' with S3 URL: %s", project_name, s3_url)
        
        return {
            "message": "uploaded",
            "project_name": project_name,
            "s3_url": s3_url
        }
        
    except Exception as e:
        logger.exception("✗ Failed to save project to database: %s", e)
        raise HTTPException(
            status_code=500, 
            detail=f"Database save failed: {str(e)}"
        )
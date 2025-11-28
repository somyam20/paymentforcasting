import aioboto3
import io
from botocore.exceptions import ClientError
from config.settings import AWS_REGION, AWS_S3_BUCKET, S3_UPLOAD_PREFIX
import logging
from urllib.parse import urlparse, unquote, parse_qs

logger = logging.getLogger(__name__)

# Create aioboto3 session
session = aioboto3.Session()


async def upload_fileobj(fileobj, key: str, content_type: str = None) -> str:
    """Upload a file-like object to S3 and return the object URL."""
    try:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        async with session.client("s3", region_name=AWS_REGION) as s3_client:
            await s3_client.upload_fileobj(fileobj, AWS_S3_BUCKET, key, ExtraArgs=extra_args)

        url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        return url
    except ClientError as e:
        raise


def extract_filename_from_url(s3_url: str) -> str:
    """
    Extract just the filename from S3 URL, ignoring query parameters.
    
    Example:
        Input: https://bucket.s3.amazonaws.com/path/file.xlsx?X-Amz-Algorithm=...
        Output: file.xlsx
    """
    parsed = urlparse(s3_url)
    # Get the path without query parameters
    path = parsed.path
    # Extract filename (last part after /)
    filename = path.split("/")[-1]
    # URL decode it
    filename = unquote(filename)
    
    logger.info("Extracted filename: %s from URL", filename)
    return filename


async def download_to_bytes(s3_url: str) -> bytes:
    """
    Download an S3 object based on the URL and return bytes.
    
    Supports multiple S3 URL formats including pre-signed URLs:
    - https://bucket.s3.region.amazonaws.com/key/path/file.ext
    - https://bucket.s3.amazonaws.com/key/path/file.ext?X-Amz-Algorithm=...
    - https://s3.region.amazonaws.com/bucket/key/path/file.ext
    - s3://bucket/key/path/file.ext
    """
    logger.info("Downloading from S3 URL: %s", s3_url[:100] + "..." if len(s3_url) > 100 else s3_url)
    
    try:
        # Parse the URL
        if s3_url.startswith("s3://"):
            # s3://bucket/key/path/file.ext
            parts = s3_url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
        elif "s3" in s3_url:
            # Parse HTTPS URL (including pre-signed URLs)
            parsed = urlparse(s3_url)
            
            # Extract bucket and key based on URL format
            if parsed.netloc.endswith(".amazonaws.com"):
                # Format: https://bucket.s3.region.amazonaws.com/key/path/file.ext
                # or: https://bucket.s3.amazonaws.com/key/path/file.ext
                bucket = parsed.netloc.split(".")[0]
                # Get path without query parameters
                key = parsed.path.lstrip("/")
            else:
                # Format: https://s3.region.amazonaws.com/bucket/key/path/file.ext
                path_parts = parsed.path.lstrip("/").split("/", 1)
                bucket = path_parts[0]
                key = path_parts[1] if len(path_parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        
        # URL decode the key (in case of spaces or special characters)
        key = unquote(key)
        
        logger.info("Extracted - Bucket: %s, Key: %s", bucket, key)
        
        # Use the extracted bucket or fall back to configured bucket
        bucket_to_use = bucket if bucket else AWS_S3_BUCKET
        
        logger.info("Attempting to download from bucket '%s' with key '%s'", bucket_to_use, key)
        
        # Download the object
        async with session.client("s3", region_name=AWS_REGION) as s3_client:
            obj = await s3_client.get_object(Bucket=bucket_to_use, Key=key)
            async with obj["Body"] as stream:
                data = await stream.read()
        
        logger.info("✓ Successfully downloaded %d bytes", len(data))
        return data
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error("✗ S3 ClientError: %s - %s", error_code, e.response['Error']['Message'])
        
        if error_code == 'NoSuchKey':
            logger.error("The key '%s' does not exist in bucket '%s'", key, bucket_to_use)
            logger.error("Please verify:")
            logger.error("  1. The S3 URL is correct")
            logger.error("  2. The file exists at this location")
            logger.error("  3. The bucket name is correct")
        elif error_code == 'NoSuchBucket':
            logger.error("The bucket '%s' does not exist", bucket_to_use)
        elif error_code == 'AccessDenied':
            logger.error("Access denied to bucket '%s' or key '%s'", bucket_to_use, key)
            logger.error("Please check AWS credentials and S3 bucket permissions")
        
        raise
    except Exception as e:
        logger.exception("✗ Unexpected error downloading from S3: %s", e)
        raise
import asyncio
import boto3
import io
from botocore.exceptions import ClientError
from config.settings import AWS_REGION, AWS_S3_BUCKET, S3_UPLOAD_PREFIX
import logging
from urllib.parse import urlparse, unquote, parse_qs

logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name=AWS_REGION)


# ==========================================================
# ORIGINAL SYNC FUNCTIONS (UNCHANGED)
# ==========================================================

def upload_fileobj(fileobj, key: str, content_type: str = None) -> str:
    """Upload a file-like object to S3 and return the object URL."""
    try:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        s3_client.upload_fileobj(fileobj, AWS_S3_BUCKET, key, ExtraArgs=extra_args)

        url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
        return url
    except ClientError as e:
        raise


def extract_filename_from_url(s3_url: str) -> str:
    """Extract just the filename from S3 URL."""
    parsed = urlparse(s3_url)
    path = parsed.path
    filename = path.split("/")[-1]
    filename = unquote(filename)

    logger.info("Extracted filename: %s from URL", filename)
    return filename


def download_to_bytes(s3_url: str) -> bytes:
    """Download S3 file synchronously and return bytes."""
    logger.info("Downloading from S3 URL: %s", s3_url[:100] + "..." if len(s3_url) > 100 else s3_url)

    try:
        # s3://bucket/key
        if s3_url.startswith("s3://"):
            parts = s3_url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

        elif "s3" in s3_url:
            parsed = urlparse(s3_url)

            # bucket.s3.amazonaws.com
            if parsed.netloc.endswith(".amazonaws.com"):
                bucket = parsed.netloc.split(".")[0]
                key = parsed.path.lstrip("/")
            else:
                # s3.region.amazonaws.com/bucket/key
                parts = parsed.path.lstrip("/").split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""

        else:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")

        key = unquote(key)

        logger.info("Extracted - Bucket: %s, Key: %s", bucket, key)

        bucket_to_use = bucket if bucket else AWS_S3_BUCKET

        obj = s3_client.get_object(Bucket=bucket_to_use, Key=key)
        data = obj["Body"].read()

        logger.info("✓ Successfully downloaded %d bytes", len(data))
        return data

    except ClientError as e:
        code = e.response['Error']['Code']
        msg = e.response['Error']['Message']
        logger.error("✗ S3 ClientError: %s - %s", code, msg)
        raise

    except Exception as e:
        logger.exception("✗ Unexpected error downloading from S3: %s", e)
        raise


# ==========================================================
# ASYNC WRAPPERS (THIS IS WHAT YOU WILL CALL)
# ==========================================================

async def async_upload_fileobj(fileobj, key: str, content_type: str = None) -> str:
    """Async wrapper for upload_fileobj."""
    return await asyncio.to_thread(upload_fileobj, fileobj, key, content_type)


async def async_download_to_bytes(s3_url: str) -> bytes:
    """Async wrapper for download_to_bytes."""
    return await asyncio.to_thread(download_to_bytes, s3_url)


async def async_extract_filename_from_url(s3_url: str) -> str:
    """This one is cheap, but keep it async anyway for consistency."""
    return await asyncio.to_thread(extract_filename_from_url, s3_url)

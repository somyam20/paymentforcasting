import boto3
import io
from botocore.exceptions import ClientError
from config.settings import AWS_REGION, AWS_S3_BUCKET, S3_UPLOAD_PREFIX


s3_client = boto3.client("s3", region_name=AWS_REGION)




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




def download_to_bytes(s3_url: str) -> bytes:
    """Download an S3 object based on the URL and return bytes."""
    # Accept URLs of format https://bucket.s3.region.amazonaws.com/key
    # naive parse:
    parts = s3_url.split("/")
    key = "/".join(parts[3:]) if "s3" in parts[2] else "/".join(parts[4:])
    obj = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=key)
    return obj["Body"].read()
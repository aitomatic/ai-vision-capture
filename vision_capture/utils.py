import asyncio
from typing import Any, List, Optional, Union

from loguru import logger


def get_s3_client() -> Any:
    import boto3  # type: ignore

    return boto3.client("s3")


def ensure_bucket_exists(bucket_name: str) -> None:
    s3_client = get_s3_client()
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception:
        # The bucket does not exist or you have no access.
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} created successfully")
        except Exception as e:
            logger.error(f"Could not create bucket {bucket_name}: {e}")
            raise


async def list_s3_files(bucket: str, prefix: str) -> List[str]:
    s3_client = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if "/images/" in str(obj["Key"]):
                    continue
                files.append(obj["Key"])
    if prefix in files:
        files.remove(prefix)
    return files


async def upload_file_to_s3_async(
    bucket: str, file_or_data: Union[str, bytes], s3_path: str
) -> None:
    """Async version of upload_file_to_s3.

    Args:
        bucket: S3 bucket name
        file_or_data: Either a file path (str) or bytes data to upload
        s3_path: S3 key to upload to
    """
    try:
        loop = asyncio.get_running_loop()
        s3_client = get_s3_client()

        if isinstance(file_or_data, str):
            # If it's a file path, use upload_file
            logger.info(f"Uploading file {file_or_data} to {bucket}/{s3_path}")
            await loop.run_in_executor(
                None, s3_client.upload_file, file_or_data, bucket, s3_path
            )
        else:
            # If it's bytes, use put_object
            logger.info(f"Uploading data to {bucket}/{s3_path}")
            await loop.run_in_executor(
                None,
                lambda: s3_client.put_object(
                    Bucket=bucket, Key=s3_path, Body=file_or_data
                ),
            )
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")


async def delete_file_from_s3_async(bucket: str, key: str) -> None:
    """Async delete file from S3."""
    try:
        loop = asyncio.get_running_loop()
        s3_client = get_s3_client()
        await loop.run_in_executor(
            None, lambda: s3_client.delete_object(Bucket=bucket, Key=key)
        )
    except Exception as e:
        logger.error(f"Error deleting from S3: {e}")


async def get_file_from_s3_async(bucket: str, key: str) -> Optional[bytes]:
    """Async get file content from S3.

    Args:
        bucket: S3 bucket name
        key: S3 key/path to the file

    Returns:
        File contents as bytes if successful, None otherwise
    """
    try:
        loop = asyncio.get_running_loop()
        s3_client = get_s3_client()
        response = await loop.run_in_executor(
            None, lambda: s3_client.get_object(Bucket=bucket, Key=key)
        )
        return await loop.run_in_executor(None, lambda: response["Body"].read())
    except Exception as e:
        logger.error(f"Unexpected error reading from S3: {str(e)}")
        return None

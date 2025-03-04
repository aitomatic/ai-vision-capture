import asyncio
import os
from typing import List, Optional, Union

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from loguru import logger


def get_s3_client() -> boto3.client:
    return boto3.client("s3")


def ensure_bucket_exists(bucket_name: str) -> None:
    s3_client = get_s3_client()
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        # The bucket does not exist or you have no access.
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} created successfully")
        except ClientError as e:
            logger.error(f"Could not create bucket {bucket_name}: {e}")
            raise


async def download_file_from_s3_to_local(
    bucket: str, s3_path: str, local_path: str
) -> None:
    loop = asyncio.get_running_loop()
    s3_client = get_s3_client()
    await loop.run_in_executor(
        None, s3_client.download_file, bucket, s3_path, local_path
    )


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


async def download_folder_from_s3_to_local(
    bucket: str, s3_folder_path: str, local_path: str
) -> None:
    if os.path.exists(local_path) and os.path.isdir(local_path):
        print(f"Folder {local_path} already exists locally. Skipping download.")
        return

    if not s3_folder_path.endswith("/"):
        s3_folder_path += "/"
    print(f"Downloading folder {s3_folder_path} to {local_path}")
    s3_files = await list_s3_files(bucket, s3_folder_path)

    download_tasks = []
    for s3_file_path in s3_files:
        local_file_path = os.path.join(
            local_path, os.path.relpath(s3_file_path, s3_folder_path)
        )
        local_file_dir = os.path.dirname(local_file_path)
        os.makedirs(local_file_dir, exist_ok=True)
        if "/images/" in s3_file_path:
            print(f"Skipping image file {s3_file_path}")
            continue

        print(f"Downloading file {s3_file_path} to {local_file_path}")

        download_tasks.append(
            download_file_from_s3_to_local(bucket, s3_file_path, local_file_path)
        )

    await asyncio.gather(*download_tasks)


def upload_file_to_s3(bucket: str, file_path: str, s3_path: str) -> None:
    try:
        logger.info(f"Uploading {file_path} to {bucket}/{s3_path}")
        s3_client = get_s3_client()
        s3_client.upload_file(file_path, bucket, s3_path)
    except Exception as e:
        print(e)


def upload_folder_to_s3(
    bucket_name: str, local_directory_path: str, s3_folder_path: str
) -> None:
    s3_client = get_s3_client()

    if not s3_folder_path.endswith("/"):
        s3_folder_path += "/"

    for root, dirs, files in os.walk(local_directory_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory_path)
            s3_path = os.path.join(s3_folder_path, relative_path)

            try:
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f"Uploaded {local_path} to {bucket_name}/{s3_path}")
            except ClientError as e:
                print(f"Couldn't upload {local_path}. Reason: {e}")


def get_s3_folder_size(bucket_name: str, prefix: str, unit: str = "MB") -> float:
    s3_client = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")

    total_size = 0
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            total_size += obj["Size"]

    if unit == "KB":
        return total_size / 1024
    elif unit == "MB":
        return total_size / (1024 * 1024)
    elif unit == "GB":
        return total_size / (1024 * 1024 * 1024)
    else:
        return total_size


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
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            logger.debug(f"File not found in S3: {bucket}/{key}")
        else:
            logger.error(f"S3 ClientError: {error_code} - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading from S3: {str(e)}")
        return None

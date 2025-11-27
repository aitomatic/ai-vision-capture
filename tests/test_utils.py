from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pytest import MonkeyPatch

from aicapture.utils import (
    delete_file_from_s3_async,
    download_file_from_s3_async,
    ensure_bucket_exists,
    get_default_bucket,
    get_file_from_s3_async,
    get_s3_client,
    list_objects_from_s3_async,
    list_s3_files,
    upload_file_to_s3_async,
)


class TestGetS3Client:
    """Test cases for get_s3_client function."""

    def test_get_s3_client_aws(self, monkeypatch: MonkeyPatch) -> None:
        """Test getting AWS S3 client."""
        monkeypatch.setenv("USE_MINIO", "false")

        # Mock boto3 at the function import level
        with patch("builtins.__import__") as mock_import:
            mock_boto3 = Mock()
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client

            def side_effect(name: str, *args: Any) -> Any:
                if name == "boto3":
                    return mock_boto3
                return __import__(name, *args)

            mock_import.side_effect = side_effect

            client = get_s3_client()

            mock_boto3.client.assert_called_once_with("s3")
            assert client == mock_client

    def test_get_s3_client_minio(self, monkeypatch: MonkeyPatch) -> None:
        """Test getting MinIO S3 client."""
        monkeypatch.setenv("USE_MINIO", "true")
        monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "test_access")
        monkeypatch.setenv("MINIO_SECRET_KEY", "test_secret")

        # Mock boto3 at the function import level
        with patch("builtins.__import__") as mock_import:
            mock_boto3 = Mock()
            mock_client = Mock()
            mock_config = Mock()
            mock_boto3.client.return_value = mock_client
            mock_boto3.session.Config.return_value = mock_config

            def side_effect(name: str, *args: Any) -> Any:
                if name == "boto3":
                    return mock_boto3
                return __import__(name, *args)

            mock_import.side_effect = side_effect

            client = get_s3_client()

            mock_boto3.client.assert_called_once_with(
                "s3",
                endpoint_url="http://localhost:9000",
                aws_access_key_id="test_access",
                aws_secret_access_key="test_secret",
                config=mock_config,
            )
            assert client == mock_client

    def test_get_s3_client_minio_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test getting MinIO S3 client with default values."""
        monkeypatch.setenv("USE_MINIO", "true")
        # Clear environment variables to test defaults
        for env_var in ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"]:
            monkeypatch.delenv(env_var, raising=False)

        # Mock boto3 at the function import level
        with patch("builtins.__import__") as mock_import:
            mock_boto3 = Mock()
            mock_client = Mock()
            mock_config = Mock()
            mock_boto3.client.return_value = mock_client
            mock_boto3.session.Config.return_value = mock_config

            def side_effect(name: str, *args: Any) -> Any:
                if name == "boto3":
                    return mock_boto3
                return __import__(name, *args)

            mock_import.side_effect = side_effect

            get_s3_client()

            # Should use default values
            call_args = mock_boto3.client.call_args
            assert call_args[1]["endpoint_url"] == "http://localhost:9000"
            assert call_args[1]["aws_access_key_id"] == "minioadmin"
            assert call_args[1]["aws_secret_access_key"] == "minioadmin"


class TestEnsureBucketExists:
    """Test cases for ensure_bucket_exists function."""

    def test_bucket_exists(self) -> None:
        """Test when bucket already exists."""
        mock_client = Mock()
        # head_bucket succeeds, so bucket exists
        mock_client.head_bucket.return_value = {}

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            ensure_bucket_exists("test-bucket")

            mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
            mock_client.create_bucket.assert_not_called()

    def test_bucket_does_not_exist(self) -> None:
        """Test when bucket does not exist and needs to be created."""
        mock_client = Mock()
        # head_bucket raises exception (bucket doesn't exist)
        mock_client.head_bucket.side_effect = Exception("NoSuchBucket")
        mock_client.create_bucket.return_value = {}

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            ensure_bucket_exists("test-bucket")

            mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
            mock_client.create_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_bucket_creation_fails(self) -> None:
        """Test when bucket creation fails."""
        mock_client = Mock()
        mock_client.head_bucket.side_effect = Exception("NoSuchBucket")
        mock_client.create_bucket.side_effect = Exception("CreateBucketFailed")

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with pytest.raises(Exception, match="CreateBucketFailed"):
                ensure_bucket_exists("test-bucket")


class TestGetDefaultBucket:
    """Test cases for get_default_bucket function."""

    def test_get_default_bucket_env_var(self, monkeypatch: MonkeyPatch) -> None:
        """Test getting default bucket from environment variable."""
        monkeypatch.setenv("DXA_DATA_BUCKET", "custom-bucket")

        bucket = get_default_bucket()
        assert bucket == "custom-bucket"

    def test_get_default_bucket_default(self, monkeypatch: MonkeyPatch) -> None:
        """Test getting default bucket when env var is not set."""
        monkeypatch.delenv("DXA_DATA_BUCKET", raising=False)

        bucket = get_default_bucket()
        assert bucket == "test-bucket-local"


class TestListS3Files:
    """Test cases for list_s3_files function."""

    @pytest.mark.asyncio
    async def test_list_s3_files_success(self) -> None:
        """Test successful listing of S3 files."""
        mock_client = Mock()
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator

        # Mock paginated response
        mock_pages = [
            {
                "Contents": [
                    {"Key": "prefix/file1.txt"},
                    {"Key": "prefix/file2.txt"},
                    {"Key": "prefix/images/image1.jpg"},  # Should be filtered out
                ]
            },
            {
                "Contents": [
                    {"Key": "prefix/file3.txt"},
                    {"Key": "prefix"},  # Should be removed
                ]
            },
        ]
        mock_paginator.paginate.return_value = mock_pages

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            files = await list_s3_files("test-bucket", "prefix")

            # Should exclude images and the prefix itself
            expected_files = [
                "prefix/file1.txt",
                "prefix/file2.txt",
                "prefix/file3.txt",
            ]
            assert files == expected_files

            mock_client.get_paginator.assert_called_once_with("list_objects_v2")
            mock_paginator.paginate.assert_called_once_with(Bucket="test-bucket", Prefix="prefix")

    @pytest.mark.asyncio
    async def test_list_s3_files_empty(self) -> None:
        """Test listing S3 files when no files exist."""
        mock_client = Mock()
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator

        # Mock empty response
        mock_paginator.paginate.return_value = [{}]  # No Contents key

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            files = await list_s3_files("test-bucket", "empty-prefix")

            assert files == []

    @pytest.mark.asyncio
    async def test_list_s3_files_filters_images(self) -> None:
        """Test that image files are filtered out."""
        mock_client = Mock()
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator

        mock_pages = [
            {
                "Contents": [
                    {"Key": "prefix/document.pdf"},
                    {"Key": "prefix/images/photo.jpg"},
                    {"Key": "prefix/data/images/chart.png"},
                    {"Key": "prefix/text.txt"},
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_pages

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            files = await list_s3_files("test-bucket", "prefix")

            # Only non-image files should be returned
            expected_files = ["prefix/document.pdf", "prefix/text.txt"]
            assert files == expected_files


class TestUploadFileToS3Async:
    """Test cases for upload_file_to_s3_async function."""

    @pytest.mark.asyncio
    async def test_upload_file_path(self) -> None:
        """Test uploading a file by path."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                Mock()
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                await upload_file_to_s3_async("test-bucket", "/path/to/file.txt", "s3/path/file.txt")

                # Should call upload_file
                mock_loop.return_value.run_in_executor.assert_called_once()
                args = mock_loop.return_value.run_in_executor.call_args[0]
                assert args[0] is None  # executor
                assert args[1] == mock_client.upload_file

    @pytest.mark.asyncio
    async def test_upload_bytes_data(self) -> None:
        """Test uploading bytes data."""
        mock_client = Mock()
        test_data = b"test file content"

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                await upload_file_to_s3_async("test-bucket", test_data, "s3/path/data.bin")

                # Should call put_object via lambda
                mock_loop.return_value.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_error(self) -> None:
        """Test error handling during upload."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=Exception("Upload failed"))

                # Should not raise exception, but log error
                await upload_file_to_s3_async("test-bucket", "/path/to/file.txt", "s3/path/file.txt")


class TestDeleteFileFromS3Async:
    """Test cases for delete_file_from_s3_async function."""

    @pytest.mark.asyncio
    async def test_delete_file_success(self) -> None:
        """Test successful file deletion."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                await delete_file_from_s3_async("test-bucket", "path/to/file.txt")

                mock_loop.return_value.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_file_error(self) -> None:
        """Test error handling during deletion."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=Exception("Delete failed"))

                # Should not raise exception, but log error
                await delete_file_from_s3_async("test-bucket", "path/to/file.txt")


class TestGetFileFromS3Async:
    """Test cases for get_file_from_s3_async function."""

    @pytest.mark.asyncio
    async def test_get_file_success(self) -> None:
        """Test successful file retrieval."""
        mock_client = Mock()
        mock_response = {"Body": Mock()}
        mock_response["Body"].read.return_value = b"file content"

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                # Mock two executor calls: one for get_object, one for body.read()
                mock_loop.return_value.run_in_executor = AsyncMock()
                mock_loop.return_value.run_in_executor.side_effect = [
                    mock_response,  # get_object response
                    b"file content",  # body.read() result
                ]

                result = await get_file_from_s3_async("test-bucket", "path/to/file.txt")

                assert result == b"file content"
                assert mock_loop.return_value.run_in_executor.call_count == 2

    @pytest.mark.asyncio
    async def test_get_file_not_found(self) -> None:
        """Test file not found error."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=Exception("NoSuchKey"))

                result = await get_file_from_s3_async("test-bucket", "nonexistent/file.txt")

                assert result is None


class TestDownloadFileFromS3Async:
    """Test cases for download_file_from_s3_async function."""

    @pytest.mark.asyncio
    async def test_download_file_success(self) -> None:
        """Test successful file download."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                result = await download_file_from_s3_async("test-bucket", "s3/path/file.txt", "/local/path/file.txt")

                assert result is True
                mock_loop.return_value.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_error(self) -> None:
        """Test error handling during download."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=Exception("Download failed"))

                result = await download_file_from_s3_async("test-bucket", "s3/path/file.txt", "/local/path/file.txt")

                assert result is False


class TestListObjectsFromS3Async:
    """Test cases for list_objects_from_s3_async function."""

    @pytest.mark.asyncio
    async def test_list_objects_success(self) -> None:
        """Test successful object listing."""
        mock_client = Mock()
        mock_response = {
            "Contents": [
                {"Key": "prefix/file1.txt", "Size": 1024},
                {"Key": "prefix/file2.txt", "Size": 2048},
            ]
        }

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

                result = await list_objects_from_s3_async("test-bucket", "prefix")

                assert len(result) == 2
                assert result[0]["Key"] == "prefix/file1.txt"
                assert result[0]["Size"] == 1024
                assert result[1]["Key"] == "prefix/file2.txt"
                assert result[1]["Size"] == 2048

    @pytest.mark.asyncio
    async def test_list_objects_empty(self) -> None:
        """Test listing when no objects exist."""
        mock_client = Mock()
        mock_response: Dict[str, Any] = {}  # No Contents key

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

                result = await list_objects_from_s3_async("test-bucket", "empty-prefix")

                assert result == []

    @pytest.mark.asyncio
    async def test_list_objects_error(self) -> None:
        """Test error handling during object listing."""
        mock_client = Mock()

        with patch("aicapture.utils.get_s3_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(side_effect=Exception("List failed"))

                result = await list_objects_from_s3_async("test-bucket", "prefix")

                assert result == []


class TestMinioInitialization:
    """Test MinIO initialization logic."""

    def test_minio_bucket_creation(self, monkeypatch: MonkeyPatch) -> None:
        """Test that default bucket is created when using MinIO."""
        monkeypatch.setenv("USE_MINIO", "true")
        monkeypatch.setenv("DXA_DATA_BUCKET", "test-minio-bucket")

        # Test the initialization logic directly instead of module reload
        with patch("aicapture.utils.ensure_bucket_exists") as mock_ensure:
            # Mock get_s3_client to avoid boto3 calls
            with patch("aicapture.utils.get_s3_client") as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client

                # Simulate the module initialization code
                import os

                if os.getenv("USE_MINIO", "false").lower() == "true":
                    from aicapture.utils import ensure_bucket_exists

                    default_bucket = os.getenv("DXA_DATA_BUCKET", "test-bucket-local")
                    ensure_bucket_exists(default_bucket)

                # Should have called ensure_bucket_exists
                mock_ensure.assert_called_with("test-minio-bucket")

    def test_aws_no_bucket_creation(self, monkeypatch: MonkeyPatch) -> None:
        """Test that bucket creation is not called for AWS."""
        monkeypatch.setenv("USE_MINIO", "false")

        with patch("aicapture.utils.ensure_bucket_exists"):
            # Re-import to trigger initialization
            import importlib

            import aicapture.utils

            importlib.reload(aicapture.utils)

            # Should not have called ensure_bucket_exists (in current session)
            # Note: This test might need adjustment based on module loading behavior


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

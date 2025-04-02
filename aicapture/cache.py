# flake8: noqa: E501

import abc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from loguru import logger
from PIL import Image, ImageFile

from aicapture.settings import CLOUD_CACHE_BUCKET
from aicapture.utils import (
    delete_file_from_s3_async,
    get_file_from_s3_async,
    get_s3_client,
    upload_file_to_s3_async,
)


class CacheInterface(abc.ABC):
    """Abstract base class defining the interface for cache implementations."""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item from the cache."""
        pass

    @abc.abstractmethod
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store an item in the cache."""
        pass

    @abc.abstractmethod
    def invalidate(self, key: str) -> bool:
        """Remove an item from the cache."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all items from the cache."""
        pass


class AsyncCacheInterface(abc.ABC):
    """Abstract base class defining the interface for async cache implementations."""

    @abc.abstractmethod
    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        """Async retrieve an item from the cache."""
        pass

    @abc.abstractmethod
    async def aset(self, key: str, value: Dict[str, Any]) -> None:
        """Async store an item in the cache."""
        pass

    @abc.abstractmethod
    async def ainvalidate(self, key: str) -> bool:
        """Async remove an item from the cache."""
        pass

    @abc.abstractmethod
    async def aclear(self) -> None:
        """Async clear all items from the cache."""
        pass


class FileCache(CacheInterface):
    """A file-based cache implementation that stores data in JSON files."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the file cache.

        Args:
            cache_dir: store cache files. Defaults to tmp/.vision_parser_cache
        """
        try:
            self.cache_dir = (
                Path(cache_dir) if cache_dir else Path("tmp") / ".vision_parser_cache"
            )
            if not self.cache_dir.parent.exists():
                logger.warning(
                    f"Parent directory {self.cache_dir.parent} does not exist. Creating it."
                )
                self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized file cache at {self.cache_dir}")
        except PermissionError as e:
            logger.error(
                f"Permission denied when creating cache directory at {self.cache_dir}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to initialize cache directory at {self.cache_dir}: {e}"
            )
            raise

    def _get_cache_path(self, key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            The cached data if found, None otherwise
        """
        cache_file = self._get_cache_path(key)
        try:
            if cache_file.exists():
                logger.info(f"Loading cached result from {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    return cast(Dict[str, Any], json.load(f))
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}")
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store an item in the cache.

        Args:
            key: Cache key to store
            value: Data to cache
        """
        cache_file = self._get_cache_path(key)
        try:
            logger.info(f"Saving result to cache: {cache_file}")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")

    def invalidate(self, key: str) -> bool:
        """Remove an item from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if the item was removed, False otherwise
        """
        cache_file = self._get_cache_path(key)
        try:
            if cache_file.exists():
                cache_file.unlink()
                return True
        except Exception as e:
            logger.warning(f"Error removing cache file: {str(e)}")
        return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared all cache files from {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Error clearing cache directory: {str(e)}")


class S3Cache(AsyncCacheInterface):
    """An S3-based cache implementation."""

    def __init__(self, bucket: str, prefix: str):
        """Initialize the S3 cache.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix for cache files
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        logger.debug(f"Initialized S3 cache at s3://{bucket}/{prefix}")

    def _get_s3_path(self, key: str) -> str:
        """Get the full S3 path for a cache key."""
        return f"{self.prefix}{key}.json"

    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        """Async retrieve an item from the cache."""
        s3_path = self._get_s3_path(key)
        try:
            logger.info(f"Loading from S3: s3://{self.bucket}/{s3_path}")
            data = await get_file_from_s3_async(bucket=self.bucket, key=s3_path)
            if data:
                logger.info(
                    f"Successfully loaded from S3: s3://{self.bucket}/{s3_path}"
                )
                return json.loads(data.decode("utf-8"))  # type: ignore
            logger.info(f"No data found in S3: s3://{self.bucket}/{s3_path}")
        except Exception as e:
            logger.error(f"Error loading from S3 cache: {e}")
        return None

    async def aset(self, key: str, value: Any) -> None:
        """Async store an item in the cache."""
        try:
            s3_path = self._get_s3_path(key)
            data_bytes = json.dumps(value).encode("utf-8")

            logger.info(f"Uploading to S3: s3://{self.bucket}/{s3_path}")
            await upload_file_to_s3_async(
                bucket=self.bucket, file_or_data=data_bytes, s3_path=s3_path
            )
            logger.info(f"Successfully uploaded to S3: s3://{self.bucket}/{s3_path}")
        except Exception as e:
            logger.error(f"Error setting value in S3 cache: {e}")
            # Don't raise the exception as we want to continue even if S3 fails

    async def ainvalidate(self, key: str) -> bool:
        """Async remove an item from the cache."""
        s3_path = self._get_s3_path(key)
        try:
            logger.info(f"Deleting from S3: s3://{self.bucket}/{s3_path}")
            await delete_file_from_s3_async(bucket=self.bucket, key=s3_path)
            logger.info(f"Successfully deleted from S3: s3://{self.bucket}/{s3_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from S3 cache: {e}")
            return False

    async def aclear(self) -> None:
        """Async clear all items from the cache."""
        logger.warning("S3Cache clear operation not implemented")


class TwoLayerCache:
    """A two-layer cache implementation combining file and S3 caches."""

    def __init__(
        self,
        file_cache: FileCache,
        s3_cache: Optional[S3Cache],
        invalidate_cache: bool = False,
    ):
        """Initialize the two-layer cache.

        Args:
            file_cache: Local file cache instance
            s3_cache: Optional S3 cache instance
            invalidate_cache: If True, bypass cache for reads
        """
        self.file_cache = file_cache
        self.s3_cache = s3_cache
        self.invalidate_cache = invalidate_cache

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache, checking file cache first then S3."""
        if self.invalidate_cache:
            return None

        # Check file cache first
        result = self.file_cache.get(key)
        if result:
            logger.info("Found result in file cache")
            return result

        # Check S3 cache if available
        if self.s3_cache:
            result = await self.s3_cache.aget(key)
            if result:
                # Save to file cache for future use
                logger.info("Saving S3 result to file cache")
                self.file_cache.set(key, result)
                return result

        return None

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set an item in both file and S3 caches."""
        try:
            # Save to file cache first
            logger.info("Saving to file cache")
            self.file_cache.set(key, value)

            # Save to S3 if available
            if self.s3_cache:
                logger.info("Saving to S3 cache")
                await self.s3_cache.aset(key, value)
        except Exception as e:
            logger.error(f"Error in cache set operation: {e}")

    async def invalidate(self, key: str) -> None:
        """Invalidate an item from both caches."""
        try:
            logger.info("Invalidating file cache")
            self.file_cache.invalidate(key)
            if self.s3_cache:
                logger.info("Invalidating S3 cache")
                await self.s3_cache.ainvalidate(key)
        except Exception as e:
            logger.error(f"Error in cache invalidation: {e}")

    async def clear(self) -> None:
        """Clear both caches."""
        try:
            logger.info("Clearing file cache")
            self.file_cache.clear()
            if self.s3_cache:
                logger.info("Clearing S3 cache")
                await self.s3_cache.aclear()
        except Exception as e:
            logger.error(f"Error in cache clear operation: {e}")


class HashUtils:
    """Utility class for generating cache keys."""

    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file to hash

        Returns:
            SHA-256 hash of the file
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def get_cache_key(file_hash: str, prompt: str) -> str:
        """Get a cache key for a file and prompt."""
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{file_hash}_{prompt_hash}"


class ImageCache:
    """A specialized cache implementation for storing PDF page images."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the image cache.

        Args:
            cache_dir: Base directory for storing cached images.
                      Defaults to tmp/.vision_parser_cache/images
        """
        try:
            self.cache_dir = (
                Path(cache_dir) / "images"
                if cache_dir
                else Path("tmp") / ".vision_parser_cache/images"
            )
            if not self.cache_dir.parent.exists():
                logger.warning(
                    f"Parent directory {self.cache_dir.parent} does not exist. Creating it."
                )
                self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized image cache at {self.cache_dir}")
        except PermissionError as e:
            logger.error(
                f"Permission denied when creating image cache directory at {self.cache_dir}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to initialize image cache directory at {self.cache_dir}: {e}"
            )
            raise

    def _get_local_cache_path(self, file_hash: str) -> Path:
        """Get the local cache directory path for a file hash."""
        return self.cache_dir / file_hash

    def _get_s3_prefix(self, file_hash: str) -> str:
        """Get the S3 prefix for a file hash."""
        return f"production/images/raw_images/{file_hash}"

    def _validate_cache(self, cache_path: Path, expected_pages: int) -> bool:
        """Validate that the cache directory exists and
        contains the expected number of files."""
        if not cache_path.exists():
            return False

        # Count actual image files in the directory
        image_files = list(cache_path.glob("p[0-9]*.png"))
        return len(image_files) == expected_pages

    async def get_images(
        self, file_hash: str, expected_pages: int
    ) -> Optional[List[Union[Image.Image, ImageFile.ImageFile]]]:
        """Retrieve cached images for a file.

        Args:
            file_hash: Hash of the PDF file
            expected_pages: Expected number of pages

        Returns:
            List of PIL Image objects if cache hit, None if cache miss
        """
        local_cache_path = self._get_local_cache_path(file_hash)

        # Try local cache first
        if self._validate_cache(local_cache_path, expected_pages):
            logger.info(f"Found images in local cache: {local_cache_path}")
            images: List[Union[Image.Image, ImageFile.ImageFile]] = []
            for i in range(1, expected_pages + 1):
                image_path = local_cache_path / f"p{i}.png"
                images.append(Image.open(image_path))
            return images

        # Try S3 cache if local cache misses
        try:
            s3_client = get_s3_client()
            s3_prefix = self._get_s3_prefix(file_hash)

            # List objects in S3 folder
            response = s3_client.list_objects_v2(
                Bucket=CLOUD_CACHE_BUCKET, Prefix=s3_prefix
            )

            if response.get("KeyCount", 0) == expected_pages:
                logger.info(f"Found images in S3 cache: {s3_prefix}")

                # Create local cache directory
                local_cache_path.mkdir(parents=True, exist_ok=True)

                # Download and load images
                s3_images: List[Union[Image.Image, ImageFile.ImageFile]] = []
                for i in range(1, expected_pages + 1):
                    s3_key = f"{s3_prefix}/p{i}.png"
                    local_path = local_cache_path / f"p{i}.png"

                    # Download from S3
                    s3_client.download_file(CLOUD_CACHE_BUCKET, s3_key, str(local_path))

                    # Load image
                    s3_images.append(Image.open(local_path))

                return s3_images

        except Exception as e:
            logger.error(f"Error retrieving from S3 cache: {e}")

        return None

    async def cache_images(
        self, images: List[Union[Image.Image, ImageFile.ImageFile]], file_hash: str
    ) -> None:
        """Cache a list of images both locally and in S3.

        Args:
            images: List of PIL Image objects to cache
            file_hash: Hash of the PDF file
        """
        try:
            local_cache_path = self._get_local_cache_path(file_hash)
            local_cache_path.mkdir(parents=True, exist_ok=True)

            s3_prefix = self._get_s3_prefix(file_hash)

            # Save images locally and upload to S3
            for i, image in enumerate(images, 1):
                # Save locally
                local_path = local_cache_path / f"p{i}.png"
                image.save(local_path, "PNG")

                # Upload to S3
                s3_key = f"{s3_prefix}/p{i}.png"
                await upload_file_to_s3_async(
                    CLOUD_CACHE_BUCKET, str(local_path), s3_key
                )

            logger.info(f"Cached {len(images)} images for {file_hash}")

        except Exception as e:
            logger.error(f"Error caching images: {e}")

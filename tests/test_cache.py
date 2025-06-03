import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import MonkeyPatch

from aicapture.cache import (
    AsyncCacheInterface,
    CacheInterface,
    FileCache,
    HashUtils,
    ImageCache,
    TwoLayerCache,
)


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        yield cache_dir


@pytest.fixture
def sample_cache_data() -> Dict[str, Any]:
    """Create sample data for cache testing."""
    return {
        "key": "test_key",
        "content": "test content",
        "metadata": {"pages": 10, "size": 1024},
        "timestamp": "2024-01-01T00:00:00Z"
    }


class TestHashUtils:
    """Test cases for HashUtils utility functions."""

    def test_calculate_file_hash_existing_file(self, temp_cache_dir: Path) -> None:
        """Test hashing an existing file."""
        test_file = temp_cache_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        hash_result = HashUtils.calculate_file_hash(str(test_file))
        
        # Hash should be consistent
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 produces 64-character hex string
        
        # Same file should produce same hash
        hash_result2 = HashUtils.calculate_file_hash(str(test_file))
        assert hash_result == hash_result2

    def test_calculate_file_hash_nonexistent_file(self) -> None:
        """Test hashing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            HashUtils.calculate_file_hash("nonexistent_file.txt")

    def test_get_cache_key(self) -> None:
        """Test generating cache key from file hash and prompt."""
        file_hash = "abc123def456"
        prompt = "Test prompt for caching"
        
        cache_key = HashUtils.get_cache_key(file_hash, prompt)
        
        assert isinstance(cache_key, str)
        assert file_hash in cache_key
        assert "_" in cache_key  # Should be file_hash_prompt_hash format
        
        # Same inputs should produce same cache key
        cache_key2 = HashUtils.get_cache_key(file_hash, prompt)
        assert cache_key == cache_key2
        
        # Different prompts should produce different cache keys
        cache_key3 = HashUtils.get_cache_key(file_hash, "Different prompt")
        assert cache_key != cache_key3


class TestFileCache:
    """Test cases for FileCache implementation."""

    def test_init_default(self, temp_cache_dir: Path) -> None:
        """Test FileCache initialization with default parameters."""
        cache = FileCache(str(temp_cache_dir))
        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.cache_dir.exists()

    def test_init_creates_directory(self, temp_cache_dir: Path) -> None:
        """Test that FileCache creates cache directory if it doesn't exist."""
        new_cache_dir = temp_cache_dir / "new_cache"
        assert not new_cache_dir.exists()
        
        cache = FileCache(str(new_cache_dir))
        assert new_cache_dir.exists()
        assert cache.cache_dir == new_cache_dir

    def test_set_and_get(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test storing and retrieving data from cache."""
        cache = FileCache(str(temp_cache_dir))
        key = "test_key"
        
        # Store data
        cache.set(key, sample_cache_data)
        
        # Retrieve data
        retrieved_data = cache.get(key)
        assert retrieved_data == sample_cache_data

    def test_get_nonexistent_key(self, temp_cache_dir: Path) -> None:
        """Test retrieving data with non-existent key."""
        cache = FileCache(str(temp_cache_dir))
        result = cache.get("nonexistent_key")
        assert result is None

    def test_get_invalid_json(self, temp_cache_dir: Path) -> None:
        """Test retrieving data from corrupted cache file."""
        cache = FileCache(str(temp_cache_dir))
        key = "invalid_json_key"
        
        # Create a file with invalid JSON
        cache_file = cache.cache_dir / f"{key}.json"
        cache_file.write_text("This is not valid JSON")
        
        result = cache.get(key)
        assert result is None

    def test_invalidate(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test invalidating cache entries."""
        cache = FileCache(str(temp_cache_dir))
        key = "test_key"
        
        # Store data
        cache.set(key, sample_cache_data)
        assert cache.get(key) == sample_cache_data
        
        # Invalidate
        result = cache.invalidate(key)
        assert result is True
        assert cache.get(key) is None

    def test_invalidate_nonexistent_key(self, temp_cache_dir: Path) -> None:
        """Test invalidating non-existent cache entry."""
        cache = FileCache(str(temp_cache_dir))
        result = cache.invalidate("nonexistent_key")
        assert result is False

    def test_clear(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test clearing all cache entries."""
        cache = FileCache(str(temp_cache_dir))
        
        # Store multiple entries
        cache.set("key1", sample_cache_data)
        cache.set("key2", sample_cache_data)
        cache.set("key3", sample_cache_data)
        
        # Verify they exist
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        
        # Clear cache
        cache.clear()
        
        # Verify they're gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_cache_file_path(self, temp_cache_dir: Path) -> None:
        """Test cache file path generation."""
        cache = FileCache(str(temp_cache_dir))
        key = "test_key"
        expected_path = temp_cache_dir / f"{key}.json"
        
        cache.set(key, {"test": "data"})
        assert expected_path.exists()


class TestImageCache:
    """Test cases for ImageCache implementation."""

    def test_init_default(self, temp_cache_dir: Path) -> None:
        """Test ImageCache initialization."""
        cache = ImageCache(str(temp_cache_dir))
        assert cache.cache_dir == Path(temp_cache_dir)
        assert (cache.cache_dir / "images").exists()

    def test_set_and_get(self, temp_cache_dir: Path) -> None:
        """Test storing and retrieving images from cache."""
        cache = ImageCache(str(temp_cache_dir))
        key = "test_image"
        image_data = b"fake_image_data"
        
        # Store image
        cache.set(key, image_data)
        
        # Retrieve image
        retrieved_data = cache.get(key)
        assert retrieved_data == image_data

    def test_get_nonexistent_image(self, temp_cache_dir: Path) -> None:
        """Test retrieving non-existent image."""
        cache = ImageCache(str(temp_cache_dir))
        result = cache.get("nonexistent_image")
        assert result is None

    def test_invalidate_image(self, temp_cache_dir: Path) -> None:
        """Test invalidating image cache entries."""
        cache = ImageCache(str(temp_cache_dir))
        key = "test_image"
        image_data = b"fake_image_data"
        
        # Store image
        cache.set(key, image_data)
        assert cache.get(key) == image_data
        
        # Invalidate
        result = cache.invalidate(key)
        assert result is True
        assert cache.get(key) is None

    def test_clear_images(self, temp_cache_dir: Path) -> None:
        """Test clearing all image cache entries."""
        cache = ImageCache(str(temp_cache_dir))
        
        # Store multiple images
        cache.set("image1", b"data1")
        cache.set("image2", b"data2")
        cache.set("image3", b"data3")
        
        # Clear cache
        cache.clear()
        
        # Verify they're gone
        assert cache.get("image1") is None
        assert cache.get("image2") is None
        assert cache.get("image3") is None

    def test_image_file_path(self, temp_cache_dir: Path) -> None:
        """Test image file path generation."""
        cache = ImageCache(str(temp_cache_dir))
        key = "test_image"
        expected_path = temp_cache_dir / "images" / f"{key}.bin"
        
        cache.set(key, b"test_data")
        assert expected_path.exists()


class TestTwoLayerCache:
    """Test cases for TwoLayerCache implementation."""

    def test_init_default(self, temp_cache_dir: Path) -> None:
        """Test TwoLayerCache initialization."""
        cache = TwoLayerCache(str(temp_cache_dir))
        assert isinstance(cache.local_cache, FileCache)
        assert cache.cloud_cache is None  # Default should be None

    def test_init_with_cloud_cache(self, temp_cache_dir: Path) -> None:
        """Test TwoLayerCache initialization with cloud cache enabled."""
        with patch.dict('os.environ', {'USE_CLOUD_CACHE': 'true'}):
            cache = TwoLayerCache(str(temp_cache_dir))
            assert isinstance(cache.local_cache, FileCache)
            assert cache.cloud_cache is not None

    @pytest.mark.asyncio
    async def test_aget_local_cache_hit(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test getting data when it exists in local cache."""
        cache = TwoLayerCache(str(temp_cache_dir))
        key = "test_key"
        
        # Store in local cache
        cache.local_cache.set(key, sample_cache_data)
        
        # Retrieve
        result = await cache.aget(key)
        assert result == sample_cache_data

    @pytest.mark.asyncio
    async def test_aget_local_cache_miss_no_cloud(self, temp_cache_dir: Path) -> None:
        """Test getting data when not in local cache and no cloud cache."""
        cache = TwoLayerCache(str(temp_cache_dir))
        result = await cache.aget("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_aset_local_only(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test setting data in local cache only."""
        cache = TwoLayerCache(str(temp_cache_dir))
        key = "test_key"
        
        await cache.aset(key, sample_cache_data)
        
        # Should be in local cache
        assert cache.local_cache.get(key) == sample_cache_data

    @pytest.mark.asyncio
    async def test_ainvalidate_local_only(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test invalidating data from local cache only."""
        cache = TwoLayerCache(str(temp_cache_dir))
        key = "test_key"
        
        # Store data
        await cache.aset(key, sample_cache_data)
        assert cache.local_cache.get(key) == sample_cache_data
        
        # Invalidate
        result = await cache.ainvalidate(key)
        assert result is True
        assert cache.local_cache.get(key) is None

    @pytest.mark.asyncio
    async def test_aclear_local_only(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test clearing local cache only."""
        cache = TwoLayerCache(str(temp_cache_dir))
        
        # Store data
        await cache.aset("key1", sample_cache_data)
        await cache.aset("key2", sample_cache_data)
        
        # Clear
        await cache.aclear()
        
        # Verify cleared
        assert cache.local_cache.get("key1") is None
        assert cache.local_cache.get("key2") is None

    def test_get_sync_interface(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test synchronous interface methods."""
        cache = TwoLayerCache(str(temp_cache_dir))
        key = "test_key"
        
        # Test set and get
        cache.set(key, sample_cache_data)
        result = cache.get(key)
        assert result == sample_cache_data
        
        # Test invalidate
        assert cache.invalidate(key) is True
        assert cache.get(key) is None

    def test_clear_sync_interface(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test synchronous clear interface."""
        cache = TwoLayerCache(str(temp_cache_dir))
        
        cache.set("key1", sample_cache_data)
        cache.set("key2", sample_cache_data)
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class MockCloudCache:
    """Mock cloud cache for testing TwoLayerCache with cloud functionality."""
    
    def __init__(self):
        self.data = {}
    
    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)
    
    async def aset(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
    
    async def ainvalidate(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    async def aclear(self) -> None:
        self.data.clear()


class TestTwoLayerCacheWithCloud:
    """Test TwoLayerCache with cloud cache functionality."""

    @pytest.fixture
    def cache_with_cloud(self, temp_cache_dir: Path) -> TwoLayerCache:
        """Create a TwoLayerCache with mock cloud cache."""
        cache = TwoLayerCache(str(temp_cache_dir))
        cache.cloud_cache = MockCloudCache()
        return cache

    @pytest.mark.asyncio
    async def test_aget_cloud_cache_hit(self, cache_with_cloud: TwoLayerCache, sample_cache_data: Dict[str, Any]) -> None:
        """Test getting data from cloud cache when not in local."""
        key = "test_key"
        
        # Store in cloud cache only
        await cache_with_cloud.cloud_cache.aset(key, sample_cache_data)
        
        # Retrieve (should get from cloud and cache locally)
        result = await cache_with_cloud.aget(key)
        assert result == sample_cache_data
        
        # Should now be in local cache too
        assert cache_with_cloud.local_cache.get(key) == sample_cache_data

    @pytest.mark.asyncio
    async def test_aset_with_cloud_cache(self, cache_with_cloud: TwoLayerCache, sample_cache_data: Dict[str, Any]) -> None:
        """Test setting data in both local and cloud cache."""
        key = "test_key"
        
        await cache_with_cloud.aset(key, sample_cache_data)
        
        # Should be in both caches
        assert cache_with_cloud.local_cache.get(key) == sample_cache_data
        assert await cache_with_cloud.cloud_cache.aget(key) == sample_cache_data

    @pytest.mark.asyncio
    async def test_ainvalidate_with_cloud_cache(self, cache_with_cloud: TwoLayerCache, sample_cache_data: Dict[str, Any]) -> None:
        """Test invalidating data from both caches."""
        key = "test_key"
        
        # Store in both caches
        await cache_with_cloud.aset(key, sample_cache_data)
        
        # Invalidate
        result = await cache_with_cloud.ainvalidate(key)
        assert result is True
        
        # Should be gone from both caches
        assert cache_with_cloud.local_cache.get(key) is None
        assert await cache_with_cloud.cloud_cache.aget(key) is None

    @pytest.mark.asyncio
    async def test_aclear_with_cloud_cache(self, cache_with_cloud: TwoLayerCache, sample_cache_data: Dict[str, Any]) -> None:
        """Test clearing both caches."""
        # Store data in both caches
        await cache_with_cloud.aset("key1", sample_cache_data)
        await cache_with_cloud.aset("key2", sample_cache_data)
        
        # Clear
        await cache_with_cloud.aclear()
        
        # Should be gone from both caches
        assert cache_with_cloud.local_cache.get("key1") is None
        assert cache_with_cloud.local_cache.get("key2") is None
        assert await cache_with_cloud.cloud_cache.aget("key1") is None
        assert await cache_with_cloud.cloud_cache.aget("key2") is None

    @pytest.mark.asyncio
    async def test_cloud_cache_error_handling(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test error handling when cloud cache fails."""
        cache = TwoLayerCache(str(temp_cache_dir))
        
        # Create a mock cloud cache that raises exceptions
        mock_cloud_cache = AsyncMock()
        mock_cloud_cache.aget.side_effect = Exception("Cloud cache error")
        mock_cloud_cache.aset.side_effect = Exception("Cloud cache error")
        mock_cloud_cache.ainvalidate.side_effect = Exception("Cloud cache error")
        mock_cloud_cache.aclear.side_effect = Exception("Cloud cache error")
        
        cache.cloud_cache = mock_cloud_cache
        
        key = "test_key"
        
        # Operations should still work with local cache
        await cache.aset(key, sample_cache_data)
        assert cache.local_cache.get(key) == sample_cache_data
        
        result = await cache.aget(key)
        assert result == sample_cache_data
        
        assert await cache.ainvalidate(key) is True
        assert cache.local_cache.get(key) is None


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
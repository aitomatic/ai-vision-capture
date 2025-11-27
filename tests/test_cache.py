import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from aicapture.cache import FileCache, HashUtils, ImageCache, S3Cache, TwoLayerCache


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
        "timestamp": "2024-01-01T00:00:00Z",
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
        assert cache.cache_dir == Path(temp_cache_dir) / "images"
        assert cache.cache_dir.exists()

    def test_init_with_cloud_bucket(self, temp_cache_dir: Path) -> None:
        """Test ImageCache initialization with cloud bucket."""
        cache = ImageCache(str(temp_cache_dir), cloud_bucket="test-bucket")
        assert cache.cloud_bucket == "test-bucket"
        assert cache.cache_dir == Path(temp_cache_dir) / "images"

    def test_get_local_cache_path(self, temp_cache_dir: Path) -> None:
        """Test local cache path generation."""
        cache = ImageCache(str(temp_cache_dir))
        file_hash = "abc123"
        path = cache._get_local_cache_path(file_hash)
        expected_path = Path(temp_cache_dir) / "images" / file_hash
        assert path == expected_path

    def test_get_s3_prefix(self, temp_cache_dir: Path) -> None:
        """Test S3 prefix generation."""
        cache = ImageCache(str(temp_cache_dir))
        file_hash = "abc123"
        prefix = cache._get_s3_prefix(file_hash)
        expected_prefix = f"production/images/raw_images/{file_hash}"
        assert prefix == expected_prefix

    def test_validate_cache_exists(self, temp_cache_dir: Path) -> None:
        """Test cache validation when cache exists."""
        cache = ImageCache(str(temp_cache_dir))
        file_hash = "test_hash"
        cache_path = cache._get_local_cache_path(file_hash)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Create some test image files
        for i in range(3):
            (cache_path / f"page_{i}.png").touch()

        assert cache._validate_cache(cache_path, 3) is True
        assert cache._validate_cache(cache_path, 5) is False  # Not enough files

    def test_validate_cache_missing(self, temp_cache_dir: Path) -> None:
        """Test cache validation when cache doesn't exist."""
        cache = ImageCache(str(temp_cache_dir))
        file_hash = "nonexistent_hash"
        cache_path = cache._get_local_cache_path(file_hash)

        assert cache._validate_cache(cache_path, 1) is False


class TestTwoLayerCache:
    """Test cases for TwoLayerCache implementation."""

    def test_init_file_cache_only(self, temp_cache_dir: Path) -> None:
        """Test TwoLayerCache initialization with file cache only."""
        file_cache = FileCache(str(temp_cache_dir))
        cache = TwoLayerCache(file_cache, None)
        assert cache.file_cache == file_cache
        assert cache.s3_cache is None

    def test_init_with_s3_cache(self, temp_cache_dir: Path) -> None:
        """Test TwoLayerCache initialization with S3 cache."""
        file_cache = FileCache(str(temp_cache_dir))
        s3_cache = S3Cache("test-bucket", "test-prefix")
        cache = TwoLayerCache(file_cache, s3_cache)
        assert cache.file_cache == file_cache
        assert cache.s3_cache == s3_cache

    @pytest.mark.asyncio
    async def test_get_file_cache_hit(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test getting data when it exists in file cache."""
        file_cache = FileCache(str(temp_cache_dir))
        cache = TwoLayerCache(file_cache, None)
        key = "test_key"

        # Store in file cache
        file_cache.set(key, sample_cache_data)

        # Retrieve
        result = await cache.get(key)
        assert result == sample_cache_data

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, temp_cache_dir: Path) -> None:
        """Test getting data when not in cache."""
        file_cache = FileCache(str(temp_cache_dir))
        cache = TwoLayerCache(file_cache, None)
        result = await cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_file_cache_only(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test setting data in file cache only."""
        file_cache = FileCache(str(temp_cache_dir))
        cache = TwoLayerCache(file_cache, None)
        key = "test_key"

        await cache.set(key, sample_cache_data)

        # Should be in file cache
        assert file_cache.get(key) == sample_cache_data

    @pytest.mark.asyncio
    async def test_invalidate_file_cache_only(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test invalidating data from file cache only."""
        file_cache = FileCache(str(temp_cache_dir))
        cache = TwoLayerCache(file_cache, None)
        key = "test_key"

        # Store data
        await cache.set(key, sample_cache_data)
        assert file_cache.get(key) == sample_cache_data

        # Invalidate
        await cache.invalidate(key)
        assert file_cache.get(key) is None

    @pytest.mark.asyncio
    async def test_clear_file_cache_only(self, temp_cache_dir: Path, sample_cache_data: Dict[str, Any]) -> None:
        """Test clearing file cache only."""
        file_cache = FileCache(str(temp_cache_dir))
        cache = TwoLayerCache(file_cache, None)

        # Store data
        await cache.set("key1", sample_cache_data)
        await cache.set("key2", sample_cache_data)

        # Clear
        await cache.clear()

        # Verify cleared
        assert file_cache.get("key1") is None
        assert file_cache.get("key2") is None


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

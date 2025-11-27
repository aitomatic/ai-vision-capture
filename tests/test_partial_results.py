import json
import shutil
from pathlib import Path
from typing import Dict, Generator, List

import pytest
from loguru import logger

from aicapture.vision_models import create_default_vision_model
from aicapture.vision_parser import VisionParser

# Define test PDF path
TEST_PDF_PATH = Path(__file__).parent / "sample" / "pdfs" / "sample.pdf"


@pytest.fixture
def test_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for tests."""
    cache_dir = Path(__file__).parent / ".test_cache"
    cache_dir.mkdir(exist_ok=True)
    yield cache_dir
    # Clean up after tests
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.fixture
def vision_parser(test_cache_dir: Path) -> VisionParser:
    """Create a VisionParser instance for testing."""
    parser = VisionParser(
        vision_model=create_default_vision_model(),
        cache_dir=str(test_cache_dir),
        invalidate_cache=True,
    )
    return parser


@pytest.fixture
def sample_pages() -> List[Dict]:
    """Create sample page data for testing."""
    return [
        {
            "page_number": 1,
            "page_content": "Test content for page 1",
            "page_hash": "hash1",
            "page_objects": [{"md": "Test content for page 1", "has_image": False}],
        },
        {
            "page_number": 2,
            "page_content": "Test content for page 2",
            "page_hash": "hash2",
            "page_objects": [{"md": "Test content for page 2", "has_image": False}],
        },
    ]


@pytest.mark.asyncio
async def test_save_partial_results(
    vision_parser: VisionParser, sample_pages: List[Dict], test_cache_dir: Path
) -> None:
    """Test saving partial results to cache."""
    # Generate a test cache key
    cache_key = "test_cache_key_123"

    # Save partial results
    await vision_parser._save_partial_results(cache_key, sample_pages)

    # Check that the partial results file was created
    partial_cache_path = vision_parser._get_partial_cache_path(cache_key)
    assert partial_cache_path.exists(), "Partial results file should exist"

    # Verify file content
    with open(partial_cache_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    # Check that the data was saved correctly
    assert "1" in saved_data, "Page 1 should be in the saved data"
    assert "2" in saved_data, "Page 2 should be in the saved data"
    assert saved_data["1"]["page_content"] == "Test content for page 1"
    assert saved_data["2"]["page_content"] == "Test content for page 2"


@pytest.mark.asyncio
async def test_load_partial_results(
    vision_parser: VisionParser, sample_pages: List[Dict], test_cache_dir: Path
) -> None:
    """Test loading partial results from cache."""
    # Generate a test cache key
    cache_key = "test_cache_key_456"

    # Save sample data to load later
    await vision_parser._save_partial_results(cache_key, sample_pages)

    # Load the partial results
    loaded_results = await vision_parser._load_partial_results(cache_key)

    # Check that the data was loaded correctly
    assert len(loaded_results) == 2, "Should have loaded 2 pages"
    assert 1 in loaded_results, "Page 1 should be in the loaded results"
    assert 2 in loaded_results, "Page 2 should be in the loaded results"
    assert loaded_results[1]["page_content"] == "Test content for page 1"
    assert loaded_results[2]["page_content"] == "Test content for page 2"


@pytest.mark.asyncio
async def test_load_partial_results_nonexistent(vision_parser: VisionParser, test_cache_dir: Path) -> None:
    """Test loading partial results when the file doesn't exist."""
    # Generate a non-existent cache key
    cache_key = "nonexistent_cache_key"

    # Try to load non-existent partial results
    loaded_results = await vision_parser._load_partial_results(cache_key)

    # Should return an empty dict
    assert isinstance(loaded_results, dict), "Should return a dict"
    assert len(loaded_results) == 0, "Should return an empty dict"


@pytest.mark.asyncio
async def test_save_load_update_cycle(
    vision_parser: VisionParser, sample_pages: List[Dict], test_cache_dir: Path
) -> None:
    """Test the full cycle of saving, loading, updating, and saving again."""
    cache_key = "test_cycle_key"

    # Save initial pages (just page 1)
    initial_pages = [sample_pages[0]]
    await vision_parser._save_partial_results(cache_key, initial_pages)

    # Load the partial results
    loaded_results = await vision_parser._load_partial_results(cache_key)
    assert len(loaded_results) == 1, "Should have loaded 1 page"
    assert 1 in loaded_results, "Page 1 should be in the loaded results"

    # Save additional pages (just page 2)
    additional_pages = [sample_pages[1]]
    await vision_parser._save_partial_results(cache_key, additional_pages)

    # Load the updated partial results
    updated_results = await vision_parser._load_partial_results(cache_key)
    assert len(updated_results) == 2, "Should have loaded 2 pages"
    assert 1 in updated_results, "Page 1 should still be in the results"
    assert 2 in updated_results, "Page 2 should now be in the results"


@pytest.mark.asyncio
async def test_overwrite_existing_page(
    vision_parser: VisionParser, sample_pages: List[Dict], test_cache_dir: Path
) -> None:
    """Test that saving a page with the same number overwrites the existing one."""
    cache_key = "test_overwrite_key"

    # Save initial page
    await vision_parser._save_partial_results(cache_key, [sample_pages[0]])

    # Create a modified version of page 1
    modified_page = sample_pages[0].copy()
    modified_page["page_content"] = "Modified content for page 1"

    # Save the modified page
    await vision_parser._save_partial_results(cache_key, [modified_page])

    # Load the results and check that page 1 was overwritten
    loaded_results = await vision_parser._load_partial_results(cache_key)
    assert len(loaded_results) == 1, "Should still have 1 page"
    assert loaded_results[1]["page_content"] == "Modified content for page 1"


@pytest.mark.asyncio
async def test_invalid_json_handling(vision_parser: VisionParser, test_cache_dir: Path) -> None:
    """Test handling of invalid JSON in the partial results file."""
    cache_key = "test_invalid_json"

    # Create an invalid JSON file
    partial_cache_path = vision_parser._get_partial_cache_path(cache_key)
    with open(partial_cache_path, "w", encoding="utf-8") as f:
        f.write("This is not valid JSON")

    # Try to load the invalid file
    loaded_results = await vision_parser._load_partial_results(cache_key)

    # Should return an empty dict
    assert isinstance(loaded_results, dict), "Should return a dict"
    assert len(loaded_results) == 0, "Should return an empty dict"


@pytest.mark.asyncio
async def test_partial_results_with_real_pdf(vision_parser: VisionParser, test_cache_dir: Path) -> None:
    """Test partial results with a real PDF processing flow."""
    # Ensure the test PDF exists
    assert TEST_PDF_PATH.exists(), f"Test PDF not found at {TEST_PDF_PATH}"

    try:
        # Get file hash of the test PDF
        # pdf_file = Path(TEST_PDF_PATH)
        result = await vision_parser._validate_and_setup(str(TEST_PDF_PATH))
        (
            pdf_file,
            file_hash,
        ) = result  # _validate_and_setup returns (pdf_file, file_hash)
        cache_key = file_hash  # For simplicity, just use the file hash as the cache key

        # Create mock pages that would come from processing
        mock_pages = [
            {
                "page_number": 1,
                "page_content": "Content from the first page of the sample PDF",
                "page_hash": "mock_hash1",
                "page_objects": [{"md": "Sample content", "has_image": False}],
            },
            {
                "page_number": 2,
                "page_content": "Content from the second page of the sample PDF",
                "page_hash": "mock_hash2",
                "page_objects": [{"md": "More sample content", "has_image": False}],
            },
        ]

        # Save partial results for the PDF
        await vision_parser._save_partial_results(cache_key, mock_pages)

        # Load partial results
        loaded_results = await vision_parser._load_partial_results(cache_key)

        # Check loaded results
        assert len(loaded_results) == 2, "Should have loaded 2 pages"
        assert 1 in loaded_results, "Page 1 should be in the loaded results"
        assert 2 in loaded_results, "Page 2 should be in the loaded results"

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        raise


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

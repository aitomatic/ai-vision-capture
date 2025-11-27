from pathlib import Path

import pytest

from aicapture.vision_models import create_default_vision_model
from aicapture.vision_parser import VisionParser

# Define test image path
TEST_IMAGE_PATH = Path(__file__).parent / "sample" / "images" / "logic.png"


@pytest.fixture
async def vision_parser() -> VisionParser:
    """Create a VisionParser instance with real OpenAI model."""
    parser = VisionParser(vision_model=create_default_vision_model(), cache_dir="tests/.cache")
    return parser


@pytest.mark.asyncio
async def test_process_image_real(vision_parser: VisionParser) -> None:
    """Test real image processing with OpenAI vision model."""
    # Ensure the test image exists
    assert TEST_IMAGE_PATH.exists(), f"Test image not found at {TEST_IMAGE_PATH}"

    # Process the image
    result = await vision_parser.process_image_async(str(TEST_IMAGE_PATH))

    # Validate the result structure
    assert "file_object" in result
    assert "file_name" in result["file_object"]
    assert "pages" in result["file_object"]
    assert len(result["file_object"]["pages"]) == 1

    # Validate page content
    page = result["file_object"]["pages"][0]
    assert "page_content" in page
    assert len(page["page_content"]) > 0

    # Print the extracted content for manual verification
    print("\nExtracted content from image:")
    print(page["page_content"])


if __name__ == "__main__":
    import asyncio

    parser = VisionParser(
        vision_model=create_default_vision_model(),
        cache_dir="tests/.cache",
        invalidate_cache=True,
        prompt="You are an expert electrical engineer. The uploaded image is a logical diagram that processes input signals (sensors) on the left-hand side and produces an output result on the right-hand side. There is an alarm in the diagram. Identify all possible input signals (dependencies) that contribute to triggering this alarm. Provide the exact sensor names and any relevant identifiers (e.g., TAG, RF LOGICA) for each dependency. Be precise",  # noqa
    )
    result = asyncio.run(parser.process_image_async(str(TEST_IMAGE_PATH)))
    print(result)

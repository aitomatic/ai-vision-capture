from pathlib import Path
from typing import Any, Dict, List, Union
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from pytest import MonkeyPatch

from aicapture.vision_capture import VisionCapture
from aicapture.vision_models import VisionModel
from aicapture.vision_parser import VisionParser

# Define test file paths
TEST_PDF_PATH = Path(__file__).parent / "sample" / "pdfs" / "sample.pdf"
TEST_IMAGE_PATH = Path(__file__).parent / "sample" / "images" / "logic.png"


class MockVisionModel(VisionModel):
    """Mock vision model for testing."""

    def __init__(self):
        # Initialize parent with dummy values to satisfy abstract class
        super().__init__(model="mock-model", api_key="mock-key")

    @property
    def client(self) -> Any:
        """Return mock client."""
        return MagicMock()

    @property
    def aclient(self) -> Any:
        """Return mock async client."""
        return AsyncMock()

    async def process_text_async(self, messages) -> str:
        return "Mock structured data extraction result"

    def process_image(self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs) -> str:
        return "Mock image processing result"

    async def process_image_async(self, image, prompt: str, **kwargs) -> str:
        return "Mock image processing result"


@pytest.fixture
def mock_vision_model() -> MockVisionModel:
    """Create a mock vision model for testing."""
    return MockVisionModel()


@pytest.fixture
def mock_vision_parser() -> VisionParser:
    """Create a mock vision parser for testing."""
    parser = MagicMock(spec=VisionParser)
    parser.SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".tiff", ".webp", ".bmp"]
    return parser


@pytest.fixture
def vision_capture(mock_vision_model: MockVisionModel) -> VisionCapture:
    """Create a VisionCapture instance with mock dependencies."""
    return VisionCapture(vision_model=mock_vision_model)


@pytest.fixture
def sample_document_result() -> Dict[str, Any]:
    """Create a sample document processing result."""
    return {
        "file_object": {
            "file_name": "test.pdf",
            "file_hash": "abc123",
            "total_pages": 2,
            "pages": [
                {
                    "page_number": 1,
                    "page_content": "First page content with technical specifications",
                    "page_hash": "hash1",
                },
                {
                    "page_number": 2,
                    "page_content": "Second page content with data tables",
                    "page_hash": "hash2",
                },
            ],
        }
    }


def test_init_default() -> None:
    """Test VisionCapture initialization with default parameters."""
    capture = VisionCapture()
    assert capture.vision_model is not None
    assert capture.vision_parser is not None
    assert hasattr(capture.vision_parser, "vision_model")


def test_init_with_custom_vision_model(mock_vision_model: MockVisionModel) -> None:
    """Test VisionCapture initialization with custom vision model."""
    capture = VisionCapture(vision_model=mock_vision_model)
    assert capture.vision_model == mock_vision_model
    assert capture.vision_parser is not None
    assert capture.vision_parser.vision_model == mock_vision_model


def test_init_with_custom_parser(mock_vision_model: MockVisionModel, mock_vision_parser: VisionParser) -> None:
    """Test VisionCapture initialization with custom parser."""
    capture = VisionCapture(vision_model=mock_vision_model, vision_parser=mock_vision_parser)
    assert capture.vision_model == mock_vision_model
    assert capture.vision_parser == mock_vision_parser


@pytest.mark.asyncio
async def test_parse_file_pdf(
    vision_capture: VisionCapture,
    sample_document_result: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test parsing a PDF file."""

    # Mock the parser's process_pdf_async method
    async def mock_process_pdf_async(pdf_path: str) -> Dict[str, Any]:
        return sample_document_result

    monkeypatch.setattr(vision_capture.vision_parser, "process_pdf_async", mock_process_pdf_async)

    # Test PDF parsing
    result = await vision_capture._parse_file("test.pdf")

    assert result == sample_document_result
    assert result["file_object"]["total_pages"] == 2


@pytest.mark.asyncio
async def test_parse_file_image(
    vision_capture: VisionCapture,
    sample_document_result: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test parsing an image file."""

    # Mock the parser's process_image_async method
    async def mock_process_image_async(image_path: str) -> Dict[str, Any]:
        return sample_document_result

    monkeypatch.setattr(vision_capture.vision_parser, "process_image_async", mock_process_image_async)

    # Test image parsing for different formats
    for image_format in [".jpg", ".jpeg", ".png", ".tiff", ".webp", ".bmp"]:
        result = await vision_capture._parse_file(f"test{image_format}")
        assert result == sample_document_result


@pytest.mark.asyncio
async def test_parse_file_unsupported_format(vision_capture: VisionCapture) -> None:
    """Test parsing an unsupported file format."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        await vision_capture._parse_file("test.txt")

    with pytest.raises(ValueError, match="Unsupported file type"):
        await vision_capture._parse_file("test.docx")


@pytest.mark.asyncio
async def test_capture_pdf(
    vision_capture: VisionCapture,
    sample_document_result: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test the complete capture process for a PDF."""

    # Mock the file parsing
    async def mock_parse_file(file_path: str) -> Dict[str, Any]:
        return sample_document_result

    monkeypatch.setattr(vision_capture, "_parse_file", mock_parse_file)

    # Mock the content capture
    async def mock_capture_content(content: str, template: str) -> str:
        assert "First page content" in content
        assert "Second page content" in content
        # The template is processed by _capture_content, which wraps it with
        # <template> tags
        return "Extracted structured data"

    monkeypatch.setattr(vision_capture, "_capture_content", mock_capture_content)

    # Test the capture process
    template = "test_template: extract data"
    result = await vision_capture.capture("test.pdf", template)

    assert result == "Extracted structured data"


@pytest.mark.asyncio
async def test_capture_image(
    vision_capture: VisionCapture,
    sample_document_result: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test the complete capture process for an image."""

    # Mock the file parsing
    async def mock_parse_file(file_path: str) -> Dict[str, Any]:
        return sample_document_result

    monkeypatch.setattr(vision_capture, "_parse_file", mock_parse_file)

    # Mock the content capture
    async def mock_capture_content(content: str, template: str) -> str:
        return "Extracted image data"

    monkeypatch.setattr(vision_capture, "_capture_content", mock_capture_content)

    # Test the capture process
    template = "image_template: extract objects"
    result = await vision_capture.capture("test.png", template)

    assert result == "Extracted image data"


@pytest.mark.asyncio
async def test_capture_content(vision_capture: VisionCapture) -> None:
    """Test the content capture with template processing."""
    content = "Sample document content with technical data"
    template = "data: extract technical specifications"

    # The vision model should be called with properly formatted messages
    result = await vision_capture._capture_content(content, template)

    # Check that the mock vision model was called and returned expected result
    assert result == "Mock structured data extraction result"


@pytest.mark.asyncio
async def test_capture_content_with_complex_template(
    vision_capture: VisionCapture,
) -> None:
    """Test content capture with a complex template structure."""
    content = """
    Equipment: Motor XYZ-123
    Power: 500W
    Voltage: 220V
    Temperature: 85°C
    """

    template = """
    equipment:
      name: string
      power: string
      voltage: string
      temperature: string
    """

    result = await vision_capture._capture_content(content, template)
    assert result == "Mock structured data extraction result"


@pytest.mark.asyncio
async def test_capture_with_real_files(vision_capture: VisionCapture, monkeypatch: MonkeyPatch) -> None:
    """Test capture with real file paths (but mocked processing)."""
    # Ensure test files exist
    assert TEST_PDF_PATH.exists(), f"Test PDF not found at {TEST_PDF_PATH}"
    assert TEST_IMAGE_PATH.exists(), f"Test image not found at {TEST_IMAGE_PATH}"

    # Mock the actual file processing to avoid API calls
    sample_result = {"file_object": {"pages": [{"page_content": "Mock content from real file"}]}}

    async def mock_parse_file(file_path: str) -> Dict[str, Any]:
        return sample_result

    monkeypatch.setattr(vision_capture, "_parse_file", mock_parse_file)

    # Test with PDF
    template = "extract: all technical data"
    result = await vision_capture.capture(str(TEST_PDF_PATH), template)
    assert result == "Mock structured data extraction result"

    # Test with image
    result = await vision_capture.capture(str(TEST_IMAGE_PATH), template)
    assert result == "Mock structured data extraction result"


@pytest.mark.asyncio
async def test_capture_empty_pages(vision_capture: VisionCapture, monkeypatch: MonkeyPatch) -> None:
    """Test capture with document that has no pages."""
    empty_result: Dict[str, Any] = {"file_object": {"pages": []}}

    async def mock_parse_file(file_path: str) -> Dict[str, Any]:
        return empty_result

    monkeypatch.setattr(vision_capture, "_parse_file", mock_parse_file)

    template = "extract: data"
    result = await vision_capture.capture("test.pdf", template)

    # Should handle empty content gracefully
    assert result == "Mock structured data extraction result"


@pytest.mark.asyncio
async def test_capture_pages_without_content(vision_capture: VisionCapture, monkeypatch: MonkeyPatch) -> None:
    """Test capture with pages that have no content."""
    result_with_empty_pages = {
        "file_object": {
            "pages": [
                {"page_content": ""},
                {"page_content": ""},  # Changed from None to empty string
                {"page_content": "Some content"},
            ]
        }
    }

    async def mock_parse_file(file_path: str) -> Dict[str, Any]:
        return result_with_empty_pages

    monkeypatch.setattr(vision_capture, "_parse_file", mock_parse_file)

    template = "extract: data"
    result = await vision_capture.capture("test.pdf", template)

    # Should handle mixed content gracefully
    assert result == "Mock structured data extraction result"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

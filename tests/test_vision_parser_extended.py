import asyncio
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image

from aicapture.vision_models import VisionModel
from aicapture.vision_parser import VisionParser


class MockVisionModel(VisionModel):
    """Mock vision model for testing."""

    def __init__(self, response: str = "Mock response"):
        self.response = response
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
        return self.response

    def process_image(self, image, prompt: str, **kwargs) -> str:
        return f"Mock image analysis: {self.response}"

    async def process_image_async(self, image, prompt: str, **kwargs) -> str:
        return f"Mock image analysis: {self.response}"


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir)
        yield cache_dir


@pytest.fixture
def mock_vision_model() -> MockVisionModel:
    """Create a mock vision model for testing."""
    return MockVisionModel("Test content extracted")


@pytest.fixture
def vision_parser(
    temp_cache_dir: Path, mock_vision_model: MockVisionModel
) -> VisionParser:
    """Create a VisionParser instance for testing."""
    return VisionParser(
        vision_model=mock_vision_model,
        cache_dir=str(temp_cache_dir),
        invalidate_cache=True,
    )


@pytest.fixture
def test_pdf_path() -> str:
    """Get path to test PDF."""
    return str(Path(__file__).parent / "sample" / "pdfs" / "sample.pdf")


@pytest.fixture
def test_image_path() -> str:
    """Get path to test image."""
    return str(Path(__file__).parent / "sample" / "images" / "logic.png")


@pytest.fixture
def temp_image_file(temp_cache_dir: Path) -> str:
    """Create a temporary image file for testing."""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='blue')
    image_path = temp_cache_dir / "test_image.png"
    img.save(image_path)
    return str(image_path)


class TestVisionParserInitialization:
    """Test VisionParser initialization and configuration."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        parser = VisionParser()
        assert parser.vision_model is not None
        assert parser.dpi == 333
        assert parser.cache_dir is not None
        assert not parser.invalidate_cache

    def test_init_custom_config(
        self, temp_cache_dir: Path, mock_vision_model: MockVisionModel
    ) -> None:
        """Test initialization with custom parameters."""
        custom_prompt = "Custom extraction prompt"
        parser = VisionParser(
            vision_model=mock_vision_model,
            cache_dir=str(temp_cache_dir),
            dpi=400,
            prompt=custom_prompt,
            invalidate_cache=True,
        )

        assert parser.vision_model == mock_vision_model
        assert parser.cache_dir == Path(temp_cache_dir)
        assert parser.dpi == 400
        assert parser.prompt == custom_prompt
        assert parser.invalidate_cache

    def test_supported_image_formats(self, vision_parser: VisionParser) -> None:
        """Test supported image formats."""
        expected_formats = [".jpg", ".jpeg", ".png", ".tiff", ".webp", ".bmp"]
        assert vision_parser.SUPPORTED_IMAGE_FORMATS == expected_formats


class TestVisionParserValidation:
    """Test file validation methods."""

    @pytest.mark.asyncio
    async def test_validate_and_setup_pdf(
        self, vision_parser: VisionParser, test_pdf_path: str
    ) -> None:
        """Test validation and setup for PDF files."""
        result = await vision_parser._validate_and_setup(test_pdf_path)
        pdf_file, file_hash = result

        assert pdf_file.exists()
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hash length

    @pytest.mark.asyncio
    async def test_validate_and_setup_nonexistent_file(
        self, vision_parser: VisionParser
    ) -> None:
        """Test validation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            await vision_parser._validate_and_setup("nonexistent_file.pdf")

    def test_validate_image_format_valid(self, vision_parser: VisionParser) -> None:
        """Test validation of valid image formats."""
        valid_formats = [
            "test.jpg",
            "test.jpeg",
            "test.png",
            "test.tiff",
            "test.webp",
            "test.bmp",
        ]

        for image_path in valid_formats:
            assert vision_parser._validate_image_format(image_path) is True

    def test_validate_image_format_invalid(self, vision_parser: VisionParser) -> None:
        """Test validation of invalid image formats."""
        invalid_formats = ["test.gif", "test.svg", "test.txt", "test.pdf"]

        for image_path in invalid_formats:
            assert vision_parser._validate_image_format(image_path) is False

    def test_validate_image_format_case_insensitive(
        self, vision_parser: VisionParser
    ) -> None:
        """Test that image format validation is case insensitive."""
        formats = ["test.JPG", "test.JPEG", "test.PNG", "test.TIFF"]

        for image_path in formats:
            assert vision_parser._validate_image_format(image_path) is True


class TestVisionParserImageProcessing:
    """Test image processing methods."""

    @pytest.mark.asyncio
    async def test_process_image_async(
        self, vision_parser: VisionParser, temp_image_file: str
    ) -> None:
        """Test async image processing."""
        result = await vision_parser.process_image_async(temp_image_file)

        assert "file_object" in result
        assert "file_name" in result["file_object"]
        assert "pages" in result["file_object"]
        assert len(result["file_object"]["pages"]) == 1

        page = result["file_object"]["pages"][0]
        assert page["page_number"] == 1
        assert "page_content" in page
        assert "page_hash" in page

    def test_process_image_sync(
        self, vision_parser: VisionParser, temp_image_file: str
    ) -> None:
        """Test synchronous image processing."""
        result = vision_parser.process_image(temp_image_file)

        assert "file_object" in result
        assert result["file_object"]["file_name"] == Path(temp_image_file).name

    @pytest.mark.asyncio
    async def test_process_image_invalid_format(
        self, vision_parser: VisionParser, temp_cache_dir: Path
    ) -> None:
        """Test processing image with invalid format."""
        # Create a text file with image extension
        invalid_file = temp_cache_dir / "fake_image.jpg"
        invalid_file.write_text("This is not an image")

        with pytest.raises(Exception):  # Should raise an error when trying to process
            await vision_parser.process_image_async(str(invalid_file))

    @pytest.mark.asyncio
    async def test_process_image_nonexistent(self, vision_parser: VisionParser) -> None:
        """Test processing non-existent image."""
        with pytest.raises(FileNotFoundError):
            await vision_parser.process_image_async("nonexistent_image.jpg")


class TestVisionParserPDFProcessing:
    """Test PDF processing methods."""

    @pytest.mark.asyncio
    async def test_process_pdf_async(
        self, vision_parser: VisionParser, test_pdf_path: str
    ) -> None:
        """Test async PDF processing."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=2)  # 2 pages
        mock_doc.__iter__ = Mock(return_value=iter([MagicMock(), MagicMock()]))

        mock_page = MagicMock()
        mock_page.number = 0
        mock_page.get_pixmap.return_value.pil_tobytes.return_value = b"fake_image_data"
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        with patch('aicapture.vision_parser.fitz.open', return_value=mock_doc):
            result = await vision_parser.process_pdf_async(test_pdf_path)

            assert "file_object" in result
            assert "pages" in result["file_object"]

    def test_process_pdf_sync(
        self, vision_parser: VisionParser, test_pdf_path: str
    ) -> None:
        """Test synchronous PDF processing."""
        with patch('aicapture.vision_parser.fitz.open') as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = Mock(return_value=1)
            mock_fitz.return_value = mock_doc

            # Mock the async method
            with patch.object(
                vision_parser, 'process_pdf_async', return_value={"test": "result"}
            ):
                result = vision_parser.process_pdf(test_pdf_path)
                assert result == {"test": "result"}

    @pytest.mark.asyncio
    async def test_process_pdf_with_cache(
        self, vision_parser: VisionParser, test_pdf_path: str
    ) -> None:
        """Test PDF processing with caching."""
        # First call should process the PDF
        with patch('aicapture.vision_parser.fitz.open') as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = Mock(return_value=1)
            mock_fitz.return_value = mock_doc

            # Set up cache to return None first (cache miss)
            vision_parser.cache.get = Mock(return_value=None)
            vision_parser.cache.set = Mock()

            with patch.object(vision_parser, '_process_pdf_pages', return_value=[]):
                await vision_parser.process_pdf_async(test_pdf_path)

                # Should have called cache.set
                vision_parser.cache.set.assert_called()

    @pytest.mark.asyncio
    async def test_process_pdf_metadata_extraction(
        self, vision_parser: VisionParser, test_pdf_path: str
    ) -> None:
        """Test PDF metadata extraction."""
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "creator": "Test Creator",
        }

        # Mock first page for size info
        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        with patch('aicapture.vision_parser.fitz.open', return_value=mock_doc):
            with patch.object(vision_parser, '_process_pdf_pages', return_value=[]):
                result = await vision_parser.process_pdf_async(test_pdf_path)

                metadata = result.get("metadata", {})
                assert metadata.get("title") == "Test Document"
                assert metadata.get("author") == "Test Author"
                assert metadata.get("page_width") == 612
                assert metadata.get("page_height") == 792


class TestVisionParserFolderProcessing:
    """Test folder processing methods."""

    @pytest.mark.asyncio
    async def test_process_folder_async(
        self, vision_parser: VisionParser, temp_cache_dir: Path
    ) -> None:
        """Test async folder processing."""
        # Create test files
        test_files = []
        for i in range(3):
            pdf_file = temp_cache_dir / f"test_{i}.pdf"
            pdf_file.write_bytes(b"fake_pdf_content")
            test_files.append(pdf_file)

        # Create non-PDF file (should be ignored)
        txt_file = temp_cache_dir / "test.txt"
        txt_file.write_text("text content")

        with patch.object(
            vision_parser, 'process_pdf_async', return_value={"test": "result"}
        ):
            results = await vision_parser.process_folder_async(str(temp_cache_dir))

            # Should process only PDF files
            assert len(results) == 3

    def test_process_folder_sync(
        self, vision_parser: VisionParser, temp_cache_dir: Path
    ) -> None:
        """Test synchronous folder processing."""
        with patch.object(
            vision_parser, 'process_folder_async', return_value=[{"test": "result"}]
        ):
            results = vision_parser.process_folder(str(temp_cache_dir))
            assert results == [{"test": "result"}]

    @pytest.mark.asyncio
    async def test_process_folder_empty(
        self, vision_parser: VisionParser, temp_cache_dir: Path
    ) -> None:
        """Test processing empty folder."""
        results = await vision_parser.process_folder_async(str(temp_cache_dir))
        assert results == []

    @pytest.mark.asyncio
    async def test_process_folder_nonexistent(
        self, vision_parser: VisionParser
    ) -> None:
        """Test processing non-existent folder."""
        with pytest.raises(FileNotFoundError):
            await vision_parser.process_folder_async("nonexistent_folder")


class TestVisionParserCacheOperations:
    """Test cache-related operations."""

    def test_get_partial_cache_path(self, vision_parser: VisionParser) -> None:
        """Test partial cache path generation."""
        cache_key = "test_cache_key_123"
        path = vision_parser._get_partial_cache_path(cache_key)

        assert path.parent == vision_parser.cache_dir
        assert path.name == f"{cache_key}_partial.json"

    @pytest.mark.asyncio
    async def test_save_and_load_partial_results_integration(
        self, vision_parser: VisionParser
    ) -> None:
        """Test integration of saving and loading partial results."""
        cache_key = "integration_test_key"
        test_pages = [
            {
                "page_number": 1,
                "page_content": "Page 1 content",
                "page_hash": "hash1",
            },
            {
                "page_number": 2,
                "page_content": "Page 2 content",
                "page_hash": "hash2",
            },
        ]

        # Save partial results
        await vision_parser._save_partial_results(cache_key, test_pages)

        # Load partial results
        loaded_results = await vision_parser._load_partial_results(cache_key)

        assert len(loaded_results) == 2
        assert 1 in loaded_results
        assert 2 in loaded_results
        assert loaded_results[1]["page_content"] == "Page 1 content"
        assert loaded_results[2]["page_content"] == "Page 2 content"

    @pytest.mark.asyncio
    async def test_incremental_partial_results(
        self, vision_parser: VisionParser
    ) -> None:
        """Test incremental saving of partial results."""
        cache_key = "incremental_test_key"

        # Save first batch
        batch1 = [{"page_number": 1, "page_content": "Page 1", "page_hash": "hash1"}]
        await vision_parser._save_partial_results(cache_key, batch1)

        # Save second batch
        batch2 = [{"page_number": 2, "page_content": "Page 2", "page_hash": "hash2"}]
        await vision_parser._save_partial_results(cache_key, batch2)

        # Load all results
        loaded_results = await vision_parser._load_partial_results(cache_key)

        assert len(loaded_results) == 2
        assert loaded_results[1]["page_content"] == "Page 1"
        assert loaded_results[2]["page_content"] == "Page 2"


class TestVisionParserErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_pdf_processing_error(
        self, vision_parser: VisionParser, test_pdf_path: str
    ) -> None:
        """Test error handling during PDF processing."""
        with patch(
            'aicapture.vision_parser.fitz.open',
            side_effect=Exception("PDF parsing error"),
        ):
            with pytest.raises(Exception, match="PDF parsing error"):
                await vision_parser.process_pdf_async(test_pdf_path)

    @pytest.mark.asyncio
    async def test_vision_model_error(
        self, temp_cache_dir: Path, temp_image_file: str
    ) -> None:
        """Test error handling when vision model fails."""
        # Create parser with failing vision model
        failing_model = MockVisionModel()
        failing_model.process_image_async = AsyncMock(
            side_effect=Exception("Vision model error")
        )

        parser = VisionParser(vision_model=failing_model, cache_dir=str(temp_cache_dir))

        with pytest.raises(Exception, match="Vision model error"):
            await parser.process_image_async(temp_image_file)

    @pytest.mark.asyncio
    async def test_corrupted_cache_file_handling(
        self, vision_parser: VisionParser
    ) -> None:
        """Test handling of corrupted cache files."""
        cache_key = "corrupted_test_key"
        partial_cache_path = vision_parser._get_partial_cache_path(cache_key)

        # Create corrupted JSON file
        partial_cache_path.write_text("{ corrupted json content")

        # Should return empty dict for corrupted file
        loaded_results = await vision_parser._load_partial_results(cache_key)
        assert loaded_results == {}

    def test_invalid_dpi_setting(
        self, temp_cache_dir: Path, mock_vision_model: MockVisionModel
    ) -> None:
        """Test handling of invalid DPI settings."""
        # Very low DPI should be handled gracefully
        parser = VisionParser(
            vision_model=mock_vision_model, cache_dir=str(temp_cache_dir), dpi=1
        )
        assert parser.dpi == 1

        # Very high DPI should be handled gracefully
        parser = VisionParser(
            vision_model=mock_vision_model, cache_dir=str(temp_cache_dir), dpi=9999
        )
        assert parser.dpi == 9999


class TestVisionParserConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_pdf_processing(
        self, vision_parser: VisionParser, temp_cache_dir: Path
    ) -> None:
        """Test concurrent PDF processing."""
        # Create multiple test PDF files
        pdf_files = []
        for i in range(3):
            pdf_file = temp_cache_dir / f"concurrent_test_{i}.pdf"
            pdf_file.write_bytes(b"fake_pdf_content")
            pdf_files.append(str(pdf_file))

        # Mock PDF processing
        with patch.object(
            vision_parser, 'process_pdf_async', return_value={"test": "result"}
        ):
            # Process files concurrently
            tasks = [
                vision_parser.process_pdf_async(pdf_file) for pdf_file in pdf_files
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(result == {"test": "result"} for result in results)

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(
        self, vision_parser: VisionParser
    ) -> None:
        """Test concurrent cache operations."""
        cache_key = "concurrent_cache_test"

        # Define multiple save operations
        async def save_page(page_num: int) -> None:
            page_data = [
                {
                    "page_number": page_num,
                    "page_content": f"Page {page_num} content",
                    "page_hash": f"hash{page_num}",
                }
            ]
            await vision_parser._save_partial_results(cache_key, page_data)

        # Run concurrent save operations
        tasks = [save_page(i) for i in range(1, 6)]
        await asyncio.gather(*tasks)

        # Load results
        loaded_results = await vision_parser._load_partial_results(cache_key)

        # Should have all 5 pages
        assert len(loaded_results) == 5
        for i in range(1, 6):
            assert i in loaded_results
            assert loaded_results[i]["page_content"] == f"Page {i} content"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

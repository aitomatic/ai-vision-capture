"""Tests for NativePDFParser — native PDF text extraction with underline detection."""

import pytest

from aicapture.native_parser import NativePDFParser


class TestNativePDFParserBasic:
    """Basic functionality tests."""

    def test_init_with_valid_pdf(self):
        parser = NativePDFParser("tmp/underline_text/No.16_不動産登記簿謄本.pdf")
        assert parser is not None

    def test_init_with_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            NativePDFParser("nonexistent.pdf")

    def test_init_with_non_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="must be a PDF"):
            NativePDFParser(str(txt_file))

    def test_extract_returns_expected_structure(self):
        parser = NativePDFParser("tmp/underline_text/No.16_不動産登記簿謄本.pdf")
        result = parser.extract()
        assert "file_name" in result
        assert "total_pages" in result
        assert "pages" in result
        assert isinstance(result["pages"], list)
        assert len(result["pages"]) == result["total_pages"]
        for page in result["pages"]:
            assert "page_number" in page
            assert "content" in page

    def test_extract_specific_pages(self):
        parser = NativePDFParser("tmp/underline_text/No.16_不動産登記簿謄本.pdf")
        result = parser.extract(pages=[2])
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 2

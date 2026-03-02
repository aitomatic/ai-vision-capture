"""Tests for NativePDFParser — native PDF text extraction with underline detection."""

from pathlib import Path

import pytest

from aicapture.native_parser import NativePDFParser

SAMPLE_PDF = "tmp/underline_text/No.16_不動産登記簿謄本.pdf"
SAMPLE_PDF_23 = "tmp/underline_text/No.23_不動産登記簿謄本.pdf"
has_sample = Path(SAMPLE_PDF).exists()
has_sample_23 = Path(SAMPLE_PDF_23).exists()
requires_sample = pytest.mark.skipif(not has_sample, reason="Sample PDF not available")
requires_sample_23 = pytest.mark.skipif(not has_sample_23, reason="Sample PDF 23 not available")


class TestNativePDFParserBasic:
    """Basic functionality tests."""

    @requires_sample
    def test_init_with_valid_pdf(self):
        parser = NativePDFParser(SAMPLE_PDF)
        assert parser is not None

    def test_init_with_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            NativePDFParser("nonexistent.pdf")

    def test_init_with_non_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="must be a PDF"):
            NativePDFParser(str(txt_file))

    @requires_sample
    def test_extract_returns_expected_structure(self):
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract()
        assert "file_name" in result
        assert "total_pages" in result
        assert "pages" in result
        assert isinstance(result["pages"], list)
        assert len(result["pages"]) == result["total_pages"]
        for page in result["pages"]:
            assert "page_number" in page
            assert "content" in page

    @requires_sample
    def test_extract_specific_pages(self):
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[2])
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 2


class TestUnderlineDetection:
    """Verify underline detection against real customer PDFs."""

    @requires_sample
    def test_no16_page2_has_underlined_text(self):
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        # These lines should be detected as underlined (verified via spike analysis)
        assert "<underline>" in content
        assert "</underline>" in content
        # Known underlined text from row 1
        assert "手形取引契約同日設定</underline>" in content

    @requires_sample
    def test_no16_page2_non_underlined_text_has_no_tags(self):
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        # Table header "権利部" should NOT be underlined
        lines = content.split("\n")
        header_lines = [line for line in lines if "権利部" in line]
        for hl in header_lines:
            assert "<underline>" not in hl

    @requires_sample_23
    def test_no23_page2_has_underlined_text(self):
        parser = NativePDFParser(SAMPLE_PDF_23)
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        assert "<underline>" in content
        # Known underlined text from No.23
        assert "金銭消費貸借同日設定</underline>" in content

    @requires_sample
    def test_no16_page1_extracts_without_error(self):
        """Page 1 is different from page 2 -- just verify it extracts."""
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[1])
        content = result["pages"][0]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

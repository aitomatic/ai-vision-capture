"""Tests for NativePDFParser — native PDF text extraction with underline detection."""

from pathlib import Path

import fitz  # type: ignore
import pytest

from aicapture.native_parser import (
    NativePDFParser,
    build_line_with_underline_tags,
    extract_page_with_formatting,
    get_underline_rects,
    is_span_underlined,
)

SAMPLE_PDF = "tmp/underline_text/No.16_不動産登記簿謄本.pdf"
SAMPLE_PDF_23 = "tmp/underline_text/No.23_不動産登記簿謄本.pdf"
has_sample = Path(SAMPLE_PDF).exists()
has_sample_23 = Path(SAMPLE_PDF_23).exists()
requires_sample = pytest.mark.skipif(not has_sample, reason="Sample PDF not available")
requires_sample_23 = pytest.mark.skipif(not has_sample_23, reason="Sample PDF 23 not available")


# ---------------------------------------------------------------------------
# Helpers — create synthetic PDFs with known underline geometry
# ---------------------------------------------------------------------------


def _create_pdf_with_underline(tmp_path: Path) -> str:
    """Create a minimal PDF with a text span and a thin rect underline beneath it."""
    pdf_path = str(tmp_path / "underline_test.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Insert text at a known position
    text_point = fitz.Point(100, 100)
    page.insert_text(text_point, "Underlined Text", fontsize=12)

    # Draw a thin filled rect underneath the text (simulating a PDF underline).
    # The text baseline is at y≈100, bounding box bottom is ~y≈102 for 12pt text.
    # Place underline rect at y=101 (inside lower part of text bbox).
    underline_rect = fitz.Rect(100, 101, 220, 101.5)
    page.draw_rect(underline_rect, color=None, fill=(0, 0, 0))

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _create_pdf_without_underline(tmp_path: Path) -> str:
    """Create a minimal PDF with text but no underline rects."""
    pdf_path = str(tmp_path / "no_underline_test.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(100, 100), "Plain Text", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _create_pdf_with_strikethrough(tmp_path: Path) -> str:
    """Create a PDF with a rect at the vertical midpoint (strikethrough), not underline."""
    pdf_path = str(tmp_path / "strikethrough_test.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # 12pt text at y=100 — bbox is roughly y0≈88, y1≈102 (14px tall)
    page.insert_text(fitz.Point(100, 100), "Strikethrough Text", fontsize=12)

    # Draw rect at vertical midpoint (~y=95), which is the upper half of the span
    strike_rect = fitz.Rect(100, 94, 230, 94.5)
    page.draw_rect(strike_rect, color=None, fill=(0, 0, 0))

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _create_multipage_pdf(tmp_path: Path, num_pages: int = 3) -> str:
    """Create a multi-page PDF for bounds-checking tests."""
    pdf_path = str(tmp_path / "multipage_test.pdf")
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text(fitz.Point(100, 100), f"Page {i + 1}", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Unit tests — synthetic PDFs (always run in CI)
# ---------------------------------------------------------------------------


class TestGetUnderlineRects:
    """Test underline rect detection from PDF drawings."""

    def test_finds_thin_filled_rect(self, tmp_path):
        pdf_path = _create_pdf_with_underline(tmp_path)
        doc = fitz.open(pdf_path)
        rects = get_underline_rects(doc[0])
        assert len(rects) >= 1
        doc.close()

    def test_no_rects_for_plain_text(self, tmp_path):
        pdf_path = _create_pdf_without_underline(tmp_path)
        doc = fitz.open(pdf_path)
        rects = get_underline_rects(doc[0])
        assert len(rects) == 0
        doc.close()


class TestIsSpanUnderlined:
    """Test span-to-underline matching logic."""

    def test_matches_rect_near_baseline(self):
        # Span bbox: top=88, bottom=102 (14px tall, midpoint=95)
        span = fitz.Rect(100, 88, 200, 102)
        # Underline at y=101, in lower half, within 4px of bottom
        underline = fitz.Rect(100, 101, 200, 101.5)
        assert is_span_underlined(span, [underline]) is True

    def test_rejects_rect_in_upper_half(self):
        # Span bbox: top=88, bottom=102, midpoint=95
        span = fitz.Rect(100, 88, 200, 102)
        # Rect at y=94, in the UPPER half — should NOT match (strikethrough)
        strike = fitz.Rect(100, 94, 200, 94.5)
        assert is_span_underlined(span, [strike]) is False

    def test_rejects_rect_too_far_below(self):
        span = fitz.Rect(100, 88, 200, 102)
        # Rect 10px below span bottom — too far
        far_below = fitz.Rect(100, 112, 200, 112.5)
        assert is_span_underlined(span, [far_below]) is False

    def test_rejects_rect_with_no_x_overlap(self):
        span = fitz.Rect(100, 88, 200, 102)
        # Underline at correct Y but no horizontal overlap
        no_overlap = fitz.Rect(300, 101, 400, 101.5)
        assert is_span_underlined(span, [no_overlap]) is False


class TestBuildLineWithUnderlineTags:
    """Test tag wrapping logic."""

    def test_wraps_underlined_span(self):
        spans = [{"text": "hello", "bbox": (100, 88, 200, 102)}]
        underlines = [fitz.Rect(100, 101, 200, 101.5)]
        result = build_line_with_underline_tags(spans, underlines)
        assert result == "<underline>hello</underline>"

    def test_no_tags_for_plain_span(self):
        spans = [{"text": "hello", "bbox": (100, 88, 200, 102)}]
        result = build_line_with_underline_tags(spans, [])
        assert result == "hello"

    def test_skips_empty_spans(self):
        spans = [
            {"text": "", "bbox": (100, 88, 110, 102)},
            {"text": "world", "bbox": (110, 88, 200, 102)},
        ]
        underlines = [fitz.Rect(110, 101, 200, 101.5)]
        result = build_line_with_underline_tags(spans, underlines)
        assert result == "<underline>world</underline>"

    def test_mixed_underlined_and_plain_spans(self):
        spans = [
            {"text": "plain ", "bbox": (50, 88, 100, 102)},
            {"text": "underlined", "bbox": (100, 88, 200, 102)},
            {"text": " more plain", "bbox": (200, 88, 300, 102)},
        ]
        underlines = [fitz.Rect(100, 101, 200, 101.5)]
        result = build_line_with_underline_tags(spans, underlines)
        assert result == "plain <underline>underlined</underline> more plain"


class TestExtractPageWithFormatting:
    """Test full page extraction returning (annotated, plain) tuple."""

    def test_returns_tuple(self, tmp_path):
        pdf_path = _create_pdf_with_underline(tmp_path)
        doc = fitz.open(pdf_path)
        result = extract_page_with_formatting(doc[0])
        assert isinstance(result, tuple)
        assert len(result) == 2
        annotated, plain = result
        assert isinstance(annotated, str)
        assert isinstance(plain, str)
        doc.close()

    def test_annotated_has_underline_tags(self, tmp_path):
        pdf_path = _create_pdf_with_underline(tmp_path)
        doc = fitz.open(pdf_path)
        annotated, plain = extract_page_with_formatting(doc[0])
        assert "<underline>" in annotated
        assert "</underline>" in annotated
        doc.close()

    def test_plain_has_no_tags(self, tmp_path):
        pdf_path = _create_pdf_with_underline(tmp_path)
        doc = fitz.open(pdf_path)
        annotated, plain = extract_page_with_formatting(doc[0])
        assert "<underline>" not in plain
        assert "</underline>" not in plain
        assert "Underlined Text" in plain
        doc.close()

    def test_strikethrough_not_detected_as_underline(self, tmp_path):
        pdf_path = _create_pdf_with_strikethrough(tmp_path)
        doc = fitz.open(pdf_path)
        annotated, plain = extract_page_with_formatting(doc[0])
        assert "<underline>" not in annotated
        assert "Strikethrough Text" in plain
        doc.close()


class TestNativePDFParserBasic:
    """Basic functionality tests."""

    def test_init_with_valid_pdf(self, tmp_path):
        pdf_path = _create_pdf_without_underline(tmp_path)
        parser = NativePDFParser(pdf_path)
        assert parser is not None
        parser.close()

    def test_init_with_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            NativePDFParser("nonexistent.pdf")

    def test_init_with_non_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="must be a PDF"):
            NativePDFParser(str(txt_file))

    def test_context_manager(self, tmp_path):
        pdf_path = _create_pdf_without_underline(tmp_path)
        with NativePDFParser(pdf_path) as parser:
            result = parser.extract()
            assert "pages" in result

    def test_extract_returns_expected_structure(self, tmp_path):
        pdf_path = _create_multipage_pdf(tmp_path, num_pages=3)
        parser = NativePDFParser(pdf_path)
        result = parser.extract()
        assert result["file_name"] == "multipage_test.pdf"
        assert result["total_pages"] == 3
        assert len(result["pages"]) == 3
        for i, page in enumerate(result["pages"]):
            assert page["page_number"] == i + 1
            assert "content" in page
        parser.close()

    def test_extract_specific_pages(self, tmp_path):
        pdf_path = _create_multipage_pdf(tmp_path, num_pages=3)
        parser = NativePDFParser(pdf_path)
        result = parser.extract(pages=[2])
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 2
        parser.close()

    def test_extract_page_out_of_range_raises(self, tmp_path):
        pdf_path = _create_multipage_pdf(tmp_path, num_pages=3)
        parser = NativePDFParser(pdf_path)
        with pytest.raises(ValueError, match="out of range"):
            parser.extract(pages=[999])
        parser.close()

    def test_extract_page_zero_raises(self, tmp_path):
        pdf_path = _create_multipage_pdf(tmp_path, num_pages=3)
        parser = NativePDFParser(pdf_path)
        with pytest.raises(ValueError, match="out of range"):
            parser.extract(pages=[0])
        parser.close()

    def test_extract_negative_page_raises(self, tmp_path):
        pdf_path = _create_multipage_pdf(tmp_path, num_pages=3)
        parser = NativePDFParser(pdf_path)
        with pytest.raises(ValueError, match="out of range"):
            parser.extract(pages=[-1])
        parser.close()


# ---------------------------------------------------------------------------
# Integration tests — real customer PDFs (skipped in CI)
# ---------------------------------------------------------------------------


class TestUnderlineDetection:
    """Verify underline detection against real customer PDFs."""

    @requires_sample
    def test_no16_page2_has_underlined_text(self):
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        assert "<underline>" in content
        assert "</underline>" in content
        assert "手形取引契約同日設定</underline>" in content
        parser.close()

    @requires_sample
    def test_no16_page2_non_underlined_text_has_no_tags(self):
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        lines = content.split("\n")
        header_lines = [line for line in lines if "権利部" in line]
        for hl in header_lines:
            assert "<underline>" not in hl
        parser.close()

    @requires_sample_23
    def test_no23_page2_has_underlined_text(self):
        parser = NativePDFParser(SAMPLE_PDF_23)
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        assert "<underline>" in content
        assert "金銭消費貸借同日設定</underline>" in content
        parser.close()

    @requires_sample
    def test_no16_page1_extracts_without_error(self):
        """Page 1 is different from page 2 -- just verify it extracts."""
        parser = NativePDFParser(SAMPLE_PDF)
        result = parser.extract(pages=[1])
        content = result["pages"][0]["content"]
        assert isinstance(content, str)
        assert len(content) > 0
        parser.close()

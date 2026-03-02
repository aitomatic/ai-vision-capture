"""Native PDF text extraction with formatting detection.

Extracts text from text-based PDFs using PyMuPDF, detecting visual formatting
like underlines (drawn as thin rectangles) and preserving them as XML tags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # type: ignore


def get_underline_rects(page: Any) -> list[Any]:
    """Find thin filled rectangles on a page that represent underlines.

    Underlines in many PDFs are drawn as thin filled rectangles rather than
    being a text attribute. This function detects them by filtering for
    rectangles that are thin (< 2px height) and not full-width table borders.
    """
    page_width = page.rect.width
    underlines = []
    for d in page.get_drawings():
        items = d["items"]
        if len(items) != 1 or items[0][0] != "re":
            continue
        rect = fitz.Rect(items[0][1])
        if rect.height < 2 and rect.width < page_width * 0.8 and d.get("fill") is not None:
            underlines.append(rect)
    return underlines


def is_span_underlined(span_bbox: Any, underline_rects: list[Any]) -> bool:
    """Check if a text span has an underline rect beneath it."""
    for ul in underline_rects:
        y_close = abs(span_bbox.y1 - ul.y0) < 5
        x_overlap = span_bbox.x0 < ul.x1 and span_bbox.x1 > ul.x0
        if y_close and x_overlap:
            return True
    return False


def build_line_with_underline_tags(spans: list[dict], underline_rects: list[Any]) -> str:
    """Build a text line, wrapping underlined spans in <underline> tags."""
    parts: list[str] = []
    in_underline = False

    for span in spans:
        text = span["text"]
        if not text:
            continue
        bbox = fitz.Rect(span["bbox"])
        is_ul = is_span_underlined(bbox, underline_rects)

        if is_ul and not in_underline:
            parts.append("<underline>")
            in_underline = True
        elif not is_ul and in_underline:
            parts.append("</underline>")
            in_underline = False

        parts.append(text)

    if in_underline:
        parts.append("</underline>")

    return "".join(parts)


def extract_page_with_formatting(page: Any) -> str:
    """Extract text from a fitz page with underline detection.

    Returns text with underlined portions wrapped in <underline> tags.
    """
    underline_rects = get_underline_rects(page)
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    lines_out: list[str] = []
    for block in blocks["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            line_text = build_line_with_underline_tags(line["spans"], underline_rects)
            if line_text.strip():
                lines_out.append(line_text)

    return "\n".join(lines_out)


class NativePDFParser:
    """Extract text from PDFs with formatting annotations.

    Detects underlined text by matching thin drawn rectangles to text spans
    using spatial proximity. Outputs text with <underline> XML tags.
    """

    def __init__(self, pdf_path: str) -> None:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")
        self._pdf_path = str(path)
        self._doc = fitz.open(self._pdf_path)

    def extract(self, pages: list[int] | None = None) -> dict:
        """Extract text with formatting annotations.

        Args:
            pages: 1-based page numbers to extract. None = all pages.

        Returns:
            Dict with file_name, total_pages, and pages list.
            Each page has page_number (1-based) and content (text with XML tags).
        """
        total_pages = len(self._doc)
        if pages is None:
            page_nums = list(range(1, total_pages + 1))
        else:
            page_nums = pages

        extracted = []
        for pn in page_nums:
            page = self._doc[pn - 1]  # 0-indexed
            content = self._extract_page(page)
            extracted.append({"page_number": pn, "content": content})

        return {
            "file_name": Path(self._pdf_path).name,
            "total_pages": total_pages,
            "pages": extracted,
        }

    def _extract_page(self, page: fitz.Page) -> str:
        """Extract text from a single page with underline detection."""
        return extract_page_with_formatting(page)

    def close(self) -> None:
        """Close the PDF document."""
        self._doc.close()

    def __enter__(self) -> NativePDFParser:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

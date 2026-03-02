"""Native PDF text extraction with formatting detection.

Extracts text from text-based PDFs using PyMuPDF, detecting visual formatting
like underlines (drawn as thin rectangles) and preserving them as XML tags.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore

# Rectangles wider than this fraction of the page width are treated as table
# borders or horizontal rules rather than text underlines.  The 0.8 threshold
# was chosen empirically: Japanese property registry PDFs (不動産登記簿) use
# underlines that span individual cell content (~30-40% of page width), while
# full-width table borders are ≥95%.  The 80% cutoff sits comfortably between
# the two populations.
_MAX_UNDERLINE_WIDTH_RATIO = 0.8


def get_underline_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Find thin filled rectangles on a page that represent underlines.

    Underlines in many PDFs are drawn as thin filled rectangles rather than
    being a text attribute. This function detects them by filtering for
    rectangles that are thin (< 2px height) and narrower than
    ``_MAX_UNDERLINE_WIDTH_RATIO`` of the page width.
    """
    page_width = page.rect.width
    max_width = page_width * _MAX_UNDERLINE_WIDTH_RATIO
    underlines: list[fitz.Rect] = []
    for d in page.get_drawings():
        items = d["items"]
        if len(items) != 1 or items[0][0] != "re":
            continue
        rect = fitz.Rect(items[0][1])
        if rect.height < 2 and rect.width < max_width and d.get("fill") is not None:
            underlines.append(rect)
    return underlines


def is_span_underlined(span_bbox: fitz.Rect, underline_rects: list[fitz.Rect]) -> bool:
    """Check if a text span has an underline rect near its baseline.

    The underline must be in the lower half of the span (to exclude
    strikethrough lines) and within 4px of the span bottom (baseline area).
    """
    span_midpoint_y = (span_bbox.y0 + span_bbox.y1) / 2
    for ul in underline_rects:
        # Underline must be in the lower half of the span to exclude strikethrough
        if ul.y0 <= span_midpoint_y:
            continue
        # Underline must be within 4px of span bottom (baseline area)
        if abs(span_bbox.y1 - ul.y0) >= 4:
            continue
        # Must overlap horizontally
        if span_bbox.x0 < ul.x1 and span_bbox.x1 > ul.x0:
            return True
    return False


def build_line_with_underline_tags(spans: list[dict], underline_rects: list[fitz.Rect]) -> str:
    """Build a text line, wrapping underlined spans in <underline> tags.

    Each contiguous run of underlined spans gets its own tag pair. If an
    underlined span is separated from the previous underlined span by a
    non-underlined span, they get separate ``<underline>`` wrappers.
    """
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


def extract_page_with_formatting(page: fitz.Page) -> tuple[str, str]:
    """Extract text from a fitz page with underline detection.

    Returns a tuple of ``(annotated_text, plain_text)`` where:
    - ``annotated_text`` has underlined portions wrapped in ``<underline>`` tags
    - ``plain_text`` is the raw text without any formatting tags

    Extracting both from the same ``get_text("dict")`` call avoids parsing
    the page structure twice.
    """
    underline_rects = get_underline_rects(page)
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    annotated_lines: list[str] = []
    plain_lines: list[str] = []
    for block in blocks["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            plain = "".join(span["text"] for span in line["spans"])
            if not plain.strip():
                continue
            plain_lines.append(plain)
            annotated = build_line_with_underline_tags(line["spans"], underline_rects)
            annotated_lines.append(annotated)

    return "\n".join(annotated_lines), "\n".join(plain_lines)


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

        Raises:
            ValueError: If any page number is out of range.
        """
        total_pages = len(self._doc)
        if pages is None:
            page_nums = list(range(1, total_pages + 1))
        else:
            for pn in pages:
                if pn < 1 or pn > total_pages:
                    raise ValueError(f"Page {pn} out of range (1-{total_pages})")
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
        annotated, _plain = extract_page_with_formatting(page)
        return annotated

    def close(self) -> None:
        """Close the PDF document."""
        self._doc.close()

    def __enter__(self) -> NativePDFParser:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

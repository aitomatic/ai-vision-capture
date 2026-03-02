# Native PDF Underline Extraction — Design

**Date**: 2026-03-01
**Status**: Approved
**Driver**: Japanese customer request — extract text from text-based PDFs preserving underlined formatting

## Problem

Customer needs to extract text from Japanese real estate registration documents (不動産登記簿謄本). These are text-based PDFs with tables where certain text in the last column is underlined. The underlined text must be preserved in the output using XML tags.

## Key Technical Finding

Underlines in these PDFs are **drawn as thin filled rectangles** (0.5px height), not text attributes. PyMuPDF's `get_text("dict")` text flags do NOT include underline info. Detection requires `get_drawings()` to find thin rects and spatial matching to associate them with text spans.

## Design

### New File: `aicapture/native_parser.py`

Single class: `NativePDFParser`

```python
class NativePDFParser:
    def __init__(self, pdf_path: str):
        """Open a PDF for native text extraction."""

    def extract(self, pages: list[int] | None = None) -> dict:
        """Extract text from PDF with formatting annotations.

        Args:
            pages: 1-based page numbers to extract. None = all pages.

        Returns:
            {
                "file_name": str,
                "total_pages": int,
                "pages": [
                    {
                        "page_number": int,     # 1-based
                        "content": str,          # text with <underline> tags
                    }
                ]
            }
        """
```

### Algorithm

Per page:

1. **Get text spans**: `page.get_text("dict")` → blocks → lines → spans (each with bbox, text, font, flags)
2. **Get underline rects**: `page.get_drawings()` → filter for thin filled rectangles (height < 2px, not full-width table borders)
3. **Match underlines to spans**: For each text span, check if any underline rect is positioned just below it (y-proximity < 5px, x-overlap)
4. **Reconstruct text**: Build output line by line. Wrap underlined spans in `<underline>` tags. Preserve line breaks.

### Underline Detection Heuristics

A drawing is classified as an underline if:
- It's a rectangle (`"re"` item type)
- Height < 2px (thin horizontal bar)
- Width < 80% of page width (not a table border)
- Filled with a color (typically black)

A text span is underlined if:
- An underline rect exists where `abs(span.bbox.y1 - rect.y0) < 5` (rect is just below text bottom)
- And horizontal overlap: `span.bbox.x0 < rect.x1` and `span.bbox.x1 > rect.x0`

### Export

Add `NativePDFParser` to `aicapture/__init__.py` `__all__`.

### What's NOT in v1

- No caching
- No async
- No VLM integration
- No table structure detection (just text with underline tags)
- No other formatting types (bold, italic, etc.)

These can be added incrementally based on customer feedback.

## Implementation Plan

1. Create `aicapture/native_parser.py` with `NativePDFParser`
2. Add export to `__init__.py`
3. Write tests using the two example PDFs
4. Verify output matches expected underlined text

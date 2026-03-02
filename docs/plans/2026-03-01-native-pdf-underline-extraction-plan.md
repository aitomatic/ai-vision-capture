# NativePDFParser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract text from text-based PDFs preserving underlined formatting as `<underline>` XML tags.

**Architecture:** New `NativePDFParser` class in `aicapture/native_parser.py`. Uses PyMuPDF (fitz) to get text spans via `get_text("dict")` and detect underline drawings via `get_drawings()`. Matches underlines to text by spatial proximity, then reconstructs text with XML tags.

**Tech Stack:** PyMuPDF (fitz) — already a project dependency. No new deps.

**Design doc:** `docs/plans/2026-03-01-native-pdf-underline-extraction-design.md`

---

### Task 1: Core underline detection — test and implement

**Files:**
- Create: `tests/test_native_parser.py`
- Create: `aicapture/native_parser.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_native_parser.py -v`
Expected: FAIL — `ImportError: cannot import name 'NativePDFParser'`

**Step 3: Write minimal implementation**

Create `aicapture/native_parser.py`:

```python
"""Native PDF text extraction with formatting detection.

Extracts text from text-based PDFs using PyMuPDF, detecting visual formatting
like underlines (drawn as thin rectangles) and preserving them as XML tags.
"""

from __future__ import annotations

from pathlib import Path

import fitz


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
        underline_rects = self._get_underline_rects(page)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        lines_out: list[str] = []
        for block in blocks["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = self._build_line_with_tags(line["spans"], underline_rects)
                if line_text.strip():
                    lines_out.append(line_text)

        return "\n".join(lines_out)

    def _get_underline_rects(self, page: fitz.Page) -> list[fitz.Rect]:
        """Find thin filled rectangles that represent underlines."""
        page_width = page.rect.width
        underlines = []
        for d in page.get_drawings():
            items = d["items"]
            if len(items) != 1 or items[0][0] != "re":
                continue
            rect = fitz.Rect(items[0][1])
            # Thin filled rect, not a full-width table border
            if rect.height < 2 and rect.width < page_width * 0.8 and d.get("fill") is not None:
                underlines.append(rect)
        return underlines

    def _is_underlined(self, span_bbox: fitz.Rect, underline_rects: list[fitz.Rect]) -> bool:
        """Check if a text span has an underline rect beneath it."""
        for ul in underline_rects:
            y_close = abs(span_bbox.y1 - ul.y0) < 5
            x_overlap = span_bbox.x0 < ul.x1 and span_bbox.x1 > ul.x0
            if y_close and x_overlap:
                return True
        return False

    def _build_line_with_tags(self, spans: list[dict], underline_rects: list[fitz.Rect]) -> str:
        """Build a text line, wrapping underlined spans in <underline> tags."""
        parts: list[str] = []
        in_underline = False

        for span in spans:
            text = span["text"]
            if not text:
                continue
            bbox = fitz.Rect(span["bbox"])
            is_ul = self._is_underlined(bbox, underline_rects)

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

    def close(self) -> None:
        """Close the PDF document."""
        self._doc.close()

    def __enter__(self) -> NativePDFParser:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_native_parser.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add aicapture/native_parser.py tests/test_native_parser.py
git commit -m "feat: add NativePDFParser with underline detection"
```

---

### Task 2: Underline content verification — test against real PDFs

**Files:**
- Modify: `tests/test_native_parser.py`

**Step 1: Write the failing test**

Add to `tests/test_native_parser.py`:

```python
class TestUnderlineDetection:
    """Verify underline detection against real customer PDFs."""

    def test_no16_page2_has_underlined_text(self):
        parser = NativePDFParser("tmp/underline_text/No.16_不動産登記簿謄本.pdf")
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        # These lines should be detected as underlined (from our spike analysis)
        assert "<underline>" in content
        assert "</underline>" in content

        # Specific underlined entries from row 1 (根抵当権設定)
        assert "<underline>" in content  # at minimum, tags exist
        # Check known underlined text fragments
        assert "手形取引契約同日設定</underline>" in content
        assert "<underline>" + "元本極度額" in content or "元本極度額" in content

    def test_no16_page2_non_underlined_text_has_no_tags(self):
        parser = NativePDFParser("tmp/underline_text/No.16_不動産登記簿謄本.pdf")
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        # Row headers and non-underlined entries should NOT have tags
        # "権利部（乙区）" is a header, not underlined
        lines = content.split("\n")
        header_lines = [l for l in lines if "権利部" in l]
        for hl in header_lines:
            assert "<underline>" not in hl

    def test_no23_page2_has_underlined_text(self):
        parser = NativePDFParser("tmp/underline_text/No.23_不動産登記簿謄本.pdf")
        result = parser.extract(pages=[2])
        content = result["pages"][0]["content"]

        assert "<underline>" in content
        # Known underlined text from No.23
        assert "金銭消費貸借同日設定</underline>" in content

    def test_no16_page1_may_have_different_underlines(self):
        """Page 1 is different from page 2 — just verify it extracts without error."""
        parser = NativePDFParser("tmp/underline_text/No.16_不動産登記簿謄本.pdf")
        result = parser.extract(pages=[1])
        content = result["pages"][0]["content"]
        assert isinstance(content, str)
        assert len(content) > 0
```

**Step 2: Run tests to verify behavior**

Run: `uv run pytest tests/test_native_parser.py::TestUnderlineDetection -v`
Expected: PASS — if the spatial matching works correctly from Task 1. If tests fail, adjust the heuristics (y-proximity threshold, x-overlap logic) in `_is_underlined` and `_get_underline_rects`.

**Step 3: Iterate on heuristics if needed**

If underline detection is off, tune:
- `rect.height < 2` threshold
- `abs(span_bbox.y1 - ul.y0) < 5` proximity
- `rect.width < page_width * 0.8` border filter

**Step 4: Commit**

```bash
git add tests/test_native_parser.py
git commit -m "test: add underline detection verification against customer PDFs"
```

---

### Task 3: Export and lint

**Files:**
- Modify: `aicapture/__init__.py`

**Step 1: Add export**

Add to `aicapture/__init__.py`:
- Import: `from aicapture.native_parser import NativePDFParser`
- Add `"NativePDFParser"` to `__all__` under a `# Native parser` comment

**Step 2: Verify import works**

Run: `uv run python -c "from aicapture import NativePDFParser; print('OK')"`
Expected: `OK`

**Step 3: Run linting**

Run: `make lint`
Expected: PASS (ruff + mypy clean)

**Step 4: Run full test suite**

Run: `make test`
Expected: All tests pass, including existing tests (no regressions)

**Step 5: Commit**

```bash
git add aicapture/__init__.py
git commit -m "feat: export NativePDFParser from aicapture"
```

---

### Task 4: Manual verification — print output for review

**Files:** None (verification only)

**Step 1: Run extraction and print output**

```bash
uv run python -c "
from aicapture import NativePDFParser
parser = NativePDFParser('tmp/underline_text/No.16_不動産登記簿謄本.pdf')
result = parser.extract(pages=[2])
print(result['pages'][0]['content'])
"
```

Review: underlined text should be wrapped in `<underline>` tags, non-underlined text should be plain.

**Step 2: Same for No.23**

```bash
uv run python -c "
from aicapture import NativePDFParser
parser = NativePDFParser('tmp/underline_text/No.23_不動産登記簿謄本.pdf')
result = parser.extract(pages=[2])
print(result['pages'][0]['content'])
"
```

**Step 3: If output looks correct, done. If not, go back to Task 1/2 and adjust.**

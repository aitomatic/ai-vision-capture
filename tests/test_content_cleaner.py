from aicapture.content_cleaner import ContentCleaner, clean_vision_parser_output


class TestContentCleaner:
    """Test suite for ContentCleaner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = ContentCleaner()

    def test_clean_base64_content(self):
        """Test removal of base64 encoded content."""
        content = """
        Normal text here.

        iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA6ZzFNQAAAABJRU5ErkJggg==

        More normal text.
        """

        cleaned = self.cleaner.clean_content(content)
        assert (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA6ZzFNQAAAABJRU5ErkJggg=="
            not in cleaned
        )
        assert "Normal text here." in cleaned
        assert "More normal text." in cleaned

    def test_clean_repetitive_spaces(self):
        """Test removal of repetitive HTML spaces."""
        content = """
        Normal text here.
        Content with&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;too many spaces.
        Content with&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;five spaces (should be removed).
        Content with&nbsp;&nbsp;&nbsp;&nbsp;four spaces (should be kept).
        Content with&nbsp;&nbsp;two spaces should be kept.
        """

        cleaned = self.cleaner.clean_content(content)
        assert "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" not in cleaned
        assert "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" not in cleaned  # 5 spaces should be removed
        assert "&nbsp;&nbsp;&nbsp;&nbsp;" in cleaned  # 4 spaces should be kept
        assert "&nbsp;&nbsp;" in cleaned  # Two spaces should be kept
        assert "Normal text here." in cleaned

    def test_clean_vision_parser_output(self):
        """Test cleaning of VisionParser output."""
        result = {
            "file_object": {
                "file_name": "test.pdf",
                "pages": [
                    {
                        "page_number": 1,
                        "page_content": "Normal text\n\niVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA6ZzFNQAAAABJRU5ErkJggg==\n\nMore text&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
                    }
                ],
            }
        }

        cleaned = clean_vision_parser_output(result)
        page_content = cleaned["file_object"]["pages"][0]["page_content"]

        assert (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA6ZzFNQAAAABJRU5ErkJggg=="
            not in page_content
        )
        assert "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" not in page_content
        assert "Normal text" in page_content
        assert "More text" in page_content

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty content
        assert self.cleaner.clean_content("") == ""
        assert self.cleaner.clean_content(None) is None

        # Non-string input
        assert self.cleaner.clean_content(123) == 123

        # Empty result
        assert clean_vision_parser_output({}) == {}

    def test_preserves_normal_content(self):
        """Test that normal content is preserved."""
        normal_content = """
        # Document Title

        This is normal text with **bold** and *italic* formatting.

        | Column 1 | Column 2 |
        |----------|----------|
        | Data 1   | Data 2   |

        - List item 1
        - List item 2
        """

        cleaned = self.cleaner.clean_content(normal_content)

        # Should preserve all normal content
        assert "# Document Title" in cleaned
        assert "**bold**" in cleaned
        assert "*italic*" in cleaned
        assert "| Column 1 | Column 2 |" in cleaned
        assert "- List item 1" in cleaned

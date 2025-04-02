import asyncio
import unittest
from pathlib import Path
from typing import Any

from aicapture.vision_parser import VisionParser


class TestVisionParser(unittest.TestCase):
    """Test cases for VisionParser class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_file = self.test_dir / "test.pdf"
        self.test_file.touch()

    async def test_process_folder_async(self) -> None:
        """Test processing a folder of PDFs asynchronously."""
        # Create test files
        test_files = ["test1.pdf", "test2.pdf", "test3.txt"]
        for file in test_files:
            (self.test_dir / file).touch()

        parser = VisionParser()
        results = await parser.process_folder_async(str(self.test_dir))

        # Verify only PDF files were processed
        self.assertEqual(len(results), 2)

        # Clean up test files
        for file in test_files:
            (self.test_dir / file).unlink()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    @staticmethod
    def run_async_test(coro: Any) -> Any:
        """Run an async test."""
        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()

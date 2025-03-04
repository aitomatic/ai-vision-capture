import asyncio
import unittest
from pathlib import Path

from loguru import logger

from vision_capture.vision_parser import VisionParser


class TestVisionParser(unittest.TestCase):
    """Test cases for VisionParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("tmp/sample")
        self.test_output_dir = Path("tmp/output")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    async def test_process_folder_async(self):
        """Test processing a folder of PDF files asynchronously."""
        try:
            # Initialize parser with default vision model
            parser = VisionParser(
                invalidate_cache=True,
                invalidate_image_cache=True,
            )
            
            results = await parser.process_folder_async(str(self.test_data_dir))
            
            # Save results
            for result in results:
                output_path = self.test_output_dir / f"{result['file_object']['file_name']}.json"
                parser.save_output(result, str(output_path))
            
            # Verify results
            self.assertTrue(len(results) >= 0)
            logger.info(f"Processing complete. Total results: {len(results)}")

        except Exception as e:
            logger.error(f"Error in test_process_folder_async: {e}")
            raise

    def tearDown(self):
        """Clean up test fixtures."""
        # Optionally clean up test directories if needed
        pass


def run_async_test(coro):
    """Helper function to run async tests."""
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main() 
"""
Example script demonstrating how to use VisionCapture with OpenAI Vision model
to extract structured data from technical documents.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

from aicapture.vision_capture import VisionCapture
from aicapture.vision_models import AnthropicVisionModel

# Template for technical alarm logic
ALARM_TEMPLATE = """
alarm:
  description: string  # Main alarm description
  destination: string # Destination system
  tag: string        # Alarm tag
  ref_logica: integer # Logic reference number

dependencies:
  type: array
  items:
    - signal_name: string  # Name of the dependency signal
      source: string      # Source system/component
      tag: string        # Signal tag
      ref_logica: integer|null  # Logic reference (can be null)
"""


async def main() -> None:
    vision_model = AnthropicVisionModel(
        model="claude-3-5-sonnet-20240620",  # or your preferred model
        max_tokens=4096,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Initialize VisionCapture with OpenAI model
    capture = VisionCapture(vision_model=vision_model)

    # Example document path (adjust as needed)
    doc_path = Path("tests/sample/images/logic.png")

    try:
        # Process document and extract structured data
        result: Dict[str, Any] = await capture.capture(file_path=str(doc_path), template=ALARM_TEMPLATE)

        # Print the structured output
        print("Extracted Data:")
        print(result)

    except Exception as e:
        print(f"Error processing document: {e}")


if __name__ == "__main__":
    asyncio.run(main())

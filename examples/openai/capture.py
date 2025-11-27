"""
Example script demonstrating how to use VisionCapture with OpenAI Vision model
to extract structured data from technical documents.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

from aicapture.vision_capture import VisionCapture
from aicapture.vision_models import OpenAIVisionModel

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
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Create OpenAI Vision model
    vision_model = OpenAIVisionModel(
        model="gpt-4.1",  # or your preferred model
        max_tokens=4096,
        api_key=api_key,
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

"""
Example of parsing an image using auto-detected vision model.

This example demonstrates how to use the AutoDetectVisionModel feature,
which automatically detects and uses the first available vision model
based on your environment variables (GEMINI_API_KEY, OPENAI_API_KEY,
AZURE_OPENAI_API_KEY, or ANTHROPIC_API_KEY).

No need to specify USE_VISION - just set any of the supported API keys!
"""

from aicapture import VisionParser, create_default_vision_model


def main():
    # Auto-detect vision model based on available API keys
    # Checks in order: Gemini -> OpenAI -> Azure -> Anthropic
    vision_model = create_default_vision_model()

    image_path = "tests/sample/images/logic.png"

    # Initialize parser with auto-detected model
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/auto",
        invalidate_cache=True,  # Set to True to force reprocessing
    )

    result = parser.process_image(image_path)

    # Print results
    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")
    print(f"Model used: {vision_model.model}")

    # Print the extracted content
    page_content = result["file_object"]["pages"][0]["page_content"]
    print("\nExtracted Content:")
    print(page_content)


if __name__ == "__main__":
    main()

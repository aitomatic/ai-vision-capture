import os

from aicapture import AnthropicVisionModel, VisionParser


def main():
    vision_model = AnthropicVisionModel(
        model="claude-3-7-sonnet-20250219",
        temperature=0.0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    image_path = "tests/sample/images/logic.png"

    # Initialize parser with a specific prompt for logical diagram analysis
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/anthropic",
        invalidate_cache=True,  # Set to True to force reprocessing
    )

    # Process the image
    result = parser.process_image(image_path)

    # Print results
    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")

    # Print the extracted content
    page_content = result["file_object"]["pages"][0]["page_content"]
    print("\nExtracted Content:")
    print(page_content)


if __name__ == "__main__":
    main()

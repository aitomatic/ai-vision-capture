from aicapture import OpenAIVisionModel, VisionParser


def main():
    # Initialize OpenAI vision model (API key will be loaded from .env)
    vision_model = OpenAIVisionModel(model="gpt-4.1")

    image_path = "tests/sample/images/logic.png"

    # Initialize parser with a specific prompt for logical diagram analysis
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai",
        invalidate_cache=True,
    )

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

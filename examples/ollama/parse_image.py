from aicapture import OpenAIVisionModel, VisionParser

model = "qwen2.5vl:32b"


def main():
    # Initialize OpenAI vision model (API key will be loaded from .env)
    image_path = "tests/sample/images/logic.png"

    vision_model = OpenAIVisionModel(
        model=model,
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/ollama",
        invalidate_cache=False,  # change to True to invalidate cache
    )
    result = parser.process_image(image_path)

    # Print results

    # Print results
    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")

    # Print the extracted content
    page_content = result["file_object"]["pages"][0]["page_content"]
    print("\nExtracted Content:")
    print(page_content)


if __name__ == "__main__":
    main()

"""Parse an image using the OpenAI Responses API.

Requires:
    export OPENAI_API_KEY=your_key
"""

from aicapture import OpenAIResponsesVisionModel, VisionParser


def main():
    # Initialize model using the Responses API
    vision_model = OpenAIResponsesVisionModel(model="gpt-4.1")

    image_path = "tests/sample/images/logic.png"

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai-responses",
        invalidate_cache=True,
    )

    result = parser.process_image(image_path)

    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")

    page_content = result["file_object"]["pages"][0]["page_content"]
    print("\nExtracted Content:")
    print(page_content)


if __name__ == "__main__":
    main()

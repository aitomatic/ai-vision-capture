"""Parse an image using GPT-5 reasoning model with the Responses API.

The Responses API uses different parameter names for reasoning models:
- reasoning={"effort": "medium"} instead of reasoning_effort="medium"
- max_output_tokens instead of max_completion_tokens

Requires:
    export OPENAI_API_KEY=your_key
"""

from aicapture import OpenAIResponsesVisionModel, VisionParser


def main():
    # GPT-5.2 with reasoning enabled
    print("=" * 60)
    print("GPT-5.2 with reasoning (medium effort)")
    print("=" * 60)

    vision_model = OpenAIResponsesVisionModel(
        model="gpt-5.2",
        reasoning_effort="medium",  # Options: none, low, medium, high, xhigh
        max_output_tokens=5000,
    )

    image_path = "tests/sample/images/logic.png"

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai-responses-gpt5",
        invalidate_cache=True,
    )

    result = parser.process_image(image_path)

    print(f"\nFile: {result['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result["file_object"]["pages"][0]["page_content"])

    # GPT-5.2 instant mode (reasoning disabled, temperature allowed)
    print("\n" + "=" * 60)
    print("GPT-5.2 instant mode (reasoning_effort=none)")
    print("=" * 60)

    vision_model_instant = OpenAIResponsesVisionModel(
        model="gpt-5.2",
        reasoning_effort="none",
        max_output_tokens=5000,
        temperature=0.0,
    )

    parser_instant = VisionParser(
        vision_model=vision_model_instant,
        cache_dir="./.vision_cache/openai-responses-gpt5-instant",
        invalidate_cache=True,
    )

    result_instant = parser_instant.process_image(image_path)

    print(f"\nFile: {result_instant['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_instant["file_object"]["pages"][0]["page_content"])


if __name__ == "__main__":
    main()

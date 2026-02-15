"""Example of using GPT-5.2 reasoning model for image parsing.

GPT-5 series models (gpt-5, gpt-5.1, gpt-5.2) are reasoning models with different parameter requirements:
- Use max_completion_tokens instead of max_tokens
- Don't support temperature unless reasoning_effort is "none"
- Support reasoning_effort parameter (none, low, medium, high, xhigh)
"""

from aicapture import OpenAIVisionModel, VisionParser


def main():
    # Example 1: Using GPT-5.2 with reasoning enabled (default)
    print("=" * 80)
    print("Example 1: GPT-5.2 with reasoning enabled")
    print("=" * 80)

    vision_model = OpenAIVisionModel(
        model="gpt-5.2",
        reasoning_effort="medium",  # Options: none, low, medium, high, xhigh
        max_completion_tokens=5000,  # Use max_completion_tokens instead of max_tokens
    )

    image_path = "tests/sample/images/logic.png"

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/gpt5",
        invalidate_cache=True,
    )

    result = parser.process_image(image_path)

    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result["file_object"]["pages"][0]["page_content"])

    # Example 2: Using GPT-5.2 with reasoning disabled (allows temperature)
    print("\n" + "=" * 80)
    print("Example 2: GPT-5.2 with reasoning disabled (instant mode)")
    print("=" * 80)

    vision_model_instant = OpenAIVisionModel(
        model="gpt-5.2",
        reasoning_effort="none",  # Disable reasoning to allow temperature
        max_completion_tokens=5000,
        temperature=0.0,  # Temperature only works when reasoning_effort="none"
    )

    parser_instant = VisionParser(
        vision_model=vision_model_instant,
        cache_dir="./.vision_cache/gpt5-instant",
        invalidate_cache=True,
    )

    result_instant = parser_instant.process_image(image_path)

    print("\nProcessed Image Results (Instant Mode):")
    print(f"File: {result_instant['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_instant["file_object"]["pages"][0]["page_content"])

    # Example 3: Using GPT-5.1
    print("\n" + "=" * 80)
    print("Example 3: GPT-5.1 with high reasoning effort")
    print("=" * 80)

    vision_model_5_1 = OpenAIVisionModel(
        model="gpt-5.1",
        reasoning_effort="high",  # GPT-5.1 supports: none, low, medium, high
        max_completion_tokens=5000,
    )

    parser_5_1 = VisionParser(
        vision_model=vision_model_5_1,
        cache_dir="./.vision_cache/gpt5.1",
        invalidate_cache=True,
    )

    result_5_1 = parser_5_1.process_image(image_path)

    print("\nProcessed Image Results (GPT-5.1):")
    print(f"File: {result_5_1['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_5_1["file_object"]["pages"][0]["page_content"])


if __name__ == "__main__":
    main()

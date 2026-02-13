"""Example of using Azure OpenAI GPT-5.2 reasoning model for image parsing.

Azure OpenAI supports GPT-5 series reasoning models with the same parameter requirements
as standard OpenAI:
- Use max_completion_tokens instead of max_tokens
- Don't support temperature unless reasoning_effort is "none"
- Support reasoning_effort parameter (none, low, medium, high, xhigh)

Prerequisites:
- Deploy a GPT-5 model in your Azure OpenAI resource
- Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_URL, and AZURE_OPENAI_MODEL in .env
"""

from aicapture import AzureOpenAIVisionModel, VisionParser


def main():
    # Example 1: Azure GPT-5.2 with reasoning enabled (default)
    print("=" * 80)
    print("Example 1: Azure GPT-5.2 with reasoning enabled")
    print("=" * 80)

    vision_model = AzureOpenAIVisionModel(
        model="gpt-5.2",  # Your Azure deployment name
        reasoning_effort="medium",  # Options: none, low, medium, high, xhigh
        max_completion_tokens=5000,  # Use max_completion_tokens instead of max_tokens
    )

    image_path = "tests/sample/images/logic.png"

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/azure-gpt5",
        invalidate_cache=True,
    )

    result = parser.process_image(image_path)

    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result["file_object"]["pages"][0]["page_content"])

    # Example 2: Azure GPT-5.2 with reasoning disabled (instant mode)
    print("\n" + "=" * 80)
    print("Example 2: Azure GPT-5.2 with reasoning disabled (instant mode)")
    print("=" * 80)

    vision_model_instant = AzureOpenAIVisionModel(
        model="gpt-5.2",  # Your Azure deployment name
        reasoning_effort="none",  # Disable reasoning to allow temperature
        max_completion_tokens=5000,
        temperature=0.0,  # Temperature only works when reasoning_effort="none"
    )

    parser_instant = VisionParser(
        vision_model=vision_model_instant,
        cache_dir="./.vision_cache/azure-gpt5-instant",
        invalidate_cache=True,
    )

    result_instant = parser_instant.process_image(image_path)

    print("\nProcessed Image Results (Instant Mode):")
    print(f"File: {result_instant['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_instant["file_object"]["pages"][0]["page_content"])

    # Example 3: Azure GPT-5.1 with high reasoning effort
    print("\n" + "=" * 80)
    print("Example 3: Azure GPT-5.1 with high reasoning effort")
    print("=" * 80)

    vision_model_5_1 = AzureOpenAIVisionModel(
        model="gpt-5.1",  # Your Azure deployment name
        reasoning_effort="high",  # GPT-5.1 supports: none, low, medium, high
        max_completion_tokens=5000,
    )

    parser_5_1 = VisionParser(
        vision_model=vision_model_5_1,
        cache_dir="./.vision_cache/azure-gpt5.1",
        invalidate_cache=True,
    )

    result_5_1 = parser_5_1.process_image(image_path)

    print("\nProcessed Image Results (GPT-5.1):")
    print(f"File: {result_5_1['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_5_1["file_object"]["pages"][0]["page_content"])

    # Example 4: Using environment variables
    print("\n" + "=" * 80)
    print("Example 4: Using environment variables from .env")
    print("=" * 80)
    print("Set these in your .env file:")
    print("AZURE_OPENAI_MODEL=gpt-5.2")
    print("AZURE_OPENAI_REASONING_EFFORT=medium")
    print("")

    # This will use settings from .env
    vision_model_env = AzureOpenAIVisionModel()

    parser_env = VisionParser(
        vision_model=vision_model_env,
        cache_dir="./.vision_cache/azure-env",
        invalidate_cache=True,
    )

    result_env = parser_env.process_image(image_path)

    print(f"Model: {vision_model_env.model}")
    print(f"Is reasoning model: {vision_model_env.is_reasoning_model}")
    print(f"Reasoning effort: {vision_model_env.reasoning_effort}")
    print("\nProcessed Image Results (from .env):")
    print(f"File: {result_env['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_env["file_object"]["pages"][0]["page_content"])


if __name__ == "__main__":
    main()

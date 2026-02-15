"""Parse an image using Azure OpenAI GPT-5 with the Responses API.

Prerequisites:
- Deploy a GPT-5 model in your Azure OpenAI resource
- Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_URL in .env

Requires:
    export AZURE_OPENAI_API_KEY=your_key
    export AZURE_OPENAI_API_URL=https://your-resource.openai.azure.com
"""

from aicapture import AzureOpenAIResponsesVisionModel, VisionParser


def main():
    # GPT-5.2 with reasoning enabled
    print("=" * 60)
    print("Azure GPT-5.2 with reasoning (medium effort)")
    print("=" * 60)

    vision_model = AzureOpenAIResponsesVisionModel(
        model="gpt-5.2",  # Your Azure deployment name
        reasoning_effort="medium",
        max_output_tokens=5000,
    )

    image_path = "tests/sample/images/logic.png"

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/azure-responses-gpt5",
        invalidate_cache=True,
    )

    result = parser.process_image(image_path)

    print(f"\nFile: {result['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result["file_object"]["pages"][0]["page_content"])

    # GPT-5.2 instant mode
    print("\n" + "=" * 60)
    print("Azure GPT-5.2 instant mode (reasoning_effort=none)")
    print("=" * 60)

    vision_model_instant = AzureOpenAIResponsesVisionModel(
        model="gpt-5.2",
        reasoning_effort="none",
        max_output_tokens=5000,
        temperature=0.0,
    )

    parser_instant = VisionParser(
        vision_model=vision_model_instant,
        cache_dir="./.vision_cache/azure-responses-gpt5-instant",
        invalidate_cache=True,
    )

    result_instant = parser_instant.process_image(image_path)

    print(f"\nFile: {result_instant['file_object']['file_name']}")
    print("\nExtracted Content:")
    print(result_instant["file_object"]["pages"][0]["page_content"])


if __name__ == "__main__":
    main()

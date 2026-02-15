"""Parse a PDF using the Azure OpenAI Responses API.

Requires:
    export AZURE_OPENAI_API_KEY=your_key
    export AZURE_OPENAI_API_URL=https://your-resource.openai.azure.com
"""

from aicapture import AzureOpenAIResponsesVisionModel, VisionParser


def main():
    vision_model = AzureOpenAIResponsesVisionModel()

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/azure-responses",
        invalidate_cache=True,
    )

    result = parser.process_pdf("tests/sample/pdfs/sample.pdf")

    print(f"Document: {result['file_object']['file_name']}")
    print(f"Pages: {result['file_object']['total_pages']}")
    print(f"Words: {result['file_object']['total_words']}")


if __name__ == "__main__":
    main()

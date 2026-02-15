"""Parse a PDF using the OpenAI Responses API.

Requires:
    export OPENAI_API_KEY=your_key
"""

from aicapture import OpenAIResponsesVisionModel, VisionParser


def main():
    vision_model = OpenAIResponsesVisionModel()

    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai-responses",
        invalidate_cache=False,  # change to True to invalidate cache
    )

    result = parser.process_pdf("tests/sample/pdfs/sample.pdf")

    print(f"Document: {result['file_object']['file_name']}")
    print(f"Pages: {result['file_object']['total_pages']}")
    print(f"Words: {result['file_object']['total_words']}")


if __name__ == "__main__":
    main()

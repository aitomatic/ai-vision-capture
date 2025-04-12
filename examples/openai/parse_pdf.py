from aicapture import OpenAIVisionModel, VisionParser


def main():
    # Initialize OpenAI vision model (API key will be loaded from .env)
    vision_model = OpenAIVisionModel()

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai",
        invalidate_cache=False,  # change to True to invalidate cache
    )

    # Process a single PDF
    result = parser.process_pdf("tests/sample/pdfs/sample.pdf")

    # Save results
    # parser.save_output(result, "output.json")
    # parser.save_markdown_output(result, "output")

    print(f"Document: {result['file_object']['file_name']}")
    print(f"Pages: {result['file_object']['total_pages']}")
    print(f"Words: {result['file_object']['total_words']}")


if __name__ == "__main__":
    main()

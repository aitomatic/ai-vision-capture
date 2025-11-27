"""
Example of parsing PDFs using auto-detected vision model.

This example demonstrates how to use the AutoDetectVisionModel feature,
which automatically detects and uses the first available vision model
based on your environment variables.

The detection order is:
1. Gemini (GEMINI_API_KEY)
2. OpenAI (OPENAI_API_KEY or OPENAI_VISION_API_KEY)
3. Azure OpenAI (AZURE_OPENAI_API_KEY)
4. Anthropic (ANTHROPIC_API_KEY)

Simply set any of these environment variables and the library will
automatically use the first available provider!
"""

import time

from aicapture import VisionParser, create_default_vision_model


def main():
    # Auto-detect vision model based on available API keys
    # No need to set USE_VISION environment variable!
    vision_model = create_default_vision_model()

    print(f"Using auto-detected model: {vision_model.__class__.__name__}")
    print(f"Model: {vision_model.model}")

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/auto",
        invalidate_cache=False,  # change to True to invalidate cache
    )

    # Process a single PDF
    pdf_path = "tests/sample/pdfs/sample.pdf"

    # Or process an entire folder
    # folder_path = "tmp/long"
    # result = parser.process_folder(folder_path)

    result = parser.process_pdf(pdf_path)

    # Print summary
    print("\nProcessed PDF Results:")
    print(f"File: {result['file_object']['file_name']}")
    print(f"Total Pages: {result['file_object']['total_pages']}")
    print(f"Total Words: {result['file_object']['total_words']}")

    # Print first page content preview
    first_page = result["file_object"]["pages"][0]
    print("\nFirst Page Content (preview):")
    print(first_page["page_content"][:500] + "...")

    # Save results (optional)
    # parser.save_output(result, "output.json")
    # parser.save_markdown_output(result, "output")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"\nTime taken: {t2 - t1:.2f} seconds")

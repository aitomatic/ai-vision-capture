from aicapture import OpenAIVisionModel, VisionParser


def main():
    # Initialize OpenAI vision model (API key will be loaded from .env)
    vision_model = OpenAIVisionModel(
        model="gpt-4o",
        temperature=0.2,
    )

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai",
        invalidate_cache=False,  # change to True to invalidate cache
        prompt="""
        Extract from this technical document:
        1. Main topics and key points
        2. Technical specifications
        3. Important procedures
        4. Tables and data (in markdown)
        5. Diagrams and figures
        
        Preserve all numerical values and maintain document structure.
        """,
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

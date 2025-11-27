from aicapture import OpenAIVisionModel, VisionParser

### set up ollama
# ollama run llama3.2-vision:11b-instruct-q4_K_M
# model = "llama3.2-vision:11b-instruct-q4_K_M"

model = "qwen2.5vl:32b"


def main():
    vision_model = OpenAIVisionModel(
        model=model,
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai",
        invalidate_cache=True,  # change to True to invalidate cache
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
    print(result)


if __name__ == "__main__":
    main()

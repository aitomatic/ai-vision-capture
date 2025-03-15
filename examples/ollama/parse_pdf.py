from vision_capture import OpenAIVisionModel, VisionParser

### set up ollama
# ollama run llama3.2-vision:11b-instruct-q4_K_M


def main():
    # Initialize OpenAI vision model (API key will be loaded from .env)
    vision_model = OpenAIVisionModel(
        model="llama3.2-vision:11b-instruct-q4_K_M",
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
    result = parser.process_pdf("examples/pdfs/sample.pdf")
    print(result)


if __name__ == "__main__":
    main()

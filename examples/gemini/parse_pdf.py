from aicapture import GeminiVisionModel, VisionParser


def main():
    # Initialize Gemini vision model (API key will be loaded from .env)
    vision_model = GeminiVisionModel(
        model="gemini-2.5-flash",
        temperature=0.0,
    )

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/gemini",
        invalidate_cache=True,  # change to True to invalidate cache
    )

    # Process a single PDF
    # pdf_path = "tests/sample/pdfs/sample.pdf"
    folder_path = "tmp/long"
    result = parser.process_folder(folder_path)

    # Save results
    # parser.save_output(result, "output.json")
    # parser.save_markdown_output(result, "output")

    from pprint import pprint

    pprint(result[0]["file_object"]["pages"][0])


if __name__ == "__main__":
    import time

    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")

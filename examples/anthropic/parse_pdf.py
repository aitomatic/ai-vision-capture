from aicapture import AnthropicVisionModel, VisionParser


def main():
    vision_model = AnthropicVisionModel(
        model="claude-3-7-sonnet-20250219",
        temperature=0.0,
    )

    # Initialize parser
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/anthropic",
        invalidate_cache=True,  # change to True to invalidate cache
        cloud_bucket="aitomatic-dxa-data",
    )

    # Process a single PDF
    result = parser.process_pdf("tests/sample/pdfs/sample.pdf")
    # result = parser.process_folder("tmp/terna")

    # result is a dictionary containing the parsed document
    # you can save the result to a json file
    # parser.save_output(result, "output.json")
    # a markdown file is also saved automatically at tmp/md

    print(f"Document: {result['file_object']['file_name']}")
    print(f"Pages: {result['file_object']['total_pages']}")
    print(f"Words: {result['file_object']['total_words']}")


if __name__ == "__main__":
    main()

import os

from aicapture import AnthropicVisionModel, VisionParser


def main():
    vision_model = AnthropicVisionModel(
        model="claude-3-7-sonnet-20250219",
        temperature=0.0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    image_path = "tests/sample/images/logic.png"

    # Initialize parser with a specific prompt for logical diagram analysis
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/anthropic",
        invalidate_cache=True,  # Set to True to force reprocessing
        prompt="""
        You are an expert electrical engineer. The uploaded image is a logical diagram
        that processes input signals (sensors) on the left-hand side and produces an
        output result on the right-hand side. There is an alarm in the diagram.
        Identify all possible input signals (dependencies) that contribute to triggering
        this alarm. Provide the exact sensor names and any relevant identifiers
        (e.g., TAG, RF LOGICA) for each dependency. Be precise.
        """,
    )

    # Process the image
    result = parser.process_image(image_path)

    # Print results
    print("\nProcessed Image Results:")
    print(f"File: {result['file_object']['file_name']}")

    # Print the extracted content
    page_content = result["file_object"]["pages"][0]["page_content"]
    print("\nExtracted Content:")
    print(page_content)


if __name__ == "__main__":
    main()

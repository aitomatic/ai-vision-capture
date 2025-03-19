from vision_capture import OpenAIVisionModel, VisionParser


def main():
    # Initialize OpenAI vision model (API key will be loaded from .env)
    vision_model = OpenAIVisionModel(
        model="gpt-4o",  # Use the vision-capable model
        temperature=0.2,
    )

    image_path = "tests/sample/images/logic.png"

    # Initialize parser with a specific prompt for logical diagram analysis
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/openai",
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

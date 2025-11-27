import os

from aicapture import AnthropicAWSBedrockVisionModel, VisionParser


def main():
    # Initialize the Anthropic AWS Bedrock Vision Model
    vision_model = AnthropicAWSBedrockVisionModel(
        model=os.getenv("ANTHROPIC_BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_vpc_endpoint_url=os.getenv("AWS_BEDROCK_VPC_ENDPOINT_URL"),
    )

    # Path to the sample image
    image_path = "tests/sample/images/logic.png"

    # Initialize parser with a specific prompt for logical diagram analysis
    parser = VisionParser(
        vision_model=vision_model,
        cache_dir="./.vision_cache/anthropic_bedrock",
        invalidate_cache=True,  # Set to True to force reprocessing
        # prompt="""
        # You are an expert electrical engineer. The uploaded image is a logical diagram
        # that processes input signals (sensors) on the left-hand side and produces an
        # output result on the right-hand side. There is an alarm in the diagram.
        # Identify all possible input signals (dependencies) that contribute to triggering
        # this alarm. Provide the exact sensor names and any relevant identifiers
        # (e.g., TAG, RF LOGICA) for each dependency. Be precise.
        # """,
    )

    # Process the image
    result = parser.process_file(image_path)

    # Print results
    print("\nProcessed Image Results via AWS Bedrock:")
    print(f"File: {result['file_object']['file_name']}")

    # Print the extracted content
    page_content = result["file_object"]["pages"][0]["page_content"]
    print("\nExtracted Content:")
    print(page_content)

    # Print token usage if available
    if hasattr(vision_model, "last_token_usage") and vision_model.last_token_usage:
        print("\nToken Usage:")
        for key, value in vision_model.last_token_usage.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

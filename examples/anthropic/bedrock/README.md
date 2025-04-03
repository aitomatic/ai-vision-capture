# Anthropic Claude via AWS Bedrock Example

This example demonstrates how to use Anthropic Claude Vision models through AWS Bedrock to analyze images using AI Vision Capture.

## Prerequisites

- An AWS account with access to Claude on AWS Bedrock
- AWS credentials with permissions to call Bedrock APIs
- Python 3.8+ with the AI Vision Capture package installed

## Setup

1. Configure your AWS credentials and Bedrock model settings:

```bash
export USE_VISION=anthropic_bedrock
export ANTHROPIC_BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
export AWS_ACCESS_KEY_ID=your_aws_access_key_id
export AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
export AWS_REGION=us-east-1  # or your preferred region
```

2. Optional: Configure AWS VPC endpoint if using PrivateLink:

```bash
export AWS_BEDROCK_VPC_ENDPOINT_URL=https://your-vpc-endpoint-url
```

## Running the Example

```bash
# From the project root directory
python examples/anthropic/bedrock/parser_image.py
```

This script will:
1. Initialize an AnthropicAWSBedrockVisionModel with your AWS credentials
2. Load a sample logical diagram image
3. Create a VisionParser with a specialized engineering prompt
4. Process the image and extract relevant information
5. Display the results and token usage

## Troubleshooting

- If you encounter authentication errors, verify your AWS credentials and permissions
- Check that Claude models are enabled in your AWS Bedrock account
- Ensure your AWS region has Bedrock and Claude available

## Using in Your Own Projects

You can integrate AWS Bedrock Claude Vision into your projects in two ways:

1. Using environment variables and default provider:

```python
from aicapture import create_default_vision_model, VisionParser

# Will use AWS Bedrock when USE_VISION=anthropic_bedrock
model = create_default_vision_model()
parser = VisionParser(vision_model=model)
```

2. Explicit configuration:

```python
from aicapture import AnthropicAWSBedrockVisionModel, VisionParser

model = AnthropicAWSBedrockVisionModel(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_access_key_id="your_key",
    aws_secret_access_key="your_secret",
    aws_region="us-east-1"
)
parser = VisionParser(vision_model=model)
``` 
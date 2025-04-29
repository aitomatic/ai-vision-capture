# AI Vision Capture Configuration Guide

This guide provides detailed configuration options for AI Vision Capture in different environments.

## Environment Variables Reference

### Vision Provider Selection
Choose ONE of the following providers:

```bash
# OpenAI
export USE_VISION=openai
export OPENAI_API_KEY=your_openai_key
export OPENAI_VISION_MODEL=gpt-4o

# Anthropic Claude
export USE_VISION=claude
export ANTHROPIC_API_KEY=your_anthropic_key
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Google Gemini
export USE_VISION=gemini
export GEMINI_API_KEY=your_gemini_key
export GEMINI_MODEL=gemini-2.5-flash-preview-04-17

# Azure OpenAI
export USE_VISION=azure-openai
export AZURE_OPENAI_API_KEY=your_azure_key
export AZURE_OPENAI_MODEL=gpt-4o
export AZURE_OPENAI_API_URL=https://xxx.openai.azure.com
export AZURE_OPENAI_API_VERSION=2025-02-15-preview
```

### Anthropic AWS Bedrock
```bash
export USE_VISION=anthropic_bedrock
export ANTHROPIC_BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
export AWS_ACCESS_KEY_ID=your_aws_access_key_id
export AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
export AWS_REGION=your_aws_region
export AWS_BEDROCK_VPC_ENDPOINT_URL=your_aws_vpc_endpoint_url
```

### Performance Settings
```bash
# Concurrent Processing
export MAX_CONCURRENT_TASKS=5
export VISION_PARSER_DPI=333
```

## Development Setup

For local development, you can use a `.env` file instead of setting environment variables:

1. Create a new file named `.env` in your project root
2. Copy the relevant environment variables from above
3. Replace `export` with simple variable assignment
4. Add your values

Example `.env` file:
```bash
USE_VISION=openai
OPENAI_API_KEY=sk-...
MAX_CONCURRENT_TASKS=5
VISION_PARSER_DPI=333
```

## Advanced Configuration Examples

### Custom Vision Model Setup
```python
from vision_capture import VisionParser, OpenAIVisionModel

model = OpenAIVisionModel(
    model="gpt-4o",
    api_key="your_key",
    max_tokens=8000,
    temperature=0.0
)

parser = VisionParser(vision_model=model)
```

### Batch Processing Configuration
```python
parser = VisionParser(
    max_concurrent_tasks=10,
    dpi=400,
    cache_enabled=True
)
``` 
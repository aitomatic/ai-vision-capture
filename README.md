# AI Vision Capture

A powerful Python library for extracting and analyzing content from PDF documents using Vision Language Models (VLMs). This library provides a flexible and efficient way to process documents with support for multiple VLM providers including OpenAI, Anthropic Claude, Google Gemini, and Azure OpenAI.

## Features

- 🔍 **Multi-Provider Support**: Compatible with major VLM providers (OpenAI, Claude, Gemini, Azure)
- 📄 **PDF Processing**: Efficient PDF to image conversion with configurable DPI
- 🚀 **Async Processing**: Asynchronous processing with configurable concurrency
- 💾 **Two-Layer Caching**: Local file system and S3 caching for improved performance
- 🔄 **Batch Processing**: Process multiple PDFs in parallel
- 📝 **Text Extraction**: Enhanced accuracy through combined OCR and VLM processing
- 🎨 **Image Quality Control**: Configurable image quality settings
- 📊 **Structured Output**: Well-organized JSON and Markdown output

## Installation

```bash
pip install ai-vision-capture
```

## Quick Start

```python
from vision_capture import VisionParser

# Initialize parser
parser = VisionParser()

# Process a single PDF
result = parser.process_pdf("path/to/your/document.pdf")

# Process a folder of PDFs asynchronously
async def process_folder():
    results = await parser.process_folder_async("path/to/folder")
    return results
```

## Configuration

The library can be configured through environment variables:

```env
# Vision Model Selection
USE_VISION=openai  # Options: openai, claude, gemini, azure-openai

# API Keys
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
AZURE_OPENAI_API_KEY=your_key

# Cache Settings
DXA_DATA_BUCKET=your_s3_bucket_name

# Performance Settings
MAX_CONCURRENT_TASKS=5
VISION_PARSER_DPI=345
```

## Output Format

The library produces structured output in both JSON and Markdown formats:

```json
{
  "file_object": {
    "file_name": "example.pdf",
    "file_hash": "sha256_hash",
    "total_pages": 10,
    "total_words": 5000,
    "pages": [
      {
        "page_number": 1,
        "page_content": "extracted content",
        "page_hash": "sha256_hash"
      }
    ]
  }
}
```

## Advanced Usage

### Custom Vision Models

```python
from vision_capture import VisionParser, OpenAIVisionModel

# Configure custom vision model
vision_model = OpenAIVisionModel(
    model="gpt-4-vision-preview",
    api_key="your_key"
)

# Initialize parser with custom model
parser = VisionParser(
    vision_model=vision_model,
    image_quality="high",
    invalidate_cache=False
)
```

### Custom Prompts

```python
# Initialize parser with custom prompt
parser = VisionParser(
    prompt="""
    Extract the following information:
    1. Key technical specifications
    2. Important measurements
    3. Critical warnings
    """
)
```

## Contributing

--

## License

--
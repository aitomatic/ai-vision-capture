# AI Vision Capture

A powerful Python library for extracting and analyzing content from PDF documents using Vision Language Models (VLMs). This library provides a flexible and efficient way to process documents with support for multiple VLM providers including OpenAI, Anthropic Claude, Google Gemini, and Azure OpenAI.

## Features

- 🔍 **Multi-Provider Support**: Compatible with major VLM providers (OpenAI, Claude, Gemini, Azure, OpenSource models)
- 📄 **PDF Processing**: Efficient PDF to image conversion with configurable DPI
- 🚀 **Async Processing**: Asynchronous processing with configurable concurrency
- 💾 **Two-Layer Caching**: Local file system and cloud caching for improved performance
- 🔄 **Batch Processing**: Process multiple PDFs in parallel
- 📝 **Text Extraction**: Enhanced accuracy through combined OCR and VLM processing
- 🎨 **Image Quality Control**: Configurable image quality settings
- 📊 **Structured Output**: Well-organized JSON and Markdown output


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
VISION_PARSER_DPI=333
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

```python
from vision_capture import VisionParser, GeminiVisionModel

# Configure Gemini vision model with custom settings
vision_model = GeminiVisionModel(
    model="gemini-pro-vision",
    api_key="your_gemini_api_key"
)

# Initialize parser with custom configuration
parser = VisionParser(
    vision_model=vision_model,
    dpi=400,
    image_quality="high",
    prompt="""
    Please analyze this technical document and extract:
    1. Equipment specifications and model numbers
    2. Operating parameters and limits
    3. Maintenance requirements
    4. Safety protocols
    5. Quality control metrics
    """
)

# Process PDF with custom settings
result = parser.process_pdf(
    pdf_path="path/to/document.pdf",
    cache_enabled=True
)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For detailed guidelines, see our [Contributing Guide](CONTRIBUTING.md).

## License

Copyright 2024 Aitomatic, Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
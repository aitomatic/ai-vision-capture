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

## Coming Soon Features

- 🔗 **Cross-Document Knowledge Capture**: Capture structured knowledge across multiple documents

- 🎥 **Video Knowledge Capture**: Capture structured knowledge from video

## Quick Start

### Installation

```bash
pip install aicapture
```

### Basic Setup

1. Set your chosen provider and API key (example using OpenAI):
```bash
export USE_VISION=openai
export OPENAI_API_KEY=your_openai_key
```

2. Use in your code:
```python
from vision_capture import VisionParser

# Initialize parser
parser = VisionParser()

# Process a PDF
result = parser.process_pdf("path/to/your/document.pdf")

# Process multiple PDFs asynchronously
async def process_folder():
    results = await parser.process_folder_async("path/to/folder")
    return results
```

For detailed configuration options and examples, see:
- [Configuration Guide](examples/configuration.md)
- [Advanced Usage Examples](examples/configuration.md#advanced-configuration-examples)

## Configuration

### Production Environment
In production, configure the library using environment variables in your shell or deployment environment.

Common settings you may want to adjust:
```bash
# Optional performance settings
export MAX_CONCURRENT_TASKS=5      # Number of concurrent processing tasks
export VISION_PARSER_DPI=333      # Image DPI for PDF processing
```

### Development Environment
For local development:

1. Clone the repository
2. Copy `.env.template` to `.env`
3. Edit `.env` with your settings
4. Install development dependencies: `pip install -e ".[dev]"`

See `.env.template` for all available configuration options.

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
    model="gemini-2.0-flash",
    api_key="your_gemini_api_key"
)

# Initialize parser with custom configuration
parser = VisionParser(
    vision_model=vision_model,
    dpi=400,
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
)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/tiny-but-mighty`)
3. Commit your changes (`git commit -m 'feat: add small but delightful improvement'`)
4. Push to the branch (`git push origin feature/tiny-but-mighty`)
5. Open a Pull Request

For detailed guidelines, see our [Contributing Guide](CONTRIBUTING.md).

## License

Copyright 2024 Aitomatic, Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

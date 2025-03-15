# AI Vision Capture

A powerful Python library for extracting and analyzing content from PDF documents using Vision Language Models (VLMs). This library provides a flexible and efficient way to process documents with support for multiple VLM providers including OpenAI, Anthropic Claude, Google Gemini, and Azure OpenAI.

## Features

- 🔍 **Multi-Provider Support**: Compatible with major VLM providers (OpenAI, Claude, Gemini, Azure, OpenSource models)
- 📄 **Document Processing**: Process PDFs and images (JPG, PNG, TIFF, WebP, BMP)
- 🚀 **Async Processing**: Asynchronous processing with configurable concurrency
- 💾 **Two-Layer Caching**: Local file system and cloud caching for improved performance
- 🔄 **Batch Processing**: Process multiple documents in parallel
- 📝 **Text Extraction**: Enhanced accuracy through combined OCR and VLM processing
- 🎨 **Image Quality Control**: Configurable image quality settings
- 📊 **Structured Output**: Well-organized JSON and Markdown output

## Installation

```bash
pip install aicapture
```

## Environment Setup

1. Set your chosen provider and API key:
```bash
# For OpenAI
export USE_VISION=openai
export OPENAI_API_KEY=your_openai_key

# For Anthropic
export USE_VISION=anthropic
export ANTHROPIC_API_KEY=your_anthropic_key

# For Gemini
export USE_VISION=gemini
export GEMINI_API_KEY=your_google_key
```

2. Optional performance settings:
```bash
export MAX_CONCURRENT_TASKS=5      # Number of concurrent processing tasks
export VISION_PARSER_DPI=333      # Image DPI for PDF processing
```

## Core Capabilities

### 1. Document Parsing

The VisionParser provides general document processing capabilities for extracting unstructured content from documents.

```python
from vision_capture import VisionParser

# Initialize parser
parser = VisionParser()

# Process a single PDF
result = parser.process_pdf("path/to/your/document.pdf")

# Process a single image
result = parser.process_image("path/to/your/image.jpg")

# Process multiple documents asynchronously
async def process_folder():
    return await parser.process_folder_async("path/to/folder")
```

#### Parser Output Format

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

### 2. Structured Data Capture

The VisionCapture component enables extraction of structured data from images using customizable templates.

1. Define your data template:
```python
# Example template for technical alarm logic
ALARM_TEMPLATE = """
alarm:
  description: string  # Main alarm description
  destination: string # Destination system
  tag: string        # Alarm tag
  ref_logica: integer # Logic reference number

dependencies:
  type: array
  items:
    - signal_name: string  # Name of the dependency signal
      source: string      # Source system/component
      tag: string        # Signal tag
      ref_logica: integer|null  # Logic reference (can be null)
"""
```

2. Use with OpenAI Vision:
```python
from vision_capture import VisionCapture
from vision_capture.vision_models import OpenAIVisionModel

vision_model = OpenAIVisionModel(
    model="gpt-4-vision-preview",
    max_tokens=4096,
    api_key="your_openai_key"
)

capture = VisionCapture(vision_model=vision_model)
result = await capture.capture(
    file_path="path/to/image.png",
    template=ALARM_TEMPLATE
)
```

3. Or use with Anthropic Claude:
```python
from vision_capture.vision_models import AnthropicVisionModel

vision_model = AnthropicVisionModel(
    model="claude-3-sonnet-20240620",
    max_tokens=4096,
    api_key="your_anthropic_key"
)

capture = VisionCapture(vision_model=vision_model)
result = await capture.capture(
    file_path="path/to/example.pdf",
    template=ALARM_TEMPLATE
)
```

## Advanced Usage

### Custom Vision Model Configuration

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

## Development Setup

For local development:

1. Clone the repository
2. Copy `.env.template` to `.env`
3. Edit `.env` with your settings
4. Install development dependencies: `pip install -e ".[dev]"`

See `.env.template` for all available configuration options.

## Documentation

For detailed configuration options and examples, see:
- [Configuration Guide](examples/configuration.md)
- [Advanced Usage Examples](examples/configuration.md#advanced-configuration-examples)

## Coming Soon

- 🔗 **Cross-Document Knowledge Capture**: Capture structured knowledge across multiple documents
- 🎥 **Video Knowledge Capture**: Capture structured knowledge from video

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

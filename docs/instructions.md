# Vision Parser Documentation

## Overview
Vision Parser is a Python module designed to extract content (text, tables, and visual elements) from PDF documents using Vision Language Models (VLMs). The module processes PDFs page by page and generates structured JSON output containing the extracted content.

## Tech Stack

### Core Dependencies
- Python 3.10+
- pdf2image: For PDF to image conversion
- Vision Language Models (Choose one):
  - OpenAI GPT-4o
  - Anthropic Claude 3.5
  - Google Gemini Vision

### Key Features
- PDF processing and metadata extraction
- PDF to image conversion
- Page-by-page content extraction
- Structured JSON output
- Support for multiple VLM providers

## Architecture

### Processing Flow
1. PDF Input Processing
   - Read PDF from input folder
   - Extract file metadata (name, hash, page count)
   
2. Image Conversion
   - Convert PDF pages to images
   - Process image metadata

3. Content Extraction
   - Send each page image to VLM
   - Process VLM response
   - Extract text and identify visual elements

4. Output Generation
   - Combine all page contents
   - Generate structured JSON output

### Output Schema
```json
{
    "file_object": {
        "file_name": "string",
        "file_hash": "string",
        "total_pages": "integer",
        "total_words": "integer",
        "file_full_path": "string",
        "pages": [
            {
                "page_number": "integer",
                "page_content": "string",
                "page_hash": "string",
                "page_objects": [
                    {
                        "md": "string (markdown content)",
                        "has_image": "boolean"
                    }
                ]
            }
        ]
    }
}
```

## Implementation Examples

### PDF to Image Conversion
```python
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path: str) -> list:
    """Convert PDF file to list of images."""
    return convert_from_path(pdf_path)
```

### File Hash Calculation
```python
import hashlib

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
```

### VLM Integration Examples

#### Anthropic Claude
```python
import anthropic

def process_with_claude(image, client: anthropic.Client) -> dict:
    """Process image using Claude Vision."""
    image_data, media_type = convert_image_to_base64(image)
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": "Extract the content with full detail in markdown format"}
            ]
        }]
    )
    return response.content[0].text
```

#### OpenAI GPT-4o
```python
from openai import OpenAI

def process_with_gpt4o(image_path: str, client: OpenAI) -> dict:
    """Process image using GPT-4o."""
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the content with full detail"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }]
    )
    return response.choices[0].message.content
```

## Getting Started

1. Install Dependencies
   ```bash
   pip install pdf2image anthropic openai pillow
   ```

2. Configure Environment
   - Set up API keys for chosen VLM provider
   - Ensure poppler is installed for pdf2image

3. Basic Usage
   ```python
   from vision_parser import VisionParser
   
   parser = VisionParser(model="claude")  # or "gpt4o"
   result = parser.process_pdf("path/to/document.pdf")
   ```

## Best Practices
- Handle large PDFs in chunks to manage memory
- Implement proper error handling for API calls
- Cache processed results to avoid redundant processing
- Validate input PDFs before processing
- Monitor API usage and costs

## Error Handling
- Implement retries for API failures
- Validate PDF file integrity
- Handle corrupt images
- Manage API rate limits

## Future Improvements
- Support for more VLM providers
- Batch processing capabilities
- Custom prompt templates
- Enhanced error recovery
- Progress tracking and logging



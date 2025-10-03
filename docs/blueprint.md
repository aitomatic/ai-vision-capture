# Vision Parser Documentation

## Overview
Vision Parser is a Python module designed to extract content (text, tables, and visual elements) from PDF documents using Vision Language Models (VLMs). The module processes PDFs page by page and generates structured JSON output containing the extracted content.

## Tech Stack

### Core Dependencies
- Python 3.10+
- PyMuPDF (fitz): For PDF processing and image conversion
- PIL (Pillow): For image handling
- Vision Language Models (Choose one):
  - OpenAI Vision
  - Anthropic Claude
  - Google Gemini Vision
  - Azure OpenAI Vision
  - OpenAI-compatible API providers

### Key Features
- PDF processing and metadata extraction
- PDF to image conversion with PyMuPDF
- Page-by-page content extraction
- Structured JSON output
- Support for multiple VLM providers
- Local and S3-based caching
- Token usage tracking
- Configurable image quality

## Architecture

### Processing Flow
1. PDF Input Processing
   - Read PDF from input folder
   - Extract file metadata (name, hash, page count)
   - Extract text content for reference
   
2. Image Conversion
   - Convert PDF pages to images using PyMuPDF
   - Configure DPI and quality settings
   - Optional image caching

3. Content Extraction
   - Send each page image to VLM
   - Process VLM response
   - Extract text and identify visual elements
   - Track token usage

4. Output Generation
   - Combine all page contents
   - Generate structured JSON output
   - Optional: Generate markdown output
   - Cache results (local/S3)

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
import fitz
from PIL import Image

def convert_pdf_to_images(pdf_path: str, dpi: int = 333) -> list:
    """Convert PDF file to list of images using PyMuPDF."""
    images = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            zoom = dpi / 72  # Convert DPI to zoom factor
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images
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

#### OpenAI GPT-4 Vision
```python
from openai import OpenAI

def process_with_gpt(image_path: str, client: OpenAI) -> dict:
    """Process image using GPT-4.1."""
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the content with full detail"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }]
    )
    return response.choices[0].message.content
```


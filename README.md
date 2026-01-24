# AI Vision Capture

A Python library for extracting and analyzing content from PDF, Image, and Video files using Vision Language Models (VLMs). Supports multiple providers including OpenAI, Anthropic Claude, Google Gemini, and Azure OpenAI.

**Package Name:** `aicapture`
**Python Version:** >=3.10
**Build System:** `hatchling` via `uv`

## Features

- **Multi-Provider Support**: OpenAI, Claude, Gemini, Azure OpenAI, AWS Bedrock
- **Document Processing**: PDFs and images (JPG, PNG, TIFF, WebP, BMP)
- **Video Processing**: Frame extraction and analysis from MP4, AVI, MOV, MKV
- **Audio Transcription**: Whisper-powered speech-to-text with timestamps (LRC format)
- **Async Processing**: Configurable concurrency for batch operations
- **Two-Layer Caching**: Local file system + optional S3 cloud cache
- **Structured Output**: Template-based data extraction to JSON
- **Pluggable Architecture**: Provider abstraction with auto-detection

## Installation

```bash
pip install aicapture

# With video transcription support (adds moviepy for audio extraction)
pip install aicapture[video]
```

## Environment Setup

Set your provider API key (auto-detection picks the first available):

```bash
export OPENAI_API_KEY=your_key      # OpenAI
# or
export GEMINI_API_KEY=your_key      # Gemini
# or
export ANTHROPIC_API_KEY=your_key   # Anthropic
```

Optional settings:

```bash
export USE_VISION=openai            # Force specific provider (openai|claude|gemini|azure-openai|anthropic_bedrock)
export MAX_CONCURRENT_TASKS=5       # Concurrent API requests (default: 20)
export VISION_PARSER_DPI=333        # PDF rendering quality
```

See `.env.template` for full reference of available configuration options.

## Usage

### 1. Document Parsing

```python
from aicapture import VisionParser

parser = VisionParser()

# Process PDF or image
result = parser.process_pdf("document.pdf")
result = parser.process_image("photo.jpg")

# Batch processing
async def batch():
    return await parser.process_folder_async("path/to/folder")
```

### 2. Structured Data Capture

```python
from aicapture import VisionCapture, OpenAIVisionModel

model = OpenAIVisionModel(model="gpt-4.1", api_key="your_key")
capture = VisionCapture(vision_model=model)

template = """
alarm:
  description: string
  tag: string
  ref_logica: integer
"""

result = await capture.capture(file_path="diagram.png", template=template)
```

### 3. Video Processing

```python
from aicapture import VidCapture, VideoConfig

config = VideoConfig(
    frame_rate=2,                # 2 frames per second
    max_duration_seconds=30,     # Max video length to process
    target_frame_size=(768, 768),
)

vid = VidCapture(config=config)
result = vid.process_video("video.mp4", "Describe what happens in this video.")
```

### 4. Video Processing with Audio Transcription

Extract speech from video using OpenAI Whisper and include it as context for richer analysis:

```python
from aicapture import VidCapture, VideoConfig

config = VideoConfig(
    frame_rate=2,                    # Extract 2 frames per second
    enable_transcription=True,       # Enable Whisper transcription
    transcription_model="whisper-1", # OpenAI Whisper model
    max_duration_seconds=600,        # Process up to 10 minutes
    transcription_language="en",     # Optional language hint (None for auto-detect)
    # Transcriptions are cached in: tmp/.vid_capture_cache/transcriptions/
    cache_dir="tmp/.vid_capture_cache",

)

vid = VidCapture(config=config)
result = vid.process_video("lecture.mp4", "Summarize the key points discussed.")
```

The transcription is extracted with segment-level timestamps (LRC format) and automatically appended to the VLM prompt:

```
[00:00.00] Welcome to this tutorial on machine learning.
[00:05.32] Today we will cover three main topics.
[00:09.15] First, let's discuss data preprocessing.
```

You can also use the transcriber directly:

```python
from aicapture import OpenAIAudioTranscriber

transcriber = OpenAIAudioTranscriber(model="whisper-1")
transcription = transcriber.transcribe_video("video.mp4")

print(transcription.to_lrc())        # LRC-formatted output
print(transcription.full_text)       # Plain text
print(transcription.segments)        # List of TimestampedSegments
```

## Advanced Configuration

```python
from aicapture import VisionParser, GeminiVisionModel

model = GeminiVisionModel(model="gemini-2.5-flash", api_key="your_key")

parser = VisionParser(
    vision_model=model,
    dpi=400,
    prompt="Extract equipment specs, operating parameters, and safety protocols.",
)

result = parser.process_pdf("technical_manual.pdf")
```

## Architecture

### Core Components

1. **VisionParser** (`aicapture/vision_parser.py`)
   - Main entry point for document processing (PDFs, images)
   - Handles PDF-to-image conversion using PyMuPDF (fitz)
   - Manages concurrent processing with semaphore-based throttling
   - Supports batch processing via `process_folder_async()`
   - Default DPI: 333 (configurable via `VISION_PARSER_DPI` env var)

2. **Vision Models** (`aicapture/vision_models.py`)
   - Abstract `VisionModel` base class defines the interface
   - Concrete implementations for each provider:
     - `OpenAIVisionModel` - OpenAI GPT-4 Vision
     - `AnthropicVisionModel` - Claude models
     - `GeminiVisionModel` - Google Gemini
     - `AzureOpenAIVisionModel` - Azure OpenAI
     - `AnthropicAWSBedrockVisionModel` - Claude via AWS Bedrock
   - `AutoDetectVisionModel()` - Auto-selects first available provider based on API keys
   - `create_default_vision_model()` - Factory function respecting `USE_VISION` env var

3. **VisionCapture** (`aicapture/vision_capture.py`)
   - Structured data extraction using customizable templates
   - Processes images/PDFs and returns structured JSON based on template schema
   - Used for extracting specific fields from documents (e.g., forms, technical diagrams)

4. **VidCapture** (`aicapture/vid_capture.py`)
   - Video frame extraction and analysis using OpenCV
   - Extracts frames at configurable rate (default: 2 fps)
   - Supports MP4, AVI, MOV, MKV formats
   - Resizes frames to target size (default: 768x768)
   - Analyzes frames with VLM using custom prompts

5. **Cache System** (`aicapture/cache.py`)
   - Two-layer caching: local file system + optional S3 cloud storage
   - `FileCache` - Local JSON-based cache (default: `tmp/.vision_parser_cache/`)
   - `S3Cache` - AWS S3-backed cache for distributed systems
   - `TwoLayerCache` - Combines both with automatic fallback
   - `ImageCache` - Specialized cache for image preprocessing results
   - Cache keys use SHA-256 hashing of inputs (file content + prompt)

6. **Settings** (`aicapture/settings.py`)
   - Centralized configuration using environment variables
   - Auto-loads from `.env` file via `python-dotenv`
   - Provider selection via `USE_VISION` (openai|claude|gemini|azure-openai|anthropic_bedrock)
   - Image quality settings: `ImageQuality.LOW_RES` (512x512) or `HIGH_RES` (768x2000)
   - Concurrency control: `MAX_CONCURRENT_TASKS` (default: 20)

### Key Design Patterns

- **Provider Abstraction**: All VLM providers implement the same `VisionModel` interface, making them interchangeable
- **Auto-Detection**: If `USE_VISION` is not set, the library auto-detects available providers by checking for API keys (order: Gemini → OpenAI → Azure → Anthropic)
- **Async-First**: Most operations support async/await for efficient concurrent processing
- **Caching Strategy**: Two-layer cache (local + cloud) reduces API calls and costs
- **Template-Based Extraction**: VisionCapture uses YAML-like templates to define expected data structure

## Development

### Setup

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv)
uv sync --all-extras
```

### Code Quality

```bash
# Format code with ruff
make format

# Run linters (ruff + mypy) - matches CI exactly
make lint

# Run all checks (format + lint + test)
make all
```

### Testing

```bash
# Run all tests with coverage
make test

# Run tests with pytest directly (more control)
uv run pytest -v --cov=aicapture --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_vision_parser_extended.py -v

# Run specific test function
uv run pytest tests/test_vision_parser_extended.py::test_specific_function -v
```

### Build & Publish

```bash
# Build package for distribution
make build
# Or: uv build

# Publish to PyPI
make publish
# Or: uv publish
```

### Code Style

- **Linting**: Ruff (replaces black, isort, flake8)
  - Line length: 120 characters
  - Python 3.10+ syntax
  - See `pyproject.toml` [tool.ruff] for rules
- **Type Checking**: MyPy with strict mode
  - All functions in `aicapture/` must have type annotations
  - Tests are exempt from strict typing (`disallow_untyped_defs = false`)
- **Formatting**: Ruff format (auto-fixes most issues)

### Testing Requirements

- All tests use `pytest` with async support (`pytest-asyncio`)
- Tests requiring API keys are controlled via environment variables
- CI runs `ruff check`, `mypy`, and `pytest` on every PR
- Coverage target: Tests must cover new functionality
- Test files follow pattern: `test_*.py` in `tests/` directory

### Important Development Notes

- **PyMuPDF Import**: Use `import fitz` (not `pymupdf`) for PDF processing
- **Circular Import Handling**: `vid_capture.py` imports directly from `vision_models` to avoid circular dependencies
- **Cache Invalidation**: Pass `invalidate_cache=True` to constructors to force fresh results (bypasses cache reads)
- **Image Quality**: Use `ImageQuality.HIGH_RES` for best OCR results, `LOW_RES` for faster processing
- **Video Duration Limit**: Default max 30 seconds per video (configurable via `VideoConfig.max_duration_seconds`)
- **Supported Image Formats**: JPG, JPEG, PNG, TIFF, WebP, BMP

### Release Process

**⚠️ IMPORTANT: Before committing version changes, remember to bump the Python version in `pyproject.toml`**

When preparing a new release:
1. Update version in `pyproject.toml` (both package version and Python version if applicable)
2. Run all tests: `make all`
3. Build the package: `make build`
4. Create commit with version bump
5. Tag the release
6. Publish to PyPI: `make publish`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Run `make all` to ensure code quality
4. Commit your changes (remember to update version in `pyproject.toml` if needed)
5. Push and open a Pull Request

## License

Copyright 2024 Aitomatic, Inc. Licensed under the Apache License, Version 2.0.

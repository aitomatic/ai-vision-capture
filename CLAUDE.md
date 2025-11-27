# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Vision Capture is a Python library for extracting and analyzing content from PDF, Image, and Video files using Vision Language Models (VLMs). It supports multiple VLM providers (OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI, AWS Bedrock) with a pluggable architecture.

**Package Name:** `aicapture`
**Python Version:** >=3.10
**Build System:** `hatchling` via `uv`

## Development Commands

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
- **Auto-Detection**: If `USE_VISION` is not set, the library auto-detects available providers by checking for API keys (order: Gemini ’ OpenAI ’ Azure ’ Anthropic)
- **Async-First**: Most operations support async/await for efficient concurrent processing
- **Caching Strategy**: Two-layer cache (local + cloud) reduces API calls and costs
- **Template-Based Extraction**: VisionCapture uses YAML-like templates to define expected data structure

## Environment Configuration

Required environment variables depend on the chosen provider. See `.env.template` for full reference.

**Auto-Detection (Recommended):**
```bash
# Just set any ONE API key - no need to set USE_VISION
GEMINI_API_KEY=your_key
# or OPENAI_API_KEY=your_key
# or ANTHROPIC_API_KEY=your_key
```

**Manual Provider Selection:**
```bash
USE_VISION=openai  # or: claude, gemini, azure-openai, anthropic_bedrock
OPENAI_API_KEY=your_key
```

**Performance Tuning:**
```bash
MAX_CONCURRENT_TASKS=5    # Concurrent API requests
VISION_PARSER_DPI=333     # PDF rendering quality
```

## Testing Requirements

- All tests use `pytest` with async support (`pytest-asyncio`)
- Tests requiring API keys are controlled via environment variables
- CI runs `ruff check`, `mypy`, and `pytest` on every PR
- Coverage target: Tests must cover new functionality
- Test files follow pattern: `test_*.py` in `tests/` directory

## Code Style

- **Linting**: Ruff (replaces black, isort, flake8)
  - Line length: 120 characters
  - Python 3.10+ syntax
  - See `pyproject.toml` [tool.ruff] for rules
- **Type Checking**: MyPy with strict mode
  - All functions in `aicapture/` must have type annotations
  - Tests are exempt from strict typing (`disallow_untyped_defs = false`)
- **Formatting**: Ruff format (auto-fixes most issues)

## Important Notes

- **PyMuPDF Import**: Use `import fitz` (not `pymupdf`) for PDF processing
- **Circular Import Handling**: `vid_capture.py` imports directly from `vision_models` to avoid circular dependencies
- **Cache Invalidation**: Pass `invalidate_cache=True` to constructors to force fresh results (bypasses cache reads)
- **Image Quality**: Use `ImageQuality.HIGH_RES` for best OCR results, `LOW_RES` for faster processing
- **Video Duration Limit**: Default max 30 seconds per video (configurable via `VideoConfig.max_duration_seconds`)
- **Supported Image Formats**: JPG, JPEG, PNG, TIFF, WebP, BMP

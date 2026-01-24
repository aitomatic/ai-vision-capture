# AI Vision Capture

A Python library for extracting and analyzing content from PDF, Image, and Video files using Vision Language Models (VLMs). Supports multiple providers including OpenAI, Anthropic Claude, Google Gemini, and Azure OpenAI.

## Features

- **Multi-Provider Support**: OpenAI, Claude, Gemini, Azure OpenAI, AWS Bedrock
- **Document Processing**: PDFs and images (JPG, PNG, TIFF, WebP, BMP)
- **Video Processing**: Frame extraction and analysis from MP4, AVI, MOV, MKV
- **Audio Transcription**: Whisper-powered speech-to-text with timestamps (LRC format)
- **Async Processing**: Configurable concurrency for batch operations
- **Two-Layer Caching**: Local file system + optional S3 cloud cache
- **Structured Output**: Template-based data extraction to JSON

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
export USE_VISION=openai            # Force specific provider
export MAX_CONCURRENT_TASKS=5       # Concurrent API requests
export VISION_PARSER_DPI=333        # PDF rendering quality
```

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
    frame_rate=2,
    enable_transcription=True,       # Enable Whisper transcription
    transcription_model="whisper-1", # OpenAI Whisper model
    transcription_language="en",     # Optional language hint (None for auto-detect)
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

## Development

```bash
# Setup
uv sync --all-extras

# Run checks
make test     # Tests with coverage
make lint     # Ruff + mypy
make format   # Auto-format
make all      # All of the above
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

## License

Copyright 2024 Aitomatic, Inc. Licensed under the Apache License, Version 2.0.

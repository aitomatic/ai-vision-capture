# Video Capture Examples

This directory contains examples demonstrating how to use the `VidCapture` module to extract and analyze frames from video files.

## Prerequisites

Before running these examples, make sure you have:

1. Installed the `vision_capture` package
2. Set up the necessary API keys for the vision models you want to use:
   - OpenAI: `OPENAI_API_KEY`
   - Anthropic: `ANTHROPIC_API_KEY`
   - Gemini: `GEMINI_API_KEY`
   - Azure OpenAI: `AZURE_OPENAI_API_KEY`

## Examples

### Basic Example

The `example.py` file demonstrates the basic usage of the `VidCapture` module:

```bash
python examples/vid_capture/example.py
```

This example:
- Extracts frames from a sample video at 2 frames per second
- Saves the extracted frames to disk
- Analyzes the frames with a simple prompt

### Comprehensive Example

The `comprehensive_example.py` file demonstrates more advanced features:

```bash
python examples/vid_capture/comprehensive_example.py
```

This example demonstrates:
1. Basic frame extraction with default settings
2. Custom configuration options
3. Error handling
4. Frame analysis with different prompts
5. Comparing results from different vision models

## Sample Videos

The examples use sample videos located in the `tests/sample/vids` directory:
- `rock.mp4`: A video of a rock formation
- `drop.mp4`: A video of a water drop

## Configuration Options

The `VideoConfig` class provides several options to customize video processing:

```python
config = VideoConfig(
    max_duration_seconds=30,  # Maximum video duration to process
    frame_rate=2,             # Frames per second to extract
    supported_formats=(".mp4", ".avi", ".mov", ".mkv"),  # Supported video formats
    target_frame_size=(768, 768),  # Target size for resized frames
    resize_frames=True        # Whether to resize frames
)
```

## Output

By default, the examples save extracted frames to the `tmp/output_frames` directory. You can modify this path in the example code if needed. 
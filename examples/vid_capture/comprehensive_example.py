"""
Comprehensive example demonstrating the video capture functionality.

This example shows:
1. Basic frame extraction
2. Custom configuration
3. Frame analysis with different prompts
4. Saving frames to disk
5. Error handling
"""

import asyncio
import os
from pathlib import Path
from typing import List

from PIL import Image

from vision_capture import VidCapture, VideoConfig, VideoValidationError
from vision_capture.vision_models import (
    AnthropicVisionModel,
    GeminiVisionModel,
    OpenAIVisionModel,
    VisionModelProvider,
)


async def analyze_frames_with_different_models(frames: List[Image.Image]):
    """Analyze frames with different vision models."""
    prompt = """
    Analyze these video frames and describe what is happening in the video.
    Be concise but detailed.
    """

    # Check if API keys are available
    results = {}

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("Analyzing with OpenAI...")
        openai_model = OpenAIVisionModel(api_key=openai_key)
        config = VideoConfig(frame_rate=1)
        vid_capture = VidCapture(config, vision_model=openai_model)
        results["openai"] = await vid_capture.capture_async(prompt, frames)

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("Analyzing with Anthropic Claude...")
        claude_model = AnthropicVisionModel(api_key=anthropic_key)
        config = VideoConfig(frame_rate=1)
        vid_capture = VidCapture(config, vision_model=claude_model)
        results["anthropic"] = await vid_capture.capture_async(prompt, frames)

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("Analyzing with Google Gemini...")
        gemini_model = GeminiVisionModel(api_key=gemini_key)
        config = VideoConfig(frame_rate=1)
        vid_capture = VidCapture(config, vision_model=gemini_model)
        results["gemini"] = await vid_capture.capture_async(prompt, frames)

    return results


def save_frames(frames: List[Image.Image], output_dir: Path):
    """Save frames to disk."""
    output_dir.mkdir(exist_ok=True, parents=True)
    for i, frame in enumerate(frames):
        frame.save(output_dir / f"frame_{i:03d}.jpg")
    print(f"Saved {len(frames)} frames to {output_dir}")


async def run_comprehensive_example():
    """Run a comprehensive example of video capture functionality."""
    # Define video paths
    video_dir = Path("tests/sample/vids")
    rock_video = video_dir / "rock.mp4"
    drop_video = video_dir / "drop.mp4"
    output_dir = Path("tmp/output_frames")

    print(f"Using videos from {video_dir}")

    # Example 1: Basic frame extraction with default settings
    print("\n=== Example 1: Basic Frame Extraction ===")
    default_config = VideoConfig()
    default_capture = VidCapture(default_config)

    try:
        frames, interval = default_capture.extract_frames(str(rock_video))
        print(
            f"Successfully extracted {len(frames)} frames at {interval:.2f}s intervals"
        )
        save_frames(frames, output_dir / "default")
    except Exception as e:
        print(f"Error in Example 1: {e}")

    # Example 2: Custom configuration
    print("\n=== Example 2: Custom Configuration ===")
    custom_config = VideoConfig(
        frame_rate=4,  # 4 frames per second
        max_duration_seconds=10,  # Only process first 10 seconds
        target_frame_size=(512, 512),  # Smaller frame size
    )
    custom_capture = VidCapture(custom_config)

    try:
        frames, interval = custom_capture.extract_frames(str(rock_video))
        print(
            f"Successfully extracted {len(frames)} frames at {interval:.2f}s intervals"
        )
        save_frames(frames, output_dir / "custom")
    except Exception as e:
        print(f"Error in Example 2: {e}")

    # Example 3: Error handling
    print("\n=== Example 3: Error Handling ===")
    try:
        # Try to process a non-existent video
        frames, _ = default_capture.extract_frames("nonexistent.mp4")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")

    try:
        # Create a text file and try to process it
        text_file = Path("tmp/not_a_video.txt")
        text_file.parent.mkdir(exist_ok=True, parents=True)
        text_file.write_text("This is not a video file")

        frames, _ = default_capture.extract_frames(str(text_file))
    except VideoValidationError as e:
        print(f"Expected error: {e}")
        text_file.unlink(missing_ok=True)

    # Example 4: Frame analysis with default model
    print("\n=== Example 4: Frame Analysis ===")
    try:
        # Extract frames from a different video
        frames, _ = default_capture.extract_frames(str(drop_video))
        save_frames(frames, output_dir / "drop")

        # Analyze with different prompts
        prompts = [
            "Describe what is happening in this video in one sentence.",
            """
            Analyze these video frames and describe:
            1. What is happening in the video
            2. Key objects and people visible
            3. Any notable actions or events
            """,
            """
            You are a technical video analyst. Provide a detailed technical analysis
            of this video, including camera angles, lighting, and visual composition.
            """,
        ]

        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}:")
            print(prompt)
            result = default_capture.capture(prompt, frames)
            print("\nResult:")
            print(result)
    except Exception as e:
        print(f"Error in Example 4: {e}")

    # Example 5: Compare different models (if API keys are available)
    print("\n=== Example 5: Compare Different Models ===")
    try:
        # Use the frames from Example 4
        results = await analyze_frames_with_different_models(frames)

        for model, result in results.items():
            print(f"\n{model.upper()} RESULT:")
            print(result)
    except Exception as e:
        print(f"Error in Example 5: {e}")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_example())

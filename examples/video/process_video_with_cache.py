#!/usr/bin/env python
"""
Example of using the VidCapture module with caching enabled.
"""

import asyncio
import time
from pathlib import Path

from aicapture import AnthropicVisionModel
from aicapture.utils import get_default_bucket
from aicapture.vid_capture import VidCapture, VideoConfig


async def main():
    # Set up the model
    vision_model = AnthropicVisionModel(
        model="claude-3-5-sonnet-20240620",
        temperature=0.0,
    )

    # Set up the video capture with caching enabled
    vid_capture = VidCapture(
        config=VideoConfig(
            frame_rate=1,  # Extract 1 frame per second
            max_duration_seconds=60,  # Maximum video duration (seconds)
            cache_dir="tmp/.video_cache",  # Cache directory
            cloud_bucket=get_default_bucket(),
        ),
        vision_model=vision_model,
        invalidate_cache=False,  # Set to True to bypass cache
    )

    # Example video path - replace with your own video file
    video_path = "tmp/vid/TestVid.mp4"

    # Check if the video exists
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file at the specified path or update the path.")
        return

    # Prompt for the vision model
    prompt = "Describe what's happening in this video clip in detail."

    # First run - should process video and store in cache
    print("\n=== First Run (Caching) ===")
    start_time = time.time()
    result1 = await vid_capture.process_video_async(video_path, prompt)
    elapsed1 = time.time() - start_time
    print(f"First run took {elapsed1:.2f} seconds")
    print(f"Result: {result1[:100]}...")  # Show first 100 chars

    # Second run - should fetch from cache
    print("\n=== Second Run (Cache Hit) ===")
    start_time = time.time()
    result2 = await vid_capture.process_video_async(video_path, prompt)
    elapsed2 = time.time() - start_time
    print(f"Second run took {elapsed2:.2f} seconds")
    print(f"Result: {result2[:100]}...")  # Show first 100 chars

    print(f"\nCache speedup: {elapsed1/elapsed2:.1f}x faster")

    # Different prompt - should miss cache
    print("\n=== Third Run (Different Prompt) ===")
    different_prompt = "List all objects visible in this video."
    start_time = time.time()
    result3 = await vid_capture.process_video_async(video_path, different_prompt)
    elapsed3 = time.time() - start_time
    print(f"Third run took {elapsed3:.2f} seconds")
    print(f"Result: {result3[:100]}...")  # Show first 100 chars


if __name__ == "__main__":
    asyncio.run(main())

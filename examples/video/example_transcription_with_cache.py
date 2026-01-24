#!/usr/bin/env python
"""
Example demonstrating transcription caching with async processing.

This example shows how transcriptions are automatically cached to save time and cost
on repeated processing of the same video.

The cache key is based on:
- Video file hash (SHA-256)
- Transcription model name (e.g., "whisper-1")
- Language setting (e.g., "en" or "auto")

This means:
- Same video + same model + same language = cache hit (instant, no API call)
- Different video OR different model OR different language = cache miss (API call)
"""

import asyncio
from pathlib import Path

from aicapture import VidCapture, VideoConfig


async def main_async():
    """Demonstrate transcription caching with async processing."""

    vid_file = "tests/sample/vids/Maple_Baked_Salmon_Cooking_Tutorial.mp4"

    if not Path(vid_file).exists():
        print(f"Error: Video file not found at {vid_file}")
        return

    # Configure with transcription enabled
    config = VideoConfig(
        frame_rate=2,
        max_duration_seconds=600,
        chunk_duration_seconds=60,
        enable_transcription=True,
        transcription_model="whisper-1",
        transcription_language="en",
        # Transcriptions are cached in: tmp/.vid_capture_cache/transcriptions/
        cache_dir="tmp/.vid_capture_cache",
    )

    print("Processing video with transcription (async)...")
    print("First run: Will call Whisper API and cache the result")
    print("Subsequent runs: Will load transcription from cache (instant)\n")

    video_capture = VidCapture(config=config)

    prompt = """
    Analyze this cooking tutorial and provide:
    1. Recipe name and main ingredients
    2. Key cooking steps
    3. Any tips mentioned by the instructor
    """

    try:
        result = await video_capture.process_video_async(vid_file, prompt)

        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print(result)
        print("=" * 80)

        print("\n✓ Transcription has been cached!")
        print("  Run this script again to see instant transcription loading from cache.")

        # Show cache location
        cache_dir = Path(config.cache_dir) / "transcriptions"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            print(f"\nCache location: {cache_dir}")
            print(f"Cached transcriptions: {len(cache_files)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

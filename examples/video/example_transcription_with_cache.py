#!/usr/bin/env python
"""
Example demonstrating transcription caching.

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

from pathlib import Path

from aicapture import VidCapture, VideoConfig


def main():
    """Demonstrate transcription caching."""

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

    print("Processing video with transcription...")
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
        result = video_capture.process_video(vid_file, prompt)

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


if __name__ == "__main__":
    main()

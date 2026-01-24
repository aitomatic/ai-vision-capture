#!/usr/bin/env python
"""
Simple example of video processing with audio transcription.
Similar to example.py but with transcription enabled.
"""

from aicapture import VidCapture, VideoConfig


def run_example():
    """Process video with audio transcription enabled."""
    vid_file = "tests/sample/vids/Maple_Baked_Salmon_Cooking_Tutorial.mp4"

    # Create video capture with transcription enabled
    config = VideoConfig(
        frame_rate=2,
        enable_transcription=True,  # Enable Whisper transcription
        transcription_model="whisper-1",  # OpenAI Whisper model
        transcription_language="en",  # Optional: specify language
    )
    video_capture = VidCapture(config)

    # Process video with a simple prompt
    try:
        prompt = "Describe the cooking tutorial and summarize the recipe steps."

        result = video_capture.process_video(vid_file, prompt)
        print("\nAnalysis Result:")
        print(result)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_example()

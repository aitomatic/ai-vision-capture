#!/usr/bin/env python
"""
Example of video processing with audio transcription using OpenAI Whisper.

This example demonstrates how to:
1. Extract frames from a video
2. Transcribe audio using OpenAI Whisper
3. Combine visual and audio context for rich video analysis
"""

from pathlib import Path

from aicapture import VidCapture, VideoConfig


def run_example():
    """Process a cooking tutorial video with audio transcription."""
    # Path to the sample video
    vid_file = "tests/sample/vids/Maple_Baked_Salmon_Cooking_Tutorial.mp4"

    # Verify video file exists
    if not Path(vid_file).exists():
        print(f"Error: Video file not found at {vid_file}")
        return

    # Configure video processing with transcription enabled
    config = VideoConfig(
        frame_rate=2,  # Extract 2 frames per second
        max_duration_seconds=600,  # Process up to 10 minutes (default is 300s/5min)
        chunk_duration_seconds=60,  # Process in 60-second chunks for long videos
        enable_transcription=True,  # Enable Whisper transcription
        transcription_model="whisper-1",  # OpenAI Whisper model
        transcription_language="en",  # English (or None for auto-detect)
    )

    # Create video capture instance
    video_capture = VidCapture(config=config)

    print(f"Processing video: {vid_file}")
    print("This will extract frames and transcribe audio...\n")

    try:
        # Process video with a comprehensive prompt
        prompt = """
        Analyze this cooking tutorial video and provide:
        1. A summary of the recipe being demonstrated
        2. Key ingredients mentioned or shown
        3. Main cooking steps and techniques
        4. Any tips or important notes from the instructor
        5. Estimated cooking time and difficulty level
        """

        result = video_capture.process_video(vid_file, prompt)

        print("=" * 80)
        print("VIDEO ANALYSIS RESULT")
        print("=" * 80)
        print(result)
        print("=" * 80)

    except Exception as e:
        print(f"Error processing video: {e}")

        # Check if it's an OpenAI quota error
        if "insufficient_quota" in str(e) or "429" in str(e):
            print("\n" + "=" * 80)
            print("OPENAI QUOTA ERROR")
            print("=" * 80)
            print("Your OpenAI account has insufficient quota/credits.")
            print("\nTo fix this:")
            print("1. Visit https://platform.openai.com/account/billing")
            print("2. Add credits or update your payment method")
            print("3. Or disable transcription: set enable_transcription=False")
            print("=" * 80)

        import traceback

        traceback.print_exc()


def run_transcription_only_example():
    """Example of using the transcriber directly without video analysis."""
    from aicapture import OpenAIAudioTranscriber

    vid_file = "tests/sample/vids/Maple_Baked_Salmon_Cooking_Tutorial.mp4"

    if not Path(vid_file).exists():
        print(f"Error: Video file not found at {vid_file}")
        return

    print("Transcribing audio only...\n")

    try:
        # Create transcriber instance
        transcriber = OpenAIAudioTranscriber(model="whisper-1", language="en")

        # Transcribe the video
        transcription = transcriber.transcribe_video(vid_file)

        print("=" * 80)
        print("TRANSCRIPTION (LRC Format with Timestamps)")
        print("=" * 80)
        print(transcription.to_lrc())
        print("=" * 80)

        print("\n" + "=" * 80)
        print("FULL TEXT (No Timestamps)")
        print("=" * 80)
        print(transcription.full_text)
        print("=" * 80)

        print(f"\nTotal segments: {len(transcription.segments)}")

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Full video analysis with transcription (default)")
    print("2. Audio transcription only")

    choice = input("\nEnter choice (1 or 2): ").strip() or "1"

    if choice == "2":
        run_transcription_only_example()
    else:
        run_example()

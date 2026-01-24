"""
Generate a test video with TTS narration for testing audio transcription.

This script creates a short instructional video with known speech content,
allowing deterministic testing of the Whisper transcription feature.

Usage:
    python tests/generate_test_video.py

Output:
    tests/sample/vids/instruction_narrated.mp4
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from gtts import gTTS

# The known script - each entry is (text, duration_hint_seconds)
# These are the sentences we expect Whisper to transcribe
SCRIPT_SEGMENTS = [
    "Step one. Open the application and navigate to the settings menu.",
    "Step two. Click the configuration button to access advanced options.",
    "Step three. Enable the dark mode toggle to switch the display theme.",
    "Step four. Save your changes and restart the application.",
]

FULL_SCRIPT = " ".join(SCRIPT_SEGMENTS)

OUTPUT_DIR = Path(__file__).parent / "sample" / "vids"
OUTPUT_FILE = OUTPUT_DIR / "instruction_narrated.mp4"


def generate_audio(text: str, output_path: str) -> None:
    """Generate TTS audio from text using gTTS."""
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_path)
    print(f"Generated audio: {output_path}")


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file using moviepy."""
    from moviepy import AudioFileClip

    clip = AudioFileClip(audio_path)
    duration = clip.duration
    clip.close()
    return duration


def create_video_with_audio(audio_path: str, output_path: str) -> None:
    """Create a video with text slides synchronized to the audio."""
    from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, TextClip

    # Load audio
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    # Create a simple background
    bg_clip = ColorClip(size=(640, 480), color=(30, 30, 30)).with_duration(duration)

    # Create title text
    try:
        title_clip = (
            TextClip(
                text="Instructional Video Test",
                font_size=32,
                color="white",
                font="DejaVu-Sans",
            )
            .with_position(("center", 50))
            .with_duration(duration)
        )

        # Create subtitle showing current step
        segment_duration = duration / len(SCRIPT_SEGMENTS)
        subtitle_clips = []
        for i, segment in enumerate(SCRIPT_SEGMENTS):
            start_time = i * segment_duration
            sub_clip = (
                TextClip(
                    text=segment,
                    font_size=20,
                    color="yellow",
                    font="DejaVu-Sans",
                    method="caption",
                    size=(580, None),
                )
                .with_position(("center", 300))
                .with_start(start_time)
                .with_duration(segment_duration)
            )
            subtitle_clips.append(sub_clip)

        # Compose video
        video = CompositeVideoClip([bg_clip, title_clip] + subtitle_clips)
    except Exception as e:
        print(f"TextClip failed ({e}), falling back to plain background video")
        video = bg_clip

    # Add audio
    video = video.with_audio(audio_clip)

    # Write output
    ffmpeg_path = None
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    write_kwargs = {
        "codec": "libx264",
        "audio_codec": "aac",
        "fps": 24,
        "logger": None,
    }
    if ffmpeg_path:
        write_kwargs["ffmpeg_binary"] = ffmpeg_path

    video.write_videofile(output_path, **write_kwargs)

    # Cleanup
    audio_clip.close()
    video.close()
    print(f"Generated video: {output_path}")


def create_simple_video_with_audio(audio_path: str, output_path: str) -> None:
    """Fallback: create video using opencv + ffmpeg merge."""
    import subprocess

    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Get audio duration
    duration = get_audio_duration(audio_path)
    fps = 24
    total_frames = int(duration * fps)

    # Create temporary video without audio
    temp_video = tempfile.mktemp(suffix=".mp4")

    # Create video frames with opencv
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (640, 480))

    segment_duration = duration / len(SCRIPT_SEGMENTS)

    for frame_idx in range(total_frames):
        # Create frame with dark background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)

        # Determine current segment
        current_time = frame_idx / fps
        segment_idx = min(int(current_time / segment_duration), len(SCRIPT_SEGMENTS) - 1)

        # Add step number
        step_text = f"Step {segment_idx + 1} of {len(SCRIPT_SEGMENTS)}"
        cv2.putText(frame, step_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Add segment text (wrapped)
        segment_text = SCRIPT_SEGMENTS[segment_idx]
        words = segment_text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) > 45:
                lines.append(current_line)
                current_line = word
            else:
                current_line = (current_line + " " + word).strip()
        if current_line:
            lines.append(current_line)

        for i, line in enumerate(lines):
            y_pos = 200 + i * 40
            cv2.putText(frame, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        writer.write(frame)

    writer.release()

    # Merge video and audio using ffmpeg
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        temp_video,
        "-i",
        audio_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    # Cleanup temp file
    os.unlink(temp_video)
    print(f"Generated video: {output_path}")


def main() -> None:
    """Generate the test video."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate TTS audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name

    try:
        print(f"Generating TTS audio from script ({len(FULL_SCRIPT)} chars)...")
        generate_audio(FULL_SCRIPT, audio_path)

        duration = get_audio_duration(audio_path)
        print(f"Audio duration: {duration:.1f}s")

        # Create video with audio
        print("Creating video with narration...")
        try:
            create_video_with_audio(audio_path, str(OUTPUT_FILE))
        except Exception as e:
            print(f"MoviePy approach failed ({e}), trying fallback...")
            create_simple_video_with_audio(audio_path, str(OUTPUT_FILE))

        print(f"\nDone! Test video saved to: {OUTPUT_FILE}")
        print(f"Known transcript: {FULL_SCRIPT}")

    finally:
        # Cleanup temp audio
        if os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    main()

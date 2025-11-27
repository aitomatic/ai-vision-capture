from pathlib import Path

from aicapture import VidCapture, VideoConfig


def run_example():  # Example usage
    vid_file = "tests/sample/vids/rock.mp4"

    # Create video capture with 2 fps
    config = VideoConfig(frame_rate=2)
    video_capture = VidCapture(config)

    # Extract frames
    try:
        frames, interval = video_capture.extract_frames(vid_file)
        print(f"Successfully extracted {len(frames)} frames at {interval:.2f}s intervals")

        # Save frames as example (optional)
        output_dir = Path("tmp/output_frames")
        output_dir.mkdir(exist_ok=True, parents=True)

        for i, frame in enumerate(frames):
            frame.save(output_dir / f"frame_{i:03d}.jpg")

        # Example of analyzing the frames
        prompt = """
        Analyze these video frames and describe:
        1. What is happening in the video
        2. Key objects and people visible
        3. Any notable actions or events
        """

        prompt = "Describe the content of the video?"

        result = video_capture.capture(prompt, frames)
        print("\nAnalysis Result:")
        print(result)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_example()

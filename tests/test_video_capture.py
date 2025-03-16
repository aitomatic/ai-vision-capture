# from pathlib import Path

# import pytest
# from PIL import Image

# from vision_capture import VidCapture, VideoConfig, VideoValidationError

# # Define test video paths
# TEST_VIDEO_PATH = Path(__file__).parent / "sample" / "vids" / "rock.mp4"
# NONEXISTENT_VIDEO_PATH = Path(__file__).parent / "sample" / "vids" / "nonexistent.mp4"
# TEST_OUTPUT_DIR = Path(__file__).parent / ".cache" / "video_frames"


# @pytest.fixture
# def video_capture():
#     """Create a VidCapture instance with default configuration."""
#     config = VideoConfig(frame_rate=1)  # 1 fps for faster tests
#     return VidCapture(config)


# def test_video_config_defaults():
#     """Test that VideoConfig has expected defaults."""
#     config = VideoConfig()
#     assert config.max_duration_seconds == 30
#     assert config.frame_rate == 2
#     assert config.supported_formats == (".mp4", ".avi", ".mov", ".mkv")
#     assert config.target_frame_size == (768, 768)
#     assert config.resize_frames is True


# def test_extract_frames(video_capture):
#     """Test extracting frames from a video file."""
#     frames, interval = video_capture.extract_frames(str(TEST_VIDEO_PATH))

#     # Check that frames were extracted
#     assert len(frames) > 0
#     assert all(isinstance(frame, Image.Image) for frame in frames)

#     # Check frame interval
#     assert interval == pytest.approx(1.0, 0.1)  # Should be close to 1 second

#     # Save frames for visual inspection (optional)
#     TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
#     for i, frame in enumerate(frames):
#         frame.save(TEST_OUTPUT_DIR / f"test_frame_{i:03d}.jpg")


# def test_nonexistent_video(video_capture):
#     """Test that extracting frames from a nonexistent video raises FileNotFoundError."""
#     with pytest.raises(FileNotFoundError):
#         video_capture.extract_frames(str(NONEXISTENT_VIDEO_PATH))


# def test_unsupported_format(video_capture):
#     """Test that extracting frames from an unsupported format raises VideoValidationError."""
#     # Create a temporary text file
#     temp_file = Path(__file__).parent / ".cache" / "not_a_video.txt"
#     temp_file.parent.mkdir(exist_ok=True, parents=True)
#     temp_file.write_text("This is not a video file")

#     with pytest.raises(VideoValidationError):
#         video_capture.extract_frames(str(temp_file))

#     # Clean up
#     temp_file.unlink(missing_ok=True)


# def test_capture_with_frames(video_capture):
#     """Test capturing knowledge from video frames."""
#     frames, _ = video_capture.extract_frames(str(TEST_VIDEO_PATH))

#     # Use a simple prompt for testing
#     prompt = "Describe what you see in these frames in one sentence."
#     result = video_capture.capture(prompt, frames)

#     # Check that we got a non-empty result
#     assert result
#     assert isinstance(result, str)
#     assert len(result) > 10  # Should have some meaningful content


# def test_process_video(video_capture):
#     """Test processing a video file directly."""
#     prompt = "Describe what you see in these frames in one sentence."
#     result = video_capture.process_video(str(TEST_VIDEO_PATH), prompt)

#     # Check that we got a non-empty result
#     assert result
#     assert isinstance(result, str)
#     assert len(result) > 10  # Should have some meaningful content

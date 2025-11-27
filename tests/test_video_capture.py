from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pytest import MonkeyPatch

from aicapture import VidCapture, VideoConfig
from aicapture.vid_capture import VideoValidationError

# Define test video paths
TEST_VIDEO_PATH = Path(__file__).parent / "sample" / "vids" / "rock.mp4"
TEST_VIDEO_PATH_2 = Path(__file__).parent / "sample" / "vids" / "drop.mp4"


@pytest.fixture
def video_config() -> VideoConfig:
    """Create a default VideoConfig for testing."""
    return VideoConfig(frame_rate=2)


@pytest.fixture
def vid_capture(video_config: VideoConfig) -> VidCapture:
    """Create a VidCapture instance with test configuration."""
    return VidCapture(config=video_config)


def test_init_default() -> None:
    """Test initialization with default parameters."""
    vid_capture = VidCapture()
    assert vid_capture.config is not None
    assert vid_capture.vision_model is not None


def test_init_custom_config(video_config: VideoConfig) -> None:
    """Test initialization with custom config."""
    vid_capture = VidCapture(config=video_config)
    assert vid_capture.config == video_config
    assert vid_capture.config.frame_rate == 2


def test_extract_frames(vid_capture: VidCapture) -> None:
    """Test extracting frames from a video file."""
    # Ensure the test video exists
    assert TEST_VIDEO_PATH.exists(), f"Test video not found at {TEST_VIDEO_PATH}"

    # Extract frames
    frames, interval = vid_capture.extract_frames(str(TEST_VIDEO_PATH))

    # Validate results
    assert len(frames) > 0, "No frames were extracted"
    assert all(isinstance(frame, Image.Image) for frame in frames), "Frames should be PIL Images"
    assert interval > 0, "Frame interval should be positive"
    assert interval == 1.0 / vid_capture.config.frame_rate, "Interval should match config"


def test_extract_frames_custom_rate() -> None:
    """Test extracting frames with custom frame rate."""
    custom_config = VideoConfig(frame_rate=4)  # 4 fps
    vid_capture = VidCapture(config=custom_config)

    frames, interval = vid_capture.extract_frames(str(TEST_VIDEO_PATH))

    # With higher frame rate, we should get more frames
    assert len(frames) > 0
    assert interval == 0.25, "Interval should be 0.25s for 4 fps"


def test_video_validation_error() -> None:
    """Test validation errors for invalid videos."""
    vid_capture = VidCapture()

    # Test non-existent file
    with pytest.raises(VideoValidationError):
        vid_capture.extract_frames("nonexistent.mp4")

    # Test unsupported format
    with pytest.raises(VideoValidationError):
        vid_capture.extract_frames("test.txt")


def test_optimize_frame(vid_capture: VidCapture) -> None:
    """Test frame optimization."""
    import numpy as np

    # Create a test frame
    test_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    test_frame[:, :] = (0, 0, 255)  # BGR format (red in OpenCV)

    # Optimize the frame
    optimized = vid_capture._optimize_frame(test_frame)

    # Check that it's a PIL Image
    assert isinstance(optimized, Image.Image)

    # Check that it's been resized according to config
    if vid_capture.config.resize_frames:
        assert max(optimized.size) <= max(vid_capture.config.target_frame_size)


def test_process_video(vid_capture: VidCapture, monkeypatch: MonkeyPatch) -> None:
    """Test the process_video method."""
    # Mock the capture method to avoid actual API calls
    mock_result = "This video shows a rock formation."

    def mock_capture(*args: Any, **kwargs: Any) -> str:
        return mock_result

    monkeypatch.setattr(vid_capture, "capture", mock_capture)

    # Process the video
    result = vid_capture.process_video(str(TEST_VIDEO_PATH), "Describe the video")

    # Check the result
    assert result == mock_result


@pytest.mark.asyncio
async def test_capture_async(vid_capture: VidCapture, monkeypatch: MonkeyPatch) -> None:
    """Test the capture_async method."""
    # Create a few test frames
    test_frames = [Image.new("RGB", (100, 100), color=(73, 109, 137)) for _ in range(3)]

    # Mock the vision model's method
    mock_result = "Test result from vision model"

    async def mock_process_image_async(*args: Any, **kwargs: Any) -> str:
        return mock_result

    monkeypatch.setattr(vid_capture.vision_model, "process_image_async", mock_process_image_async)

    # Call capture_async
    result = await vid_capture.capture_async("Describe these images", test_frames)

    # Check the result
    assert result == mock_result


def test_capture(vid_capture: VidCapture, monkeypatch: MonkeyPatch) -> None:
    """Test the synchronous capture method."""
    # Create a few test frames
    test_frames = [Image.new("RGB", (100, 100), color=(73, 109, 137)) for _ in range(3)]

    # Mock the capture_async method
    async def mock_capture_async(*args: Any, **kwargs: Any) -> str:
        return "Test result from vision model"

    monkeypatch.setattr(vid_capture, "capture_async", mock_capture_async)

    # Call capture
    result = vid_capture.capture("Describe these images", test_frames)

    # Check the result
    assert result == "Test result from vision model"


def test_empty_frames(vid_capture: VidCapture) -> None:
    """Test handling of empty frames list."""
    with pytest.raises(ValueError):
        vid_capture.capture("Describe these images", [])


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

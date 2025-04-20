"""
Simple video capture module for extracting frames from videos.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Fix circular import by importing directly from vision_models
from aicapture.vision_models import VisionModel, create_default_vision_model


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    max_duration_seconds: int = 30
    frame_rate: int = 2  # Frames per second to extract
    supported_formats: tuple = (".mp4", ".avi", ".mov", ".mkv")
    target_frame_size: tuple = (768, 768)  # Target size for resized frames
    resize_frames: bool = True


class VideoValidationError(Exception):
    """Raised when video validation fails."""

    pass


class VidCapture:
    """
    Simple utility for extracting frames from video files.
    """

    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        vision_model: Optional[VisionModel] = None,
    ):
        """
        Initialize VideoCapture with configuration.

        Args:
            config: Configuration for video processing
            vision_model: Vision model for image analysis (created if None)
        """
        self.config = config or VideoConfig()
        self.vision_model = vision_model or create_default_vision_model()

    def _validate_video(self, video_path: str) -> None:
        """
        Validate video file format and duration.

        Args:
            video_path: Path to video file

        Raises:
            VideoValidationError: If validation fails
        """
        if not any(
            video_path.lower().endswith(fmt) for fmt in self.config.supported_formats
        ):
            raise VideoValidationError(
                f"Unsupported video format. Supported formats: "
                f"{self.config.supported_formats}"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoValidationError("Failed to open video file")

        # Check duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        if duration > self.config.max_duration_seconds:
            raise VideoValidationError(
                f"Video duration ({duration:.1f}s) exceeds maximum allowed "
                f"({self.config.max_duration_seconds}s)"
            )

        cap.release()

    def _optimize_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Optimize video frame for processing.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            PIL Image optimized for processing
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Resize if needed while maintaining aspect ratio
        if self.config.resize_frames:
            width, height = image.size
            if (
                width > self.config.target_frame_size[0]
                or height > self.config.target_frame_size[1]
            ):
                scale = min(
                    self.config.target_frame_size[0] / width,
                    self.config.target_frame_size[1] / height,
                )
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def extract_frames(self, video_path: str) -> Tuple[List[Image.Image], float]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (list of frames, frame interval in seconds)
        """
        # Validate the video first
        self._validate_video(video_path)

        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Calculate frame interval based on desired frame rate
        frame_interval = 1.0 / self.config.frame_rate
        frames = []

        # Calculate how many frames to extract
        num_frames_to_extract = min(
            int(duration * self.config.frame_rate),
            int(self.config.max_duration_seconds * self.config.frame_rate),
        )

        print(
            f"Extracting {num_frames_to_extract} frames "
            f"at {self.config.frame_rate} fps "
            f"from video with duration {duration:.1f}s"
        )

        for frame_idx in range(num_frames_to_extract):
            # Calculate the frame position
            frame_position = int(frame_idx * frame_interval * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

            ret, frame = cap.read()
            if not ret:
                break

            # Optimize and store frame
            pil_frame = self._optimize_frame(frame)
            frames.append(pil_frame)

        cap.release()
        return frames, frame_interval

    async def capture_async(
        self, prompt: str, images: List[Image.Image], **kwargs: Any
    ) -> str:
        """
        Extract knowledge from a list of images using a vision model.

        Args:
            prompt: Instruction prompt for the vision model
            images: List of images to analyze
            **kwargs: Additional parameters to pass to the vision model

        Returns:
            String containing the extracted knowledge
        """
        if not images:
            raise ValueError("No images provided for analysis")

        print(f"Analyzing {len(images)} images with vision model")

        # Process the images with the vision model
        result = await self.vision_model.process_image_async(
            image=images, prompt=prompt, **kwargs
        )

        return result

    def capture(self, prompt: str, images: List[Image.Image], **kwargs: Any) -> str:
        """
        Synchronous wrapper for capture_async.

        Args:
            prompt: Instruction prompt for the vision model
            images: List of images to analyze
            **kwargs: Additional parameters to pass to the vision model

        Returns:
            String containing the extracted knowledge
        """
        return asyncio.run(self.capture_async(prompt, images, **kwargs))

    def process_video(self, video_path: str, prompt: str, **kwargs: Any) -> str:
        """
        Extract frames from a video and analyze them with a vision model.

        Args:
            video_path: Path to the video file
            prompt: Instruction prompt for the vision model
            **kwargs: Additional parameters to pass to the vision model

        Returns:
            String containing the extracted knowledge from the video frames
        """
        # Extract frames from the video
        frames, _ = self.extract_frames(video_path)

        if not frames:
            raise ValueError(f"No frames could be extracted from {video_path}")

        return self.capture(prompt, frames, **kwargs)

    async def process_video_async(
        self, video_path: str, prompt: str, **kwargs: Any
    ) -> str:
        """
        Asynchronous wrapper for process_video.
        """
        frames, _ = self.extract_frames(video_path)

        if not frames:
            raise ValueError(f"No frames could be extracted from {video_path}")

        return await self.capture_async(prompt, frames, **kwargs)

    @classmethod
    def analyze_video(cls, video_path: str) -> dict:
        """
        Analyze a video and return video metadata.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video metadata (resolution, duration, fps, etc.)
            or error information if analysis fails.
        """
        # Check if file exists
        if not Path(video_path).exists():
            return {
                "status": "error",
                "message": f"Video file not found: {video_path}",
            }

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoValidationError("Failed to open video file")

            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Convert codec integer to string representation
            codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

            return {
                "status": "success",
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "codec": codec,
            }
        except VideoValidationError as e:
            return {
                "status": "error",
                "message": str(e),
            }
        except cv2.error as e:
            return {
                "status": "error",
                "message": f"OpenCV error: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
            }
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

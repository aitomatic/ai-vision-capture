# flake8: noqa: E501

"""
Video capture module for processing and analyzing video content using vision models.
Extends the base vision capture functionality with video-specific features.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from vision_capture.cache import FileCache, HashUtils, TwoLayerCache
from vision_capture.vision_models import VisionModel
from vision_capture.vision_parser import VisionParser

DEFAULT_VIDEO_PROMPT = """
Analyze this sequence of video frames and provide a comprehensive analysis of the video content:

1. Overall Scene Description:
   - Describe the main setting and environment
   - Identify key subjects/actors in the video
   - Note any significant changes in scene or setting

2. Temporal Analysis:
   - Describe the sequence of events in chronological order
   - Note any significant actions or movements
   - Identify any patterns or repetitive elements

3. Key Objects and Elements:
   - List important objects and their roles in the scene
   - Note any text overlays or visual indicators
   - Describe any relevant technical equipment or tools

4. Technical Observations:
   - Note any camera movements or angle changes
   - Identify lighting conditions and changes
   - Comment on video quality or notable visual characteristics

5. Context and Purpose:
   - Infer the likely purpose or context of the video
   - Note any educational, instructional, or documentary elements
   - Identify the target audience if apparent

Provide the analysis in clear, concise language focusing on the most relevant details.
"""


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    max_duration_seconds: int = 30
    frame_rate: int = 1  # Frames per second to extract
    max_frames: int = 30
    batch_size: int = 30  # Process all frames in one batch by default
    min_confidence: float = 0.7
    supported_formats: tuple = (".mp4", ".avi", ".mov", ".mkv")
    target_frame_size: tuple = (768, 768)  # Aligned with example
    frame_stride: int = 50  # Take every Nth frame for batch processing
    resize_frames: bool = True
    prompt: str = DEFAULT_VIDEO_PROMPT


class VideoValidationError(Exception):
    """Raised when video validation fails."""

    pass


class VidCapture(VisionParser):
    """
    Parser for extracting and analyzing content from video files.
    Extends the base VisionParser with video-specific functionality.
    """

    def __init__(
        self,
        vision_model: Optional[VisionModel] = None,
        cache_dir: Optional[str] = None,
        config: Optional[VideoConfig] = None,
        **kwargs: Any,
    ):
        """Initialize VideoParser with configuration."""
        super().__init__(vision_model=vision_model, cache_dir=cache_dir, **kwargs)
        self.config = config or VideoConfig()
        self._frame_cache = TwoLayerCache(
            file_cache=FileCache(cache_dir), s3_cache=None
        )

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
                f"Unsupported video format. Supported formats: {self.config.supported_formats}"
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
                f"Video duration ({duration:.1f}s) exceeds maximum allowed ({self.config.max_duration_seconds}s)"
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


    async def _extract_frames(
        self, video_path: str, file_hash: str
    ) -> Tuple[List[Image.Image], float]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to video file
            file_hash: Hash of video file for caching

        Returns:
            Tuple of (list of frames, frame interval in seconds)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / self.config.frame_rate
        frames = []

        frame_count = 0
        while cap.isOpened() and frame_count < self.config.max_frames:
            # Calculate the frame position
            frame_position = int(frame_count * frame_interval * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

            ret, frame = cap.read()
            if not ret:
                break

            # Optimize and store frame
            pil_frame = self._optimize_frame(frame)
            frames.append(pil_frame)

            frame_count += 1

        cap.release()
        return frames, frame_interval

    async def process_video_async(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file asynchronously and return structured content.

        Args:
            video_path: Path to video file

        Returns:
            Dict containing structured video analysis
        """
        video_file = Path(video_path)
        logger.debug(f"Starting to process video file: {video_file.name}")

        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        # Validate video
        self._validate_video(str(video_file))

        # Calculate file hash
        file_hash = HashUtils.calculate_file_hash(str(video_file))
        logger.debug(f"Calculated file hash: {file_hash}")

        try:
            # Check cache
            cached_result = await self.cache.get(file_hash)
            if cached_result:
                logger.debug("Found cached results - using cached data")
                return cached_result

            # Extract frames
            frames, frame_interval = await self._extract_frames(
                str(video_file), file_hash
            )
            logger.info(f"Extracted {len(frames)} frames from video")

            # Process all frames in a single batch
            analysis = await self._process_video_frames(frames, frame_interval)

            # Compile final results
            result = {
                "file_object": {
                    "file_name": video_file.name,
                    "file_hash": file_hash,
                    "file_type": "video",
                    "duration": len(frames) * frame_interval,
                    "frame_count": len(frames),
                    "frame_rate": self.config.frame_rate,
                    "file_full_path": str(video_file.absolute()),
                    "analysis": analysis,
                    "frames": [
                        {
                            "frame_number": i + 1,
                            "timestamp": i * frame_interval,
                        }
                        for i in range(len(frames))
                    ],
                }
            }

            # Cache results
            await self.cache.set(file_hash, result)
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_file}: {str(e)}")
            raise

    async def _process_video_frames(
        self, frames: List[Image.Image], frame_interval: float
    ) -> Dict[str, Any]:
        """
        Process all video frames in a single batch to extract overall meaning.

        Args:
            frames: List of video frames as PIL Images
            frame_interval: Time interval between frames

        Returns:
            Dict containing the analysis of the video content
        """
        try:
            # Process all frames in a single vision model call
            logger.info(f"Processing batch of {len(frames)} frames")
            content = await self.vision_model.aprocess_image(
                frames,
                prompt=self.config.prompt,
            )

            # Structure the analysis
            analysis = {
                "content": content.strip(),
                "confidence": 1.0,  # TODO: Implement confidence scoring
                "metadata": {
                    "total_frames": len(frames),
                    "frame_interval": frame_interval,
                    "total_duration": len(frames) * frame_interval,
                },
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_video_async.

        Args:
            video_path: Path to video file

        Returns:
            Dict containing structured video analysis
        """
        return asyncio.run(self.process_video_async(video_path))


if __name__ == "__main__":
    vid_file = "tests/sample/vids/rock.mp4"
    parser = VidCapture()
    result = parser.process_video(vid_file)
    print(result)

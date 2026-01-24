"""
Simple video capture module for extracting frames from videos.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from aicapture.audio_transcriber import OpenAIAudioTranscriber, TimestampedTranscription
from aicapture.cache import FileCache, HashUtils, S3Cache, TwoLayerCache

# Fix circular import by importing directly from vision_models
from aicapture.vision_models import VisionModel, create_default_vision_model


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    max_duration_seconds: int = 300  # Max video duration (5 minutes)
    frame_rate: float = 0.5  # Frames per second to extract (options: 0.3, 0.5, 1.0)
    chunk_duration_seconds: int = 60  # Duration of each processing chunk in seconds
    supported_formats: tuple = (".mp4", ".avi", ".mov", ".mkv")
    target_frame_size: tuple = (768, 768)  # Target size for resized frames
    resize_frames: bool = True
    cache_dir: Optional[str] = None  # Directory for caching results
    cloud_bucket: Optional[str] = None  # S3 bucket for cloud caching
    # Transcription settings
    enable_transcription: bool = False  # Enable Whisper audio transcription
    transcription_model: str = "whisper-1"  # Whisper model name
    transcription_language: Optional[str] = None  # Language hint (ISO 639-1), None for auto-detect


class VideoValidationError(Exception):
    """Raised when video validation fails."""


class VidCapture:
    """
    Simple utility for extracting frames from video files and analyzing them.

    Features:
    - Extracts frames from video files at specified rates
    - Analyzes frames with a vision model
    - Provides caching to avoid re-processing the same video with the same prompt

    The cache key is generated based on:
    - The SHA-256 hash of the video file
    - The SHA-256 hash of the prompt
    - The frame extraction rate

    Both local file caching and S3 cloud caching are supported.
    """

    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        vision_model: Optional[VisionModel] = None,
        invalidate_cache: bool = False,
    ):
        """
        Initialize VideoCapture with configuration.

        Args:
            config: Configuration for video processing
            vision_model: Vision model for image analysis (created if None)
            invalidate_cache: If True, bypass cache for reads
        """
        self.config = config or VideoConfig()
        self.vision_model = vision_model or create_default_vision_model()
        self.invalidate_cache = invalidate_cache

        # Initialize file cache
        cache_dir = self.config.cache_dir or "tmp/.vid_capture_cache"
        file_cache = FileCache(cache_dir=cache_dir)

        # Initialize S3 cache if bucket is provided
        s3_cache = None
        if self.config.cloud_bucket:
            s3_cache = S3Cache(bucket=self.config.cloud_bucket, prefix="production/video_results")

        # Set up two-layer cache
        self.cache = TwoLayerCache(file_cache=file_cache, s3_cache=s3_cache, invalidate_cache=invalidate_cache)

        # Initialize transcription cache (separate from video results cache)
        transcription_cache_dir = f"{cache_dir}/transcriptions"
        self.transcription_file_cache = FileCache(cache_dir=transcription_cache_dir)

        # S3 cache for transcriptions
        transcription_s3_cache = None
        if self.config.cloud_bucket:
            transcription_s3_cache = S3Cache(bucket=self.config.cloud_bucket, prefix="production/transcriptions")

        self.transcription_cache = TwoLayerCache(
            file_cache=self.transcription_file_cache, s3_cache=transcription_s3_cache, invalidate_cache=invalidate_cache
        )

    def _validate_video(self, video_path: str) -> None:
        """
        Validate video file format and duration.

        Args:
            video_path: Path to video file

        Raises:
            VideoValidationError: If validation fails
        """
        if not any(video_path.lower().endswith(fmt) for fmt in self.config.supported_formats):
            raise VideoValidationError(f"Unsupported video format. Supported formats: {self.config.supported_formats}")

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
        if self.config.resize_frames:
            width, height = image.size
            if width > self.config.target_frame_size[0] or height > self.config.target_frame_size[1]:
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

    async def capture_async(self, prompt: str, images: List[Image.Image], **kwargs: Any) -> str:
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
        result = await self.vision_model.process_image_async(image=images, prompt=prompt, **kwargs)

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

    def _get_cache_key(self, video_path: str, prompt: str) -> Optional[str]:
        """
        Generate a cache key from video file hash, prompt, and frame rate.

        Args:
            video_path: Path to the video file
            prompt: Instruction prompt for the vision model

        Returns:
            Cache key or None if generation fails
        """
        try:
            # Calculate file hash
            file_hash = HashUtils.calculate_file_hash(video_path)

            # Create prompt hash
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

            # Include transcription config in cache key
            transcription_suffix = ""
            if self.config.enable_transcription:
                transcription_suffix = f"_transcribe_{self.config.transcription_model}"

            # Create cache key with frame rate
            return f"{file_hash}_{prompt_hash}_{self.config.frame_rate}{transcription_suffix}"
        except Exception as e:
            print(f"Failed to generate cache key: {str(e)}")
            return None

    async def _get_from_cache_async(self, cache_key: str) -> Optional[str]:
        """
        Try to get a result from the cache asynchronously.

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached result or None if not found
        """
        if not cache_key or self.invalidate_cache:
            return None

        try:
            cached_result = await self.cache.get(cache_key)
            if cached_result and "result" in cached_result:
                print(f"Using cached result with key: {cache_key}")
                return cached_result.get("result", None)  # type: ignore
        except Exception as e:
            print(f"Cache lookup failed: {str(e)}")

        return None

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Try to get a result from the cache (synchronous version).

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached result or None if not found
        """
        return asyncio.run(self._get_from_cache_async(cache_key))

    async def _save_to_cache_async(self, cache_key: str, result: str) -> None:
        """
        Save a result to the cache asynchronously.

        Args:
            cache_key: The cache key to use
            result: The result to cache
        """
        try:
            await self.cache.set(cache_key, {"result": result})
            print(f"Saved result to cache with key: {cache_key}")
        except Exception as e:
            print(f"Failed to save to cache: {str(e)}")

    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """
        Save a result to the cache (synchronous version).

        Args:
            cache_key: The cache key to use
            result: The result to cache
        """
        asyncio.run(self._save_to_cache_async(cache_key, result))

    def _get_transcription_cache_key(self, video_path: str) -> Optional[str]:
        """
        Generate a cache key for transcription based on video hash, model, and language.

        Args:
            video_path: Path to the video file

        Returns:
            Cache key or None if generation fails
        """
        try:
            # Calculate video file hash
            video_hash = HashUtils.calculate_file_hash(video_path)

            # Include model and language in cache key
            model = self.config.transcription_model
            language = self.config.transcription_language or "auto"

            return f"{video_hash}_{model}_{language}"
        except Exception as e:
            logger.warning(f"Failed to generate transcription cache key: {e}")
            return None

    async def _load_transcription_from_cache_async(self, cache_key: str) -> Optional[TimestampedTranscription]:
        """
        Load transcription from cache asynchronously.

        Args:
            cache_key: The cache key to look up

        Returns:
            TimestampedTranscription or None if not found
        """
        if not cache_key or self.invalidate_cache:
            return None

        try:
            cached_data = await self.transcription_cache.get(cache_key)
            if cached_data and "transcription" in cached_data:
                logger.info(f"Loaded transcription from cache: {cache_key}")
                return TimestampedTranscription.from_dict(cached_data["transcription"])
        except Exception as e:
            logger.warning(f"Failed to load transcription from cache: {e}")

        return None

    def _load_transcription_from_cache(self, cache_key: str) -> Optional[TimestampedTranscription]:
        """
        Load transcription from cache (synchronous version).

        Args:
            cache_key: The cache key to look up

        Returns:
            TimestampedTranscription or None if not found
        """
        return asyncio.run(self._load_transcription_from_cache_async(cache_key))

    async def _save_transcription_to_cache_async(self, cache_key: str, transcription: TimestampedTranscription) -> None:
        """
        Save transcription to cache asynchronously.

        Args:
            cache_key: The cache key to use
            transcription: The transcription to cache
        """
        try:
            cache_data = {"transcription": transcription.to_dict()}
            await self.transcription_cache.set(cache_key, cache_data)
            logger.info(f"Saved transcription to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save transcription to cache: {e}")

    def _save_transcription_to_cache(self, cache_key: str, transcription: TimestampedTranscription) -> None:
        """
        Save transcription to cache (synchronous version).

        Args:
            cache_key: The cache key to use
            transcription: The transcription to cache
        """
        asyncio.run(self._save_transcription_to_cache_async(cache_key, transcription))

    def _get_transcription(self, video_path: str) -> Optional[TimestampedTranscription]:
        """
        Get audio transcription for a video file.

        Checks cache first, then calls Whisper API if not cached.

        Args:
            video_path: Path to the video file

        Returns:
            TimestampedTranscription or None if transcription fails or is disabled.
        """
        if not self.config.enable_transcription:
            return None

        # Try to load from cache first
        cache_key = self._get_transcription_cache_key(video_path)
        if cache_key:
            cached_transcription = self._load_transcription_from_cache(cache_key)
            if cached_transcription:
                return cached_transcription

        # Cache miss - call Whisper API
        try:
            logger.info(f"Transcription cache miss, calling Whisper API for {video_path}")
            transcriber = OpenAIAudioTranscriber(model=self.config.transcription_model)
            transcription = transcriber.transcribe_video(
                video_path,
                language=self.config.transcription_language,
            )
            logger.info(
                f"Transcription complete: {len(transcription.segments)} segments, "
                f"duration={transcription.duration:.1f}s"
            )

            # Save to cache for future use
            if cache_key:
                self._save_transcription_to_cache(cache_key, transcription)

            return transcription
        except Exception as e:
            logger.warning(f"Audio transcription failed, proceeding without it: {e}")
            return None

    def _enrich_prompt_with_transcription(self, prompt: str, transcription: Optional[TimestampedTranscription]) -> str:
        """
        Enrich the prompt with transcription context if available.

        Args:
            prompt: Original user prompt
            transcription: Timestamped transcription or None

        Returns:
            Enriched prompt with transcription context appended, or original prompt.
        """
        if transcription is None:
            return prompt

        context = transcription.to_prompt_context()
        if not context:
            return prompt

        return f"{prompt}\n{context}\nPlease use both the visual frames and the audio transcription above to analyze this video."

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count / fps if fps > 0 else 0.0

    def _extract_frames_for_range(self, video_path: str, start_sec: float, end_sec: float) -> List[Image.Image]:
        """
        Extract frames from a specific time range of a video.

        Args:
            video_path: Path to the video file
            start_sec: Start time in seconds
            end_sec: End time in seconds

        Returns:
            List of PIL Images extracted from the range
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        range_duration = end_sec - start_sec
        num_frames = int(range_duration * self.config.frame_rate)
        frame_interval = 1.0 / self.config.frame_rate if self.config.frame_rate > 0 else range_duration

        for i in range(num_frames):
            timestamp = start_sec + i * frame_interval
            if timestamp >= end_sec:
                break
            frame_position = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self._optimize_frame(frame))

        cap.release()
        return frames

    def _get_segments_for_range(
        self, transcription: Optional[TimestampedTranscription], start_sec: float, end_sec: float
    ) -> Optional[TimestampedTranscription]:
        """
        Get transcription segments that fall within a time range.

        Args:
            transcription: Full transcription
            start_sec: Range start in seconds
            end_sec: Range end in seconds

        Returns:
            Filtered TimestampedTranscription or None
        """
        if transcription is None or not transcription.segments:
            return None

        from aicapture.audio_transcriber import TranscriptionSegment

        filtered = [
            TranscriptionSegment(start=seg.start, end=seg.end, text=seg.text)
            for seg in transcription.segments
            if seg.start < end_sec and seg.end > start_sec
        ]
        if not filtered:
            return None

        return TimestampedTranscription(
            segments=filtered,
            language=transcription.language,
            duration=end_sec - start_sec,
            full_text=" ".join(seg.text for seg in filtered),
        )

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _build_chunk_prompt(
        self,
        user_prompt: str,
        chunk_idx: int,
        total_chunks: int,
        start_sec: float,
        end_sec: float,
        total_duration: float,
        num_frames: int,
        chunk_transcription: Optional[TimestampedTranscription],
    ) -> str:
        """
        Build a metadata-rich prompt for a video chunk.

        Includes chunk position, fps, time range, and matching transcript.
        """
        start_fmt = self._format_time(start_sec)
        end_fmt = self._format_time(end_sec)
        total_fmt = self._format_time(total_duration)

        header = (
            f"--- Video Analysis: Chunk {chunk_idx + 1}/{total_chunks} ---\n"
            f"Time range: {start_fmt} to {end_fmt} (total video duration: {total_fmt})\n"
            f"Frames: {num_frames} frames captured at {self.config.frame_rate} fps "
            f"(1 frame every {1.0 / self.config.frame_rate:.1f} seconds)\n"
        )

        # Add transcript for this chunk
        transcript_section = ""
        if chunk_transcription and chunk_transcription.segments:
            transcript_section = chunk_transcription.to_prompt_context()

        prompt = f"{header}{transcript_section}\n{user_prompt}"

        if total_chunks > 1:
            prompt += (
                f"\n\nNote: This is chunk {chunk_idx + 1} of {total_chunks}. "
                f"Focus on analyzing the content within this time range ({start_fmt} - {end_fmt})."
            )

        return prompt

    def _synthesize_results(self, chunk_results: List[str], user_prompt: str, total_duration: float) -> str:
        """
        Combine chunk analysis results into a final synthesis.

        Args:
            chunk_results: List of analysis results from each chunk
            user_prompt: Original user prompt
            total_duration: Total video duration

        Returns:
            Synthesized final result
        """
        if len(chunk_results) == 1:
            return chunk_results[0]

        total_fmt = self._format_time(total_duration)

        synthesis_prompt = (
            f"The following are analysis results from {len(chunk_results)} consecutive chunks "
            f"of a video (total duration: {total_fmt}), processed at {self.config.frame_rate} fps.\n\n"
        )

        for i, result in enumerate(chunk_results):
            chunk_start = i * self.config.chunk_duration_seconds
            chunk_end = min((i + 1) * self.config.chunk_duration_seconds, total_duration)
            start_fmt = self._format_time(chunk_start)
            end_fmt = self._format_time(chunk_end)
            synthesis_prompt += f"--- Chunk {i + 1}/{len(chunk_results)} [{start_fmt} - {end_fmt}] ---\n{result}\n\n"

        synthesis_prompt += (
            f"Based on all the chunk analyses above, provide a comprehensive and unified response to:\n{user_prompt}"
        )

        # Use the vision model for synthesis (text-only, no images)
        # Send a single blank image since VLMs require at least one image
        placeholder = Image.new("RGB", (1, 1), color=(0, 0, 0))
        return self.capture(synthesis_prompt, [placeholder])

    def _process_video_chunked(self, video_path: str, prompt: str, total_duration: float, **kwargs: Any) -> str:
        """
        Process a video in chunks, then synthesize results.

        Args:
            video_path: Path to the video file
            prompt: User prompt
            total_duration: Total video duration in seconds

        Returns:
            Synthesized result from all chunks
        """
        # Calculate chunks
        chunk_duration = self.config.chunk_duration_seconds
        num_chunks = max(1, int((total_duration + chunk_duration - 1) // chunk_duration))

        logger.info(f"Processing video in {num_chunks} chunks ({chunk_duration}s each, {self.config.frame_rate} fps)")

        # Get full transcription once (more accurate than chunking audio)
        transcription = self._get_transcription(video_path)

        # Process each chunk
        chunk_results: List[str] = []
        for chunk_idx in range(num_chunks):
            start_sec = chunk_idx * chunk_duration
            end_sec = min((chunk_idx + 1) * chunk_duration, total_duration)

            # Extract frames for this chunk
            frames = self._extract_frames_for_range(video_path, start_sec, end_sec)
            if not frames:
                logger.warning(f"No frames extracted for chunk {chunk_idx + 1}")
                continue

            # Get transcript for this time range
            chunk_transcription = self._get_segments_for_range(transcription, start_sec, end_sec)

            # Build chunk prompt with metadata
            chunk_prompt = self._build_chunk_prompt(
                user_prompt=prompt,
                chunk_idx=chunk_idx,
                total_chunks=num_chunks,
                start_sec=start_sec,
                end_sec=end_sec,
                total_duration=total_duration,
                num_frames=len(frames),
                chunk_transcription=chunk_transcription,
            )

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{num_chunks} "
                f"[{self._format_time(start_sec)} - {self._format_time(end_sec)}] "
                f"({len(frames)} frames)"
            )

            result = self.capture(chunk_prompt, frames, **kwargs)
            chunk_results.append(result)

        if not chunk_results:
            raise ValueError(f"No frames could be extracted from {video_path}")

        # Synthesize all chunk results into final answer
        return self._synthesize_results(chunk_results, prompt, total_duration)

    async def _process_video_chunked_async(
        self, video_path: str, prompt: str, total_duration: float, **kwargs: Any
    ) -> str:
        """
        Async version of chunked video processing.
        """
        chunk_duration = self.config.chunk_duration_seconds
        num_chunks = max(1, int((total_duration + chunk_duration - 1) // chunk_duration))

        logger.info(f"Processing video in {num_chunks} chunks ({chunk_duration}s each, {self.config.frame_rate} fps)")

        transcription = self._get_transcription(video_path)

        chunk_results: List[str] = []
        for chunk_idx in range(num_chunks):
            start_sec = chunk_idx * chunk_duration
            end_sec = min((chunk_idx + 1) * chunk_duration, total_duration)

            frames = self._extract_frames_for_range(video_path, start_sec, end_sec)
            if not frames:
                continue

            chunk_transcription = self._get_segments_for_range(transcription, start_sec, end_sec)

            chunk_prompt = self._build_chunk_prompt(
                user_prompt=prompt,
                chunk_idx=chunk_idx,
                total_chunks=num_chunks,
                start_sec=start_sec,
                end_sec=end_sec,
                total_duration=total_duration,
                num_frames=len(frames),
                chunk_transcription=chunk_transcription,
            )

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{num_chunks} "
                f"[{self._format_time(start_sec)} - {self._format_time(end_sec)}] "
                f"({len(frames)} frames)"
            )

            result = await self.capture_async(chunk_prompt, frames, **kwargs)
            chunk_results.append(result)

        if not chunk_results:
            raise ValueError(f"No frames could be extracted from {video_path}")

        if len(chunk_results) == 1:
            return chunk_results[0]

        # Async synthesis
        total_fmt = self._format_time(total_duration)
        synthesis_prompt = (
            f"The following are analysis results from {len(chunk_results)} consecutive chunks "
            f"of a video (total duration: {total_fmt}), processed at {self.config.frame_rate} fps.\n\n"
        )
        for i, result in enumerate(chunk_results):
            chunk_start = i * chunk_duration
            chunk_end = min((i + 1) * chunk_duration, total_duration)
            start_fmt = self._format_time(chunk_start)
            end_fmt = self._format_time(chunk_end)
            synthesis_prompt += f"--- Chunk {i + 1}/{len(chunk_results)} [{start_fmt} - {end_fmt}] ---\n{result}\n\n"

        synthesis_prompt += (
            f"Based on all the chunk analyses above, provide a comprehensive and unified response to:\n{prompt}"
        )

        placeholder = Image.new("RGB", (1, 1), color=(0, 0, 0))
        return await self.capture_async(synthesis_prompt, [placeholder])

    def process_video(self, video_path: str, prompt: str, **kwargs: Any) -> str:
        """
        Extract frames from a video and analyze them with a vision model.

        For videos longer than chunk_duration_seconds, processes in chunks
        and synthesizes a final result. Each chunk includes metadata about
        its position, fps, and matching transcript segments.

        Args:
            video_path: Path to the video file
            prompt: Instruction prompt for the vision model
            **kwargs: Additional parameters to pass to the vision model

        Returns:
            String containing the extracted knowledge from the video frames
        """
        # Validate video format
        self._validate_video(video_path)

        # Check cache first
        cache_key = self._get_cache_key(video_path, prompt)
        cached_result = self._get_from_cache(cache_key)  # type: ignore
        if cached_result:
            return cached_result

        # Get video duration to decide processing strategy
        total_duration = self._get_video_duration(video_path)

        if total_duration > self.config.chunk_duration_seconds:
            # Chunked processing for longer videos
            result = self._process_video_chunked(video_path, prompt, total_duration, **kwargs)
        else:
            # Single-chunk processing (original flow with metadata)
            frames = self._extract_frames_for_range(video_path, 0, total_duration)
            if not frames:
                raise ValueError(f"No frames could be extracted from {video_path}")

            transcription = self._get_transcription(video_path)
            chunk_transcription = self._get_segments_for_range(transcription, 0, total_duration)
            enriched_prompt = self._build_chunk_prompt(
                user_prompt=prompt,
                chunk_idx=0,
                total_chunks=1,
                start_sec=0,
                end_sec=total_duration,
                total_duration=total_duration,
                num_frames=len(frames),
                chunk_transcription=chunk_transcription,
            )
            result = self.capture(enriched_prompt, frames, **kwargs)

        # Store in cache
        self._save_to_cache(cache_key, result)  # type: ignore

        return result

    async def process_video_async(self, video_path: str, prompt: str, **kwargs: Any) -> str:
        """
        Asynchronous version of process_video.

        For videos longer than chunk_duration_seconds, processes in chunks
        and synthesizes a final result.

        Args:
            video_path: Path to the video file
            prompt: Instruction prompt for the vision model
            **kwargs: Additional parameters to pass to the vision model

        Returns:
            String containing the extracted knowledge from the video frames
        """
        self._validate_video(video_path)

        cache_key = self._get_cache_key(video_path, prompt)
        cached_result = await self._get_from_cache_async(cache_key)  # type: ignore
        if cached_result:
            return cached_result

        total_duration = self._get_video_duration(video_path)

        if total_duration > self.config.chunk_duration_seconds:
            result = await self._process_video_chunked_async(video_path, prompt, total_duration, **kwargs)
        else:
            frames = self._extract_frames_for_range(video_path, 0, total_duration)
            if not frames:
                raise ValueError(f"No frames could be extracted from {video_path}")

            transcription = self._get_transcription(video_path)
            chunk_transcription = self._get_segments_for_range(transcription, 0, total_duration)
            enriched_prompt = self._build_chunk_prompt(
                user_prompt=prompt,
                chunk_idx=0,
                total_chunks=1,
                start_sec=0,
                end_sec=total_duration,
                total_duration=total_duration,
                num_frames=len(frames),
                chunk_transcription=chunk_transcription,
            )
            result = await self.capture_async(enriched_prompt, frames, **kwargs)

        await self._save_to_cache_async(cache_key, result)  # type: ignore

        return result

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
            if "cap" in locals() and cap is not None:
                cap.release()

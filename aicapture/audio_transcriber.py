"""
Audio transcription module for extracting timestamped speech from video files.

Uses OpenAI Whisper API (or Azure OpenAI Whisper) to produce segment-level
transcriptions with timestamps, similar to LRC (lyrics) format.

The transcription can be used to enrich VLM prompts with audio context
when analyzing video content.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

SUPPORTED_VIDEO_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SUPPORTED_AUDIO_FORMATS = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed speech with timestamps.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text for this segment.
    """

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

    def to_lrc_line(self) -> str:
        """Convert segment to LRC format line.

        Returns:
            String in format "[MM:SS.cc] Text"
        """
        minutes = int(self.start // 60)
        seconds = self.start % 60
        return f"[{minutes:02d}:{seconds:05.2f}] {self.text}"


@dataclass
class TimestampedTranscription:
    """Complete transcription with timestamped segments.

    Attributes:
        segments: List of transcription segments with timestamps.
        language: Detected or specified language.
        duration: Total duration of the audio in seconds.
        full_text: Complete transcription text without timestamps.
    """

    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    full_text: str = ""

    def to_lrc(self) -> str:
        """Convert transcription to LRC (lyrics) format.

        Returns:
            Multi-line string with each segment as "[MM:SS.cc] text".
            Empty string if no segments.
        """
        if not self.segments:
            return ""
        return "\n".join(seg.to_lrc_line() for seg in self.segments)

    def to_prompt_context(self) -> str:
        """Convert transcription to a context string for VLM prompts.

        Returns:
            Formatted string with header and LRC-formatted segments,
            suitable for embedding in a prompt. Empty string if no segments.
        """
        if not self.segments:
            return ""
        lrc_content = self.to_lrc()
        return f"\n--- Video Audio Transcription ---\n{lrc_content}\n---\n"

    @classmethod
    def from_whisper_response(cls, response: Dict[str, Any]) -> "TimestampedTranscription":
        """Create a TimestampedTranscription from a Whisper API verbose_json response.

        Args:
            response: Dictionary from Whisper API with 'segments', 'language',
                     'duration', and 'text' fields.

        Returns:
            TimestampedTranscription instance.
        """
        segments = []
        for seg in response.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(
                    TranscriptionSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=text,
                    )
                )

        return cls(
            segments=segments,
            language=response.get("language", ""),
            duration=float(response.get("duration", 0.0)),
            full_text=response.get("text", "").strip(),
        )


def extract_audio_from_video(
    video_path: str,
    output_dir: Optional[str] = None,
    output_format: str = "mp3",
) -> str:
    """Extract audio track from a video file.

    Uses moviepy to extract the audio stream and save it as an audio file
    suitable for Whisper API input.

    Args:
        video_path: Path to the video file.
        output_dir: Directory for the output audio file. Uses temp dir if None.
        output_format: Output audio format (default: "mp3").

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video format is not supported.
        RuntimeError: If audio extraction fails.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if video_file.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError(f"Unsupported video format: {video_file.suffix}. Supported formats: {SUPPORTED_VIDEO_FORMATS}")

    try:
        from moviepy import VideoFileClip  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "moviepy is required for audio extraction. Install it with: pip install aicapture[video]"
        ) from e

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="aicapture_audio_")

    output_path = str(Path(output_dir) / f"{video_file.stem}_audio.{output_format}")

    logger.info(f"Extracting audio from {video_path} to {output_path}")

    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise RuntimeError(f"No audio track found in video: {video_path}")
        clip.audio.write_audiofile(output_path, logger=None)
        clip.close()
    except Exception as e:
        if "No audio track" in str(e):
            raise
        raise RuntimeError(f"Failed to extract audio: {e}") from e

    return output_path


class AudioTranscriber(ABC):
    """Base class for audio transcription providers.

    Subclasses implement the Whisper API call for their specific provider
    (OpenAI, Azure OpenAI).
    """

    model: str

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TimestampedTranscription:
        """Transcribe an audio file with timestamps.

        Args:
            audio_path: Path to the audio file.
            language: Optional language hint (ISO 639-1 code, e.g., "en", "ja").

        Returns:
            TimestampedTranscription with segment-level timestamps.
        """
        ...

    def transcribe_video(
        self,
        video_path: str,
        language: Optional[str] = None,
    ) -> TimestampedTranscription:
        """Extract audio from video and transcribe it.

        Args:
            video_path: Path to the video file.
            language: Optional language hint.

        Returns:
            TimestampedTranscription with segment-level timestamps.
        """
        audio_path = extract_audio_from_video(video_path)
        try:
            return self.transcribe(audio_path, language=language)
        finally:
            # Cleanup temporary audio file
            try:
                Path(audio_path).unlink(missing_ok=True)
                # Also try to remove the temp directory if empty
                parent = Path(audio_path).parent
                if parent.name.startswith("aicapture_audio_"):
                    parent.rmdir()
            except OSError:
                pass


class OpenAIAudioTranscriber(AudioTranscriber):
    """Audio transcriber using OpenAI Whisper API.

    Uses the whisper-1 model with verbose_json response format
    to get segment-level timestamps.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: Optional[str] = None,
    ):
        """Initialize OpenAI audio transcriber.

        Args:
            model: Whisper model name (default: "whisper-1").
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TimestampedTranscription:
        """Transcribe audio using OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file.
            language: Optional language hint (ISO 639-1).

        Returns:
            TimestampedTranscription with segment-level timestamps.
        """
        logger.info(f"Transcribing audio with OpenAI Whisper: {audio_path}")

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language

        with open(audio_path, "rb") as audio_file:
            kwargs["file"] = audio_file
            response = self._client.audio.transcriptions.create(**kwargs)

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> TimestampedTranscription:
        """Parse the Whisper API response into TimestampedTranscription."""
        segments = []
        for seg in response.segments or []:
            text = seg.text.strip() if isinstance(seg.text, str) else str(seg.text).strip()
            if text:
                start = seg.start if isinstance(seg.start, (int, float)) else float(seg.start)
                end = seg.end if isinstance(seg.end, (int, float)) else float(seg.end)
                segments.append(TranscriptionSegment(start=start, end=end, text=text))

        return TimestampedTranscription(
            segments=segments,
            language=getattr(response, "language", "") or "",
            duration=float(getattr(response, "duration", 0.0) or 0.0),
            full_text=(getattr(response, "text", "") or "").strip(),
        )


class AzureOpenAIAudioTranscriber(AudioTranscriber):
    """Audio transcriber using Azure OpenAI Whisper API.

    Uses the whisper-1 model deployed on Azure with verbose_json
    response format for segment-level timestamps.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """Initialize Azure OpenAI audio transcriber.

        Args:
            model: Whisper deployment name (default: "whisper-1").
            api_key: Azure OpenAI API key. Falls back to env var.
            api_base: Azure OpenAI endpoint URL. Falls back to env var.
            api_version: API version. Falls back to env var.
        """
        self.model = model
        self._api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY", "")
        self._api_base = api_base if api_base is not None else os.getenv("AZURE_OPENAI_API_URL", "")
        self._api_version = (
            api_version if api_version is not None else os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-01-preview")
        )

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._api_base,
            api_version=self._api_version,
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TimestampedTranscription:
        """Transcribe audio using Azure OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file.
            language: Optional language hint (ISO 639-1).

        Returns:
            TimestampedTranscription with segment-level timestamps.
        """
        logger.info(f"Transcribing audio with Azure OpenAI Whisper: {audio_path}")

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language

        with open(audio_path, "rb") as audio_file:
            kwargs["file"] = audio_file
            response = self._client.audio.transcriptions.create(**kwargs)

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> TimestampedTranscription:
        """Parse the Whisper API response into TimestampedTranscription."""
        segments = []
        for seg in response.segments or []:
            text = seg.text.strip() if isinstance(seg.text, str) else str(seg.text).strip()
            if text:
                start = seg.start if isinstance(seg.start, (int, float)) else float(seg.start)
                end = seg.end if isinstance(seg.end, (int, float)) else float(seg.end)
                segments.append(TranscriptionSegment(start=start, end=end, text=text))

        return TimestampedTranscription(
            segments=segments,
            language=getattr(response, "language", "") or "",
            duration=float(getattr(response, "duration", 0.0) or 0.0),
            full_text=(getattr(response, "text", "") or "").strip(),
        )

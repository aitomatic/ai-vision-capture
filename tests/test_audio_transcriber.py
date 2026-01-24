"""
Tests for the audio transcription module.

TDD approach: these tests define the expected interface for audio transcription
with timestamped output (LRC-style) using OpenAI Whisper API.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aicapture.audio_transcriber import (
    AzureOpenAIAudioTranscriber,
    OpenAIAudioTranscriber,
    TimestampedTranscription,
    TranscriptionSegment,
    extract_audio_from_video,
)

# Test video with known narration
TEST_VIDEO_PATH = Path(__file__).parent / "sample" / "vids" / "instruction_narrated.mp4"

# Known transcript content from the test video
KNOWN_PHRASES = [
    "open the application",
    "settings",
    "configuration",
    "dark mode",
    "save",
    "restart",
]


# ============================================================
# TranscriptionSegment tests
# ============================================================


class TestTranscriptionSegment:
    """Tests for the TranscriptionSegment dataclass."""

    def test_create_segment(self):
        """Test creating a TranscriptionSegment."""
        segment = TranscriptionSegment(start=0.0, end=5.5, text="Hello world.")
        assert segment.start == 0.0
        assert segment.end == 5.5
        assert segment.text == "Hello world."

    def test_segment_duration(self):
        """Test segment duration calculation."""
        segment = TranscriptionSegment(start=2.0, end=7.5, text="Test")
        assert segment.duration == pytest.approx(5.5)

    def test_segment_to_lrc_line(self):
        """Test converting a segment to LRC format line."""
        segment = TranscriptionSegment(start=65.5, end=70.0, text="One minute in.")
        lrc_line = segment.to_lrc_line()
        assert lrc_line == "[01:05.50] One minute in."

    def test_segment_to_lrc_line_zero_start(self):
        """Test LRC format for segment starting at 0."""
        segment = TranscriptionSegment(start=0.0, end=3.2, text="Beginning.")
        lrc_line = segment.to_lrc_line()
        assert lrc_line == "[00:00.00] Beginning."

    def test_segment_to_lrc_line_large_timestamp(self):
        """Test LRC format for timestamps over 10 minutes."""
        segment = TranscriptionSegment(start=623.4, end=630.0, text="Ten minutes in.")
        lrc_line = segment.to_lrc_line()
        assert lrc_line == "[10:23.40] Ten minutes in."


# ============================================================
# TimestampedTranscription tests
# ============================================================


class TestTimestampedTranscription:
    """Tests for the TimestampedTranscription dataclass."""

    @pytest.fixture
    def sample_transcription(self):
        """Create a sample transcription with multiple segments."""
        segments = [
            TranscriptionSegment(start=0.0, end=4.5, text="Step one. Open the application."),
            TranscriptionSegment(start=4.5, end=9.0, text="Step two. Click the button."),
            TranscriptionSegment(start=9.0, end=14.0, text="Step three. Save changes."),
        ]
        return TimestampedTranscription(
            segments=segments,
            language="en",
            duration=14.0,
            full_text="Step one. Open the application. Step two. Click the button. Step three. Save changes.",
        )

    def test_create_transcription(self, sample_transcription):
        """Test creating a TimestampedTranscription."""
        assert len(sample_transcription.segments) == 3
        assert sample_transcription.language == "en"
        assert sample_transcription.duration == 14.0
        assert "Open the application" in sample_transcription.full_text

    def test_to_lrc(self, sample_transcription):
        """Test converting transcription to LRC format."""
        lrc = sample_transcription.to_lrc()
        lines = lrc.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "[00:00.00] Step one. Open the application."
        assert lines[1] == "[00:04.50] Step two. Click the button."
        assert lines[2] == "[00:09.00] Step three. Save changes."

    def test_to_prompt_context(self, sample_transcription):
        """Test converting transcription to prompt context string."""
        context = sample_transcription.to_prompt_context()
        # Should contain header
        assert "Video Audio Transcription" in context
        # Should contain LRC-formatted lines
        assert "[00:00.00]" in context
        assert "Open the application" in context
        # Should contain all segments
        assert "Step two" in context
        assert "Step three" in context

    def test_empty_transcription(self):
        """Test handling of empty transcription (no speech detected)."""
        transcription = TimestampedTranscription(
            segments=[],
            language="en",
            duration=10.0,
            full_text="",
        )
        assert transcription.to_lrc() == ""
        assert transcription.to_prompt_context() == ""

    def test_from_whisper_response(self):
        """Test creating transcription from Whisper API verbose_json response."""
        # Simulated Whisper API response structure
        whisper_response = {
            "task": "transcribe",
            "language": "english",
            "duration": 14.0,
            "text": "Step one. Open the application. Step two. Click the button.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 4.5,
                    "text": " Step one. Open the application.",
                },
                {
                    "id": 1,
                    "start": 4.5,
                    "end": 9.0,
                    "text": " Step two. Click the button.",
                },
            ],
        }
        transcription = TimestampedTranscription.from_whisper_response(whisper_response)
        assert len(transcription.segments) == 2
        assert transcription.language == "english"
        assert transcription.duration == 14.0
        assert transcription.segments[0].text == "Step one. Open the application."
        assert transcription.segments[1].start == 4.5


# ============================================================
# Audio extraction tests
# ============================================================


class TestAudioExtraction:
    """Tests for extracting audio from video files."""

    def test_extract_audio_from_video(self):
        """Test extracting audio from the test video."""
        assert TEST_VIDEO_PATH.exists(), f"Test video not found: {TEST_VIDEO_PATH}"

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = extract_audio_from_video(
                str(TEST_VIDEO_PATH),
                output_dir=tmpdir,
            )
            # Should produce a file
            assert Path(audio_path).exists()
            # Should be an audio format supported by Whisper
            assert audio_path.endswith((".mp3", ".wav", ".m4a", ".mp4"))
            # File should have content
            assert Path(audio_path).stat().st_size > 0

    def test_extract_audio_output_format(self):
        """Test that extracted audio is in mp3 format by default."""
        assert TEST_VIDEO_PATH.exists()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = extract_audio_from_video(
                str(TEST_VIDEO_PATH),
                output_dir=tmpdir,
                output_format="mp3",
            )
            assert audio_path.endswith(".mp3")

    def test_extract_audio_nonexistent_video(self):
        """Test error handling for non-existent video file."""
        with pytest.raises(FileNotFoundError):
            extract_audio_from_video("/nonexistent/video.mp4")

    def test_extract_audio_invalid_format(self):
        """Test error handling for unsupported video format."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ValueError, match="Unsupported"):
                extract_audio_from_video(f.name)


# ============================================================
# OpenAIAudioTranscriber tests
# ============================================================


class TestOpenAIAudioTranscriber:
    """Tests for the OpenAI Whisper transcriber."""

    def test_create_transcriber(self):
        """Test creating an OpenAI transcriber with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            transcriber = OpenAIAudioTranscriber()
            assert transcriber.model == "whisper-1"

    def test_create_transcriber_custom_model(self):
        """Test creating with custom model name."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            transcriber = OpenAIAudioTranscriber(model="whisper-1")
            assert transcriber.model == "whisper-1"

    def test_transcribe_video_mocked(self):
        """Test transcribe_video with mocked API response."""
        mock_response = MagicMock()
        mock_response.segments = [
            MagicMock(start=0.0, end=4.5, text=" Step one. Open the application."),
            MagicMock(start=4.5, end=9.0, text=" Step two. Click the button."),
        ]
        mock_response.language = "english"
        mock_response.duration = 9.0
        mock_response.text = "Step one. Open the application. Step two. Click the button."

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            transcriber = OpenAIAudioTranscriber()

            with patch.object(transcriber, "_client") as mock_client:
                mock_client.audio.transcriptions.create.return_value = mock_response

                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(b"fake audio content")
                    f.flush()
                    result = transcriber.transcribe(f.name)

                assert isinstance(result, TimestampedTranscription)
                assert len(result.segments) == 2
                assert result.segments[0].text == "Step one. Open the application."
                assert result.segments[0].start == 0.0
                assert result.segments[0].end == 4.5
                assert result.language == "english"

    def test_transcribe_calls_api_correctly(self):
        """Test that transcribe calls the OpenAI API with correct parameters."""
        mock_response = MagicMock()
        mock_response.segments = []
        mock_response.language = "english"
        mock_response.duration = 5.0
        mock_response.text = ""

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            transcriber = OpenAIAudioTranscriber()

            with patch.object(transcriber, "_client") as mock_client:
                mock_client.audio.transcriptions.create.return_value = mock_response

                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(b"fake audio")
                    f.flush()
                    transcriber.transcribe(f.name)

                # Verify API was called with correct params
                call_kwargs = mock_client.audio.transcriptions.create.call_args
                assert call_kwargs.kwargs["model"] == "whisper-1"
                assert call_kwargs.kwargs["response_format"] == "verbose_json"
                assert "segment" in call_kwargs.kwargs["timestamp_granularities"]

    def test_transcribe_with_language_hint(self):
        """Test transcription with explicit language hint."""
        mock_response = MagicMock()
        mock_response.segments = []
        mock_response.language = "japanese"
        mock_response.duration = 5.0
        mock_response.text = ""

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            transcriber = OpenAIAudioTranscriber()

            with patch.object(transcriber, "_client") as mock_client:
                mock_client.audio.transcriptions.create.return_value = mock_response

                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(b"fake audio")
                    f.flush()
                    transcriber.transcribe(f.name, language="ja")

                call_kwargs = mock_client.audio.transcriptions.create.call_args
                assert call_kwargs.kwargs["language"] == "ja"

    def test_transcribe_video_end_to_end_mocked(self):
        """Test the full transcribe_video flow with mocked extraction and API."""
        mock_transcription = TimestampedTranscription(
            segments=[TranscriptionSegment(start=0.0, end=5.0, text="Hello world.")],
            language="english",
            duration=5.0,
            full_text="Hello world.",
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            transcriber = OpenAIAudioTranscriber()

            with patch.object(transcriber, "transcribe", return_value=mock_transcription):
                with patch("aicapture.audio_transcriber.extract_audio_from_video") as mock_extract:
                    mock_extract.return_value = "/tmp/fake_audio.mp3"

                    result = transcriber.transcribe_video(str(TEST_VIDEO_PATH))

                    # extract_audio_from_video should have been called
                    mock_extract.assert_called_once()
                    assert isinstance(result, TimestampedTranscription)
                    assert result.full_text == "Hello world."


# ============================================================
# AzureOpenAIAudioTranscriber tests
# ============================================================


class TestAzureOpenAIAudioTranscriber:
    """Tests for the Azure OpenAI Whisper transcriber."""

    def test_create_transcriber(self):
        """Test creating an Azure OpenAI transcriber."""
        env = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_URL": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-11-01-preview",
        }
        with patch.dict("os.environ", env):
            transcriber = AzureOpenAIAudioTranscriber()
            assert transcriber.model == "whisper-1"

    def test_transcribe_mocked(self):
        """Test Azure transcription with mocked API."""
        mock_response = MagicMock()
        mock_response.segments = [
            MagicMock(start=0.0, end=3.0, text=" Test segment."),
        ]
        mock_response.language = "english"
        mock_response.duration = 3.0
        mock_response.text = "Test segment."

        env = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_URL": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-11-01-preview",
        }
        with patch.dict("os.environ", env):
            transcriber = AzureOpenAIAudioTranscriber()

            with patch.object(transcriber, "_client") as mock_client:
                mock_client.audio.transcriptions.create.return_value = mock_response

                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(b"fake audio")
                    f.flush()
                    result = transcriber.transcribe(f.name)

                assert isinstance(result, TimestampedTranscription)
                assert len(result.segments) == 1
                assert result.segments[0].text == "Test segment."


# ============================================================
# VidCapture integration tests
# ============================================================


class TestVidCaptureTranscriptionIntegration:
    """Tests for VidCapture with transcription enabled."""

    def test_video_config_transcription_fields(self):
        """Test that VideoConfig has transcription-related fields."""
        from aicapture.vid_capture import VideoConfig

        config = VideoConfig(
            enable_transcription=True,
            transcription_model="whisper-1",
            transcription_language="en",
        )
        assert config.enable_transcription is True
        assert config.transcription_model == "whisper-1"
        assert config.transcription_language == "en"

    def test_video_config_transcription_defaults(self):
        """Test default values for transcription config."""
        from aicapture.vid_capture import VideoConfig

        config = VideoConfig()
        assert config.enable_transcription is False
        assert config.transcription_model == "whisper-1"
        assert config.transcription_language is None

    def test_process_video_with_transcription(self, monkeypatch):
        """Test that process_video enriches prompt with transcription when enabled."""
        from aicapture.vid_capture import VidCapture, VideoConfig

        config = VideoConfig(enable_transcription=True)
        vid_capture = VidCapture(config=config, invalidate_cache=True)

        # Mock the transcription
        mock_transcription = TimestampedTranscription(
            segments=[
                TranscriptionSegment(start=0.0, end=5.0, text="Open the application."),
                TranscriptionSegment(start=5.0, end=10.0, text="Click settings."),
            ],
            language="en",
            duration=10.0,
            full_text="Open the application. Click settings.",
        )

        # Track what prompt is actually sent to the vision model
        captured_prompt = {}

        def mock_capture(prompt: str, images: Any, **kwargs: Any) -> str:
            captured_prompt["value"] = prompt
            return "Mocked analysis result"

        monkeypatch.setattr(vid_capture, "capture", mock_capture)

        with patch("aicapture.vid_capture.OpenAIAudioTranscriber") as MockTranscriber:
            mock_instance = MockTranscriber.return_value
            mock_instance.transcribe_video.return_value = mock_transcription

            vid_capture.process_video(str(TEST_VIDEO_PATH), "Describe this video")

        # The prompt sent to vision model should contain transcription
        assert "Video Audio Transcription" in captured_prompt["value"]
        assert "Open the application" in captured_prompt["value"]
        assert "Click settings" in captured_prompt["value"]
        assert "[00:00.00]" in captured_prompt["value"]
        # Original prompt should still be there
        assert "Describe this video" in captured_prompt["value"]

    def test_process_video_without_transcription(self, monkeypatch):
        """Test that process_video works normally when transcription is disabled."""
        from aicapture.vid_capture import VidCapture, VideoConfig

        config = VideoConfig(enable_transcription=False)
        vid_capture = VidCapture(config=config, invalidate_cache=True)

        captured_prompt = {}

        def mock_capture(prompt: str, images: Any, **kwargs: Any) -> str:
            captured_prompt["value"] = prompt
            return "Mocked result"

        monkeypatch.setattr(vid_capture, "capture", mock_capture)

        vid_capture.process_video(str(TEST_VIDEO_PATH), "Describe this video")

        # Prompt should NOT contain transcription context
        assert "Video Audio Transcription" not in captured_prompt["value"]
        assert captured_prompt["value"] == "Describe this video"

    @pytest.mark.asyncio
    async def test_process_video_async_with_transcription(self, monkeypatch):
        """Test async process_video enriches prompt with transcription."""
        from aicapture.vid_capture import VidCapture, VideoConfig

        config = VideoConfig(enable_transcription=True)
        vid_capture = VidCapture(config=config, invalidate_cache=True)

        mock_transcription = TimestampedTranscription(
            segments=[
                TranscriptionSegment(start=0.0, end=5.0, text="Step one."),
            ],
            language="en",
            duration=5.0,
            full_text="Step one.",
        )

        captured_prompt = {}

        async def mock_capture_async(prompt: str, images: Any, **kwargs: Any) -> str:
            captured_prompt["value"] = prompt
            return "Async mocked result"

        monkeypatch.setattr(vid_capture, "capture_async", mock_capture_async)

        with patch("aicapture.vid_capture.OpenAIAudioTranscriber") as MockTranscriber:
            mock_instance = MockTranscriber.return_value
            mock_instance.transcribe_video.return_value = mock_transcription

            await vid_capture.process_video_async(str(TEST_VIDEO_PATH), "Describe this video")

        assert "Video Audio Transcription" in captured_prompt["value"]
        assert "Step one" in captured_prompt["value"]

    def test_process_video_transcription_failure_graceful(self, monkeypatch):
        """Test that transcription failure doesn't break video processing."""
        from aicapture.vid_capture import VidCapture, VideoConfig

        config = VideoConfig(enable_transcription=True)
        vid_capture = VidCapture(config=config, invalidate_cache=True)

        def mock_capture(prompt: str, images: Any, **kwargs: Any) -> str:
            return "Result without transcription"

        monkeypatch.setattr(vid_capture, "capture", mock_capture)

        with patch("aicapture.vid_capture.OpenAIAudioTranscriber") as MockTranscriber:
            mock_instance = MockTranscriber.return_value
            mock_instance.transcribe_video.side_effect = Exception("API error")

            # Should NOT raise, should proceed without transcription
            result = vid_capture.process_video(str(TEST_VIDEO_PATH), "Describe this video")
            assert result == "Result without transcription"

    def test_cache_key_includes_transcription_config(self):
        """Test that cache key changes when transcription is enabled."""
        from aicapture.vid_capture import VidCapture, VideoConfig

        config_no_transcription = VideoConfig(enable_transcription=False)
        config_with_transcription = VideoConfig(enable_transcription=True)

        vid_no = VidCapture(config=config_no_transcription)
        vid_yes = VidCapture(config=config_with_transcription)

        key_no = vid_no._get_cache_key(str(TEST_VIDEO_PATH), "test prompt")
        key_yes = vid_yes._get_cache_key(str(TEST_VIDEO_PATH), "test prompt")

        # Cache keys should be different
        assert key_no != key_yes


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

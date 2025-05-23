"""
Vision Capture - A powerful Python library for extracting and analyzing content
using Vision Language Models.
"""

from aicapture.cache import FileCache, ImageCache, TwoLayerCache
from aicapture.settings import ImageQuality
from aicapture.vid_capture import VidCapture, VideoConfig, VideoValidationError
from aicapture.vision_models import (
    AnthropicAWSBedrockVisionModel,
    AnthropicVisionModel,
    AzureOpenAIVisionModel,
    GeminiVisionModel,
    OpenAIVisionModel,
    VisionModel,
    create_default_vision_model,
    is_vision_model_installed,
)
from aicapture.vision_parser import VisionParser

__version__ = "0.1.2"
__author__ = "Aitomatic, Inc."
__license__ = "Apache License 2.0"

__all__ = [
    # Main parser
    "VisionParser",
    # Vision models
    "VisionModel",
    "OpenAIVisionModel",
    "GeminiVisionModel",
    "AnthropicVisionModel",
    "AnthropicAWSBedrockVisionModel",
    "AzureOpenAIVisionModel",
    # Settings
    "ImageQuality",
    # Cache utilities
    "FileCache",
    "ImageCache",
    "TwoLayerCache",
    # Video capture
    "VidCapture",
    "VideoConfig",
    "VideoValidationError",
    "create_default_vision_model",
    "is_vision_model_installed",
]

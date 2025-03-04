"""
Vision Capture - A powerful Python library for extracting and analyzing content from PDF documents using Vision Language Models.
"""

from vision_capture.vision_parser import VisionParser
from vision_capture.vision_models import (
    VisionModel,
    OpenAIVisionModel,
    GeminiVisionModel,
    AnthropicVisionModel,
    AzureOpenAIVisionModel,
)
from vision_capture.settings import ImageQuality
from vision_capture.cache import FileCache, ImageCache, TwoLayerCache

__version__ = "0.1.0"
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
    "AzureOpenAIVisionModel",
    
    # Settings
    "ImageQuality",
    
    # Cache utilities
    "FileCache",
    "ImageCache",
    "TwoLayerCache",
]

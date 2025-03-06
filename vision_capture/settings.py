"""Vision model settings and configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


class VisionModelProvider:
    claude = "claude"
    openai = "openai"
    azure_openai = "azure-openai"
    gemini = "gemini"
    openai_alike = "openai-alike"


# Default vision model configuration
USE_VISION = os.getenv("USE_VISION", VisionModelProvider.openai).lower()

MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))

CLOUD_CACHE_BUCKET = os.getenv("CLOUD_CACHE_BUCKET", "your-bucket-name")


# Image quality settings
class ImageQuality:
    """Image quality settings for vision models."""

    LOW_RES = "low"  # 512px x 512px
    HIGH_RES = "high"  # max 768px x 2000px
    DEFAULT = HIGH_RES


@dataclass
class VisionModelConfig:
    """Base configuration for vision models."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    image_quality: str = ImageQuality.DEFAULT


class AnthropicVisionConfig(VisionModelConfig):
    """Configuration for Anthropic Claude Vision models."""

    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")


class OpenAIVisionConfig(VisionModelConfig):
    """Configuration for OpenAI GPT-4 Vision models."""

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "8000"))
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))


class GeminiVisionConfig(VisionModelConfig):
    """Configuration for Google Gemini Vision models."""

    api_key: str = os.getenv("GEMINI_API_KEY", "dummy")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


class AzureOpenAIVisionConfig(VisionModelConfig):
    """Configuration for Azure OpenAI Vision models."""

    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "dummy")
    model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    api_base: str = os.getenv(
        "AZURE_OPENAI_API_URL", "https://aitomaticjapaneast.openai.azure.com"
    )
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-01-preview")

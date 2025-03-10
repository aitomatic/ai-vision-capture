"""Vision model settings and configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


def mask_sensitive_string(value: Optional[str], visible_chars: int = 4) -> str:
    """Mask sensitive string showing only first few characters."""
    if not value:
        return ""
    return value[:visible_chars] + "*" * (len(value) - visible_chars)


def validate_required_config(name: str, value: Optional[str], provider: str) -> None:
    """Validate required configuration value."""
    if not value:
        error_msg = f"Missing required configuration '{name}' for {provider}"
        logger.error(error_msg)
        raise ValueError(error_msg)


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

    def validate(self, provider: str) -> None:
        """Validate configuration."""
        validate_required_config("model", self.model, provider)
        validate_required_config("api_key", self.api_key, provider)


class AnthropicVisionConfig(VisionModelConfig):
    """Configuration for Anthropic Claude Vision models."""

    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    def __post_init__(self) -> None:
        """Validate Anthropic configuration."""
        self.validate("Anthropic Claude")


class OpenAIVisionConfig(VisionModelConfig):
    """Configuration for OpenAI GPT-4 Vision models."""

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "8000"))
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    def __post_init__(self) -> None:
        """Validate OpenAI configuration."""
        self.validate("OpenAI")


class GeminiVisionConfig(VisionModelConfig):
    """Configuration for Google Gemini Vision models."""

    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def __post_init__(self) -> None:
        """Validate Gemini configuration."""
        self.validate("Google Gemini")


class AzureOpenAIVisionConfig(VisionModelConfig):
    """Configuration for Azure OpenAI Vision models."""

    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    api_base: str = os.getenv(
        "AZURE_OPENAI_API_URL", "https://aitomaticjapaneast.openai.azure.com"
    )
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-01-preview")

    def __post_init__(self) -> None:
        """Validate Azure OpenAI configuration."""
        self.validate("Azure OpenAI")
        validate_required_config("api_version", self.api_version, "Azure OpenAI")
        validate_required_config("api_base", self.api_base, "Azure OpenAI")

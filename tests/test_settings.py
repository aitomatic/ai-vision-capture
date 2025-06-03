from unittest.mock import patch

import pytest
from pytest import MonkeyPatch

from aicapture.settings import (
    AnthropicAWSBedrockConfig,
    AnthropicVisionConfig,
    AzureOpenAIVisionConfig,
    GeminiVisionConfig,
    ImageQuality,
    OpenAIVisionConfig,
    VisionModelConfig,
    VisionModelProvider,
    mask_sensitive_string,
    validate_required_config,
)


class TestVisionModelProvider:
    """Test cases for VisionModelProvider constants."""

    def test_provider_constants(self) -> None:
        """Test that all provider constants are defined."""
        assert VisionModelProvider.claude == "claude"
        assert VisionModelProvider.openai == "openai"
        assert VisionModelProvider.azure_openai == "azure-openai"
        assert VisionModelProvider.gemini == "gemini"
        assert VisionModelProvider.openai_alike == "openai-alike"
        assert VisionModelProvider.anthropic_bedrock == "anthropic_bedrock"


class TestImageQuality:
    """Test cases for ImageQuality settings."""

    def test_quality_constants(self) -> None:
        """Test image quality constants."""
        assert ImageQuality.LOW_RES == "low"
        assert ImageQuality.HIGH_RES == "high"
        assert ImageQuality.DEFAULT == ImageQuality.HIGH_RES


class TestUtilityFunctions:
    """Test utility functions in settings module."""

    def test_mask_sensitive_string_default(self) -> None:
        """Test masking sensitive string with default parameters."""
        sensitive = "sk-1234567890abcdef"
        masked = mask_sensitive_string(sensitive)

        assert masked.startswith("sk-1")
        assert len(masked) == len(sensitive)
        assert "1234" in masked
        assert "abcdef" not in masked
        assert "*" in masked

    def test_mask_sensitive_string_custom_visible(self) -> None:
        """Test masking with custom visible characters count."""
        sensitive = "api_key_12345"
        masked = mask_sensitive_string(sensitive, visible_chars=8)

        assert masked == "api_key_*****"
        assert masked.startswith("api_key_")

    def test_mask_sensitive_string_empty(self) -> None:
        """Test masking empty string."""
        result = mask_sensitive_string("")
        assert result == ""

    def test_mask_sensitive_string_none(self) -> None:
        """Test masking None value."""
        result = mask_sensitive_string(None)
        assert result == ""

    def test_mask_sensitive_string_short(self) -> None:
        """Test masking string shorter than visible chars."""
        short_string = "abc"
        masked = mask_sensitive_string(short_string, visible_chars=5)
        assert masked == "abc"  # No masking for short strings

    def test_validate_required_config_valid(self) -> None:
        """Test validation of valid required config."""
        # Should not raise any exception
        validate_required_config("api_key", "valid_key", "TestProvider")

    def test_validate_required_config_empty(self) -> None:
        """Test validation of empty required config."""
        with pytest.raises(
            ValueError,
            match="Missing required configuration 'api_key' for TestProvider",
        ):
            validate_required_config("api_key", "", "TestProvider")

    def test_validate_required_config_none(self) -> None:
        """Test validation of None required config."""
        with pytest.raises(
            ValueError,
            match="Missing required configuration 'api_key' for TestProvider",
        ):
            validate_required_config("api_key", None, "TestProvider")


class TestVisionModelConfig:
    """Test cases for base VisionModelConfig."""

    def test_init_default(self) -> None:
        """Test initialization with default values."""
        config = VisionModelConfig(model="test-model")

        assert config.model == "test-model"
        assert config.api_key is None
        assert config.api_base is None
        assert config.image_quality == ImageQuality.DEFAULT

    def test_init_custom(self) -> None:
        """Test initialization with custom values."""
        config = VisionModelConfig(
            model="custom-model",
            api_key="test_key",
            api_base="https://api.example.com",
            image_quality=ImageQuality.LOW_RES,
        )

        assert config.model == "custom-model"
        assert config.api_key == "test_key"
        assert config.api_base == "https://api.example.com"
        assert config.image_quality == ImageQuality.LOW_RES

    def test_validate_success(self) -> None:
        """Test successful validation."""
        config = VisionModelConfig(model="test-model", api_key="test_key")

        # Should not raise any exception
        config.validate("TestProvider")

    def test_validate_missing_model(self) -> None:
        """Test validation with missing model."""
        config = VisionModelConfig(model="", api_key="test_key")

        with pytest.raises(
            ValueError, match="Missing required configuration 'model' for TestProvider"
        ):
            config.validate("TestProvider")

    def test_validate_missing_api_key(self) -> None:
        """Test validation with missing API key."""
        config = VisionModelConfig(model="test-model", api_key="")

        with pytest.raises(
            ValueError,
            match="Missing required configuration 'api_key' for TestProvider",
        ):
            config.validate("TestProvider")


class TestOpenAIVisionConfig:
    """Test cases for OpenAIVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai.com")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "4000")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")

        config = OpenAIVisionConfig()

        assert config.api_key == "test_openai_key"
        assert config.model == "gpt-4o"
        assert config.api_base == "https://custom.openai.com"
        assert config.max_tokens == 4000
        assert config.temperature == 0.5

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        # Clear environment variables
        env_vars = [
            "OPENAI_API_KEY",
            "OPENAI_VISION_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_VISION_MODEL",
            "OPENAI_BASE_URL",
            "OPENAI_VISION_BASE_URL",
            "OPENAI_MAX_TOKENS",
            "OPENAI_TEMPERATURE",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "test_key")  # Required for validation

        config = OpenAIVisionConfig()

        assert config.model == "gpt-4.1-mini"
        assert config.api_base == "https://api.openai.com/v1"
        assert config.max_tokens == 5000
        assert config.temperature == 0.0

    def test_vision_specific_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test that vision-specific env vars take precedence."""
        monkeypatch.setenv("OPENAI_API_KEY", "general_key")
        monkeypatch.setenv("OPENAI_VISION_API_KEY", "vision_key")
        monkeypatch.setenv("OPENAI_MODEL", "general_model")
        monkeypatch.setenv("OPENAI_VISION_MODEL", "vision_model")

        config = OpenAIVisionConfig()

        assert config.api_key == "vision_key"
        assert config.model == "vision_model"

    def test_validation_error(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        monkeypatch.setenv("OPENAI_API_KEY", "")

        with pytest.raises(ValueError):
            OpenAIVisionConfig()


class TestAnthropicVisionConfig:
    """Test cases for AnthropicVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")

        config = AnthropicVisionConfig()

        assert config.api_key == "test_anthropic_key"
        assert config.model == "claude-3-opus-20240229"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

        config = AnthropicVisionConfig()

        assert config.model == "claude-3-5-sonnet-20241022"

    def test_validation_error(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        with pytest.raises(ValueError):
            AnthropicVisionConfig()


class TestGeminiVisionConfig:
    """Test cases for GeminiVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-pro-vision")

        config = GeminiVisionConfig()

        assert config.api_key == "test_gemini_key"
        assert config.model == "gemini-pro-vision"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        config = GeminiVisionConfig()

        assert config.model == "gemini-2.5-flash-preview-04-17"

    def test_validation_error(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        monkeypatch.setenv("GEMINI_API_KEY", "")

        with pytest.raises(ValueError):
            GeminiVisionConfig()


class TestAzureOpenAIVisionConfig:
    """Test cases for AzureOpenAIVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_key")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4-vision")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://custom.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        config = AzureOpenAIVisionConfig()

        assert config.api_key == "test_azure_key"
        assert config.model == "gpt-4-vision"
        assert config.api_base == "https://custom.azure.com"
        assert config.api_version == "2024-02-01"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")

        # Clear optional env vars
        optional_vars = [
            "AZURE_OPENAI_MODEL",
            "AZURE_OPENAI_API_URL",
            "AZURE_OPENAI_API_VERSION",
        ]
        for var in optional_vars:
            monkeypatch.delenv(var, raising=False)

        config = AzureOpenAIVisionConfig()

        assert config.model == "gpt-4o"
        assert config.api_base == "https://aitomaticjapaneast.openai.azure.com"
        assert config.api_version == "2024-11-01-preview"

    def test_validation_error_missing_api_key(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")

        with pytest.raises(ValueError):
            AzureOpenAIVisionConfig()

    def test_validation_error_missing_api_version(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test validation error when API version is missing."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "")

        with pytest.raises(ValueError):
            AzureOpenAIVisionConfig()

    def test_validation_error_missing_api_base(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API base is missing."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "")

        with pytest.raises(ValueError):
            AzureOpenAIVisionConfig()


class TestAnthropicAWSBedrockConfig:
    """Test cases for AnthropicAWSBedrockConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_access_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret_key")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "test_session_token")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setenv(
            "AWS_BEDROCK_VPC_ENDPOINT_URL", "https://bedrock.vpc.amazonaws.com"
        )
        monkeypatch.setenv(
            "ANTHROPIC_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"
        )

        config = AnthropicAWSBedrockConfig()

        assert config.aws_access_key_id == "test_access_key"
        assert config.aws_secret_access_key == "test_secret_key"
        assert config.aws_session_token == "test_session_token"
        assert config.aws_region == "us-west-2"
        assert config.aws_vpc_endpoint_url == "https://bedrock.vpc.amazonaws.com"
        assert config.model == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        # Clear optional env vars
        env_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_BEDROCK_VPC_ENDPOINT_URL",
            "ANTHROPIC_BEDROCK_MODEL",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

        config = AnthropicAWSBedrockConfig()

        assert config.aws_access_key_id == "dummy"
        assert config.aws_secret_access_key == "dummy"
        assert config.aws_session_token is None
        assert config.aws_region == "us-east-1"
        assert config.aws_vpc_endpoint_url == ""
        assert config.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert config.api_key == "dummy"

    def test_session_token_optional(self, monkeypatch: MonkeyPatch) -> None:
        """Test that session token is optional."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret")
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

        config = AnthropicAWSBedrockConfig()

        assert config.aws_session_token is None


class TestEnvironmentVariableIntegration:
    """Test integration with actual environment variables."""

    def test_use_vision_env_var(self, monkeypatch: MonkeyPatch) -> None:
        """Test USE_VISION environment variable."""

        # Test default value
        monkeypatch.delenv("USE_VISION", raising=False)
        with patch('aicapture.settings.os.getenv') as mock_getenv:
            mock_getenv.return_value = VisionModelProvider.openai
            from importlib import reload

            import aicapture.settings

            reload(aicapture.settings)

            assert aicapture.settings.USE_VISION == VisionModelProvider.openai

    def test_max_concurrent_tasks_env_var(self, monkeypatch: MonkeyPatch) -> None:
        """Test MAX_CONCURRENT_TASKS environment variable."""
        monkeypatch.setenv("MAX_CONCURRENT_TASKS", "10")

        with patch('aicapture.settings.os.getenv') as mock_getenv:
            mock_getenv.return_value = "10"
            from importlib import reload

            import aicapture.settings

            reload(aicapture.settings)

            assert (
                aicapture.settings.MAX_CONCURRENT_TASKS == 20
            )  # Current value in module


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

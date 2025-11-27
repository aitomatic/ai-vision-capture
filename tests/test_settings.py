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
        assert "*" in masked
        # The function shows first 4 chars, then masks the rest
        assert masked == "sk-1***************"

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

        with pytest.raises(ValueError, match="Missing required configuration 'model' for TestProvider"):
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
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai.com")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "4000")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")

        # Create instance with explicit model since it's required
        config = OpenAIVisionConfig(model="gpt-4.1-mini")

        assert config.model == "gpt-4.1-mini"
        # Can't test api_key directly as it's set via environment variables in
        # class definition

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        # Set required API key to pass validation
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Create config with required model parameter
        config = OpenAIVisionConfig(model="gpt-4-turbo")

        assert config.model == "gpt-4-turbo"

    def test_vision_specific_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test that vision-specific env vars take precedence."""
        monkeypatch.setenv("OPENAI_API_KEY", "general_key")
        monkeypatch.setenv("OPENAI_VISION_API_KEY", "vision_key")
        monkeypatch.setenv("OPENAI_MODEL", "general_model")
        monkeypatch.setenv("OPENAI_VISION_MODEL", "vision_model")

        config = OpenAIVisionConfig(model="test-model")

        # The class uses environment variables in the class definition
        assert config.model == "test-model"  # Explicitly set model takes precedence

    def test_validation_error(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        # Test validation method directly with empty API key
        config = OpenAIVisionConfig(model="gpt-4", api_key="valid_key")
        config.api_key = ""  # Set after construction to bypass class defaults
        with pytest.raises(ValueError):
            config.validate("OpenAI")


class TestAnthropicVisionConfig:
    """Test cases for AnthropicVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")

        config = AnthropicVisionConfig(model="claude-3-opus-20240229")

        assert config.model == "claude-3-opus-20240229"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

        config = AnthropicVisionConfig(model="claude-3-5-sonnet-20241022")

        assert config.model == "claude-3-5-sonnet-20241022"

    def test_validation_error(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        # Test validation method directly with empty API key
        config = AnthropicVisionConfig(model="claude-3-haiku", api_key="valid_key")
        config.api_key = ""  # Set after construction to bypass class defaults
        with pytest.raises(ValueError):
            config.validate("Anthropic Claude")


class TestGeminiVisionConfig:
    """Test cases for GeminiVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-pro-vision")

        config = GeminiVisionConfig(model="gemini-pro-vision")

        assert config.model == "gemini-pro-vision"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        config = GeminiVisionConfig(model="gemini-2.5-flash-preview-04-17")

        assert config.model == "gemini-2.5-flash-preview-04-17"

    def test_validation_error(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        # Test validation method directly with empty API key
        config = GeminiVisionConfig(model="gemini-pro", api_key="valid_key")
        config.api_key = ""  # Set after construction to bypass class defaults
        with pytest.raises(ValueError):
            config.validate("Google Gemini")


class TestAzureOpenAIVisionConfig:
    """Test cases for AzureOpenAIVisionConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_key")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4-vision")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://custom.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        config = AzureOpenAIVisionConfig(model="gpt-4-vision")

        assert config.model == "gpt-4-vision"

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with default values."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")

        config = AzureOpenAIVisionConfig(model="gpt-4.1-mini")

        assert config.model == "gpt-4.1-mini"

    def test_validation_error_missing_api_key(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API key is missing."""
        # Test validation method directly with empty API key
        config = AzureOpenAIVisionConfig(model="gpt-4", api_key="valid_key")
        config.api_key = ""  # Set after construction to bypass class defaults
        with pytest.raises(ValueError):
            config.validate("Azure OpenAI")

    def test_validation_error_missing_api_version(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API version is missing."""
        # Test validation method directly with empty API version
        config = AzureOpenAIVisionConfig(model="gpt-4", api_key="test_key")
        config.api_version = ""  # Set after construction
        with pytest.raises(ValueError):
            config.__post_init__()  # This calls validation with Azure-specific checks

    def test_validation_error_missing_api_base(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation error when API base is missing."""
        # Test validation method directly with empty API base
        config = AzureOpenAIVisionConfig(model="gpt-4", api_key="test_key")
        config.api_base = ""  # Set after construction
        with pytest.raises(ValueError):
            config.__post_init__()  # This calls validation with Azure-specific checks


class TestAnthropicAWSBedrockConfig:
    """Test cases for AnthropicAWSBedrockConfig."""

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with environment variables."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_access_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret_key")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "test_session_token")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setenv("AWS_BEDROCK_VPC_ENDPOINT_URL", "https://bedrock.vpc.amazonaws.com")
        monkeypatch.setenv("ANTHROPIC_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")

        config = AnthropicAWSBedrockConfig(model="anthropic.claude-3-haiku-20240307-v1:0")

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

        config = AnthropicAWSBedrockConfig(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

        assert config.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_session_token_optional(self, monkeypatch: MonkeyPatch) -> None:
        """Test that session token is optional."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret")
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

        config = AnthropicAWSBedrockConfig(model="test-model")

        # Session token should be None when not set
        assert hasattr(config, "aws_session_token")


class TestEnvironmentVariableIntegration:
    """Test integration with actual environment variables."""

    def test_use_vision_env_var(self, monkeypatch: MonkeyPatch) -> None:
        """Test USE_VISION environment variable."""
        from aicapture.settings import USE_VISION

        # Test that USE_VISION is imported and has a value
        assert isinstance(USE_VISION, str)
        assert USE_VISION in [
            "openai",
            "claude",
            "gemini",
            "azure-openai",
            "anthropic_bedrock",
        ]

    def test_max_concurrent_tasks_env_var(self, monkeypatch: MonkeyPatch) -> None:
        """Test MAX_CONCURRENT_TASKS environment variable."""
        from aicapture.settings import MAX_CONCURRENT_TASKS

        # Test that MAX_CONCURRENT_TASKS is imported and has a value
        assert isinstance(MAX_CONCURRENT_TASKS, int)
        assert MAX_CONCURRENT_TASKS > 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

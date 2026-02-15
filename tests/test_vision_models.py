import base64
from io import BytesIO
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image
from pytest import MonkeyPatch

from aicapture.vision_models import (
    AnthropicVisionModel,
    AzureOpenAIVisionModel,
    OpenAIVisionModel,
    VisionModel,
    create_default_vision_model,
    is_vision_model_installed,
)


@pytest.fixture
def test_image_path() -> str:
    """Create a simple test image for testing."""
    # Use the existing test image
    return "tests/sample/images/logic.png"


@pytest.fixture
def test_image_base64() -> str:
    """Create base64 encoded test image."""
    # Create a simple 10x10 RGB image
    img = Image.new("RGB", (10, 10), color="red")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


@pytest.fixture
def mock_messages() -> List[Dict[str, Any]]:
    """Create mock messages for text processing."""
    return [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Test response"},
    ]


class TestVisionModel:
    """Test cases for the abstract VisionModel base class."""

    def test_vision_model_is_abstract(self) -> None:
        """Test that VisionModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VisionModel()


class TestCreateDefaultVisionModel:
    """Test cases for create_default_vision_model function."""

    def test_create_anthropic_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Anthropic vision model."""
        # Mock the USE_VISION at module level
        with patch("aicapture.vision_models.USE_VISION", "claude"):
            with patch("aicapture.vision_models.AnthropicVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_create_openai_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating OpenAI vision model."""
        with patch("aicapture.vision_models.USE_VISION", "openai"):
            with patch("aicapture.vision_models.OpenAIVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_create_gemini_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Gemini vision model."""
        with patch("aicapture.vision_models.USE_VISION", "gemini"):
            with patch("aicapture.vision_models.GeminiVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_create_azure_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Azure OpenAI vision model."""
        with patch("aicapture.vision_models.USE_VISION", "azure-openai"):
            with patch("aicapture.vision_models.AzureOpenAIVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_create_bedrock_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Anthropic AWS Bedrock vision model."""
        with patch("aicapture.vision_models.USE_VISION", "anthropic_bedrock"):
            with patch("aicapture.vision_models.AnthropicAWSBedrockVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_unsupported_model_type(self, monkeypatch: MonkeyPatch) -> None:
        """Test error handling for unsupported model type with no API keys."""
        # Clear all API keys to ensure AutoDetectVisionModel fails
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_VISION_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("aicapture.vision_models.USE_VISION", "unsupported_model"):
            with pytest.raises(ValueError, match="No valid API key found for any vision model provider"):
                create_default_vision_model()

    def test_create_model_with_exception(self, monkeypatch: MonkeyPatch) -> None:
        """Test error handling when model creation fails."""
        with patch("aicapture.vision_models.USE_VISION", "openai"):
            with patch(
                "aicapture.vision_models.OpenAIVisionModel",
                side_effect=Exception("Model creation failed"),
            ):
                with pytest.raises(Exception, match="Model creation failed"):
                    create_default_vision_model()

    def test_auto_detect_gemini(self, monkeypatch: MonkeyPatch) -> None:
        """Test AutoDetectVisionModel with Gemini API key."""
        # Clear other API keys
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_VISION_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")

        with patch("aicapture.vision_models.USE_VISION", "auto"):
            with patch("aicapture.vision_models.GeminiVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_auto_detect_openai(self, monkeypatch: MonkeyPatch) -> None:
        """Test AutoDetectVisionModel with OpenAI API key."""
        # Clear other API keys, but set OpenAI
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")

        with patch("aicapture.vision_models.USE_VISION", "auto"):
            with patch("aicapture.vision_models.OpenAIVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_auto_detect_azure(self, monkeypatch: MonkeyPatch) -> None:
        """Test AutoDetectVisionModel with Azure API key."""
        # Clear other API keys, but set Azure
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_VISION_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_key")

        with patch("aicapture.vision_models.USE_VISION", "auto"):
            with patch("aicapture.vision_models.AzureOpenAIVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance

    def test_auto_detect_anthropic(self, monkeypatch: MonkeyPatch) -> None:
        """Test AutoDetectVisionModel with Anthropic API key."""
        # Clear other API keys, but set Anthropic
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_VISION_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")

        with patch("aicapture.vision_models.USE_VISION", "auto"):
            with patch("aicapture.vision_models.AnthropicVisionModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance
                result = create_default_vision_model()
                mock_model.assert_called_once()
                assert result == mock_instance


class TestIsVisionModelInstalled:
    """Test cases for is_vision_model_installed function."""

    def test_openai_installed(self) -> None:
        """Test that OpenAI is detected as installed."""
        with patch("aicapture.vision_models.USE_VISION", "openai"):
            result = is_vision_model_installed()
            assert result is True

    def test_anthropic_installed(self) -> None:
        """Test that Anthropic is detected as installed."""
        with patch("aicapture.vision_models.USE_VISION", "claude"):
            result = is_vision_model_installed()
            assert result is True

    def test_model_not_installed(self) -> None:
        """Test that unsupported models are detected as not installed."""
        with patch("aicapture.vision_models.USE_VISION", "unsupported_model"):
            result = is_vision_model_installed()
            assert result is False

    def test_unknown_model(self) -> None:
        """Test handling of unknown model type."""
        with patch("aicapture.vision_models.USE_VISION", "random_unknown_model"):
            result = is_vision_model_installed()
            assert result is False


class TestAnthropicVisionModel:
    """Test cases for AnthropicVisionModel."""

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test Anthropic model initialization with defaults."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")

        with patch("aicapture.vision_models.anthropic.Anthropic"):
            # Test with explicit values since defaults are set at import time
            model = AnthropicVisionModel(model="claude-3-5-sonnet-20241022", api_key="test_key")
            # Check that we get the expected model and api key
            assert model.model == "claude-3-5-sonnet-20241022"
            assert model.api_key == "test_key"

    def test_init_with_custom_params(self, monkeypatch: MonkeyPatch) -> None:
        """Test Anthropic model initialization with custom parameters."""
        with patch("aicapture.vision_models.anthropic.Anthropic"):
            model = AnthropicVisionModel(model="claude-3-haiku-20240307", api_key="custom_key")
            assert model.model == "claude-3-haiku-20240307"
            assert model.api_key == "custom_key"

    def test_api_key_required(self) -> None:
        """Test that API key is required."""
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicVisionModel(model="claude-3-haiku", api_key="")

    def test_optimize_image(self, monkeypatch: MonkeyPatch) -> None:
        """Test image optimization."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")

        with patch("aicapture.vision_models.anthropic.Anthropic"):
            model = AnthropicVisionModel()

            # Create a test image that needs optimization
            large_image = Image.new("RGB", (2000, 2000), color="red")
            optimized = model._optimize_image(large_image)

            # Should be optimized to the optimal size
            assert max(optimized.size) <= model.OPTIMAL_IMAGE_SIZE

    def test_convert_image_to_base64(self) -> None:
        """Test static method for image conversion."""
        test_image = Image.new("RGB", (10, 10), color="blue")
        base64_str, media_type = VisionModel.convert_image_to_base64(test_image)

        assert isinstance(base64_str, str)
        assert media_type == "image/jpeg"
        assert len(base64_str) > 0

        # Test that it's valid base64
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_process_image_async_mocked(self, monkeypatch: MonkeyPatch) -> None:
        """Test async image processing with mocked client."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")

        # Mock the anthropic client and response
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock()]
        mock_response.content[0].text = "Anthropic image response"
        mock_response.usage = AsyncMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        # Mock both the class and the property
        with patch("aicapture.vision_models.anthropic.AsyncAnthropic", return_value=mock_client):
            model = AnthropicVisionModel()
            # Override the aclient property to return our mock
            model._aclient = mock_client

            # Create a simple test image
            test_image = Image.new("RGB", (100, 100), color="green")
            result = await model.process_image_async(test_image, "Describe this image")

            assert result == "Anthropic image response"
            mock_client.messages.create.assert_called_once()


class TestVisionModelUtilities:
    """Test utility methods and error handling."""

    def test_log_token_usage(self, monkeypatch: MonkeyPatch) -> None:
        """Test token usage logging."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")

        with patch("aicapture.vision_models.anthropic.Anthropic"):
            model = AnthropicVisionModel()

            usage_data = {"input_tokens": 100, "output_tokens": 50}
            model.log_token_usage(usage_data)

            assert model.last_token_usage == usage_data

    def test_image_size_validation(self, monkeypatch: MonkeyPatch) -> None:
        """Test image size validation."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")

        with patch("aicapture.vision_models.anthropic.Anthropic"):
            model = AnthropicVisionModel()

            # Create an image that's too large
            huge_image = Image.new("RGB", (10000, 10000), color="red")

            with pytest.raises(ValueError, match="Image dimensions exceed maximum"):
                model._optimize_image(huge_image)

    def test_batch_image_optimization(self, monkeypatch: MonkeyPatch) -> None:
        """Test batch image optimization."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")

        with patch("aicapture.vision_models.anthropic.Anthropic"):
            model = AnthropicVisionModel()

            # Create an image for batch processing
            batch_image = Image.new("RGB", (1500, 1500), color="blue")
            optimized = model._optimize_image(batch_image, is_batch=True)

            # Should be within batch size limits
            assert optimized.size[0] <= model.MAX_BATCH_IMAGE_SIZE[0]
            assert optimized.size[1] <= model.MAX_BATCH_IMAGE_SIZE[1]


class TestVisionModelErrorHandling:
    """Test error handling across vision models."""

    def test_missing_api_key_anthropic(self) -> None:
        """Test error when Anthropic API key is missing."""
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicVisionModel(api_key="")

    def test_missing_api_key_none(self) -> None:
        """Test error when API key is None."""
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicVisionModel(api_key=None)

    def test_invalid_model_name(self) -> None:
        """Test initialization with invalid model name."""
        # Should still initialize - validation happens at API call time
        model = AnthropicVisionModel(model="invalid-model", api_key="test_key")
        assert model.model == "invalid-model"


class TestOpenAIReasoningModels:
    """Test cases for OpenAI GPT-5 reasoning models."""

    def test_detect_gpt5_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test detection of GPT-5 reasoning models."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Test GPT-5 models are detected as reasoning models
        model_gpt5 = OpenAIVisionModel(model="gpt-5")
        assert model_gpt5.is_reasoning_model is True

        model_gpt51 = OpenAIVisionModel(model="gpt-5.1")
        assert model_gpt51.is_reasoning_model is True

        model_gpt52 = OpenAIVisionModel(model="gpt-5.2")
        assert model_gpt52.is_reasoning_model is True

        # Test o1 and o3 models are also detected
        model_o1 = OpenAIVisionModel(model="o1")
        assert model_o1.is_reasoning_model is True

        model_o3 = OpenAIVisionModel(model="o3-mini")
        assert model_o3.is_reasoning_model is True

    def test_detect_non_reasoning_models(self, monkeypatch: MonkeyPatch) -> None:
        """Test that GPT-4 and older models are not detected as reasoning models."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        model_gpt4 = OpenAIVisionModel(model="gpt-4.1")
        assert model_gpt4.is_reasoning_model is False

        model_gpt4_turbo = OpenAIVisionModel(model="gpt-4-turbo")
        assert model_gpt4_turbo.is_reasoning_model is False

    def test_reasoning_model_initialization(self, monkeypatch: MonkeyPatch) -> None:
        """Test GPT-5 model initialization with reasoning parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        model = OpenAIVisionModel(
            model="gpt-5.2",
            reasoning_effort="medium",
            max_completion_tokens=5000,
            temperature=0.0,
        )

        assert model.is_reasoning_model is True
        assert model.reasoning_effort == "medium"
        assert model.max_completion_tokens == 5000
        assert model.temperature == 0.0

    def test_reasoning_model_default_parameters(self, monkeypatch: MonkeyPatch) -> None:
        """Test GPT-5 model with default parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        model = OpenAIVisionModel(model="gpt-5.2")

        assert model.is_reasoning_model is True
        assert model.reasoning_effort is None  # Default is None
        assert model.max_completion_tokens > 0

    @pytest.mark.asyncio
    async def test_reasoning_model_process_image_async_parameters(self, monkeypatch: MonkeyPatch) -> None:
        """Test that GPT-5 uses correct parameters in API calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Mock the async client
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "GPT-5 response"
        mock_response.usage = AsyncMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.AsyncOpenAI", return_value=mock_client):
            model = OpenAIVisionModel(model="gpt-5.2", reasoning_effort="high", max_completion_tokens=3000)
            model._aclient = mock_client

            test_image = Image.new("RGB", (100, 100), color="red")
            result = await model.process_image_async(test_image, "Analyze this image")

            assert result == "GPT-5 response"

            # Verify the correct parameters were passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-5.2"
            assert call_args[1]["max_completion_tokens"] == 3000
            assert call_args[1]["reasoning_effort"] == "high"
            # Temperature should NOT be included for reasoning models
            assert "temperature" not in call_args[1]
            # max_tokens should NOT be used (only max_completion_tokens)
            assert "max_tokens" not in call_args[1]

    @pytest.mark.asyncio
    async def test_reasoning_model_with_none_effort(self, monkeypatch: MonkeyPatch) -> None:
        """Test that temperature is included when reasoning_effort is 'none'."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "GPT-5 instant response"
        mock_response.usage = AsyncMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.AsyncOpenAI", return_value=mock_client):
            model = OpenAIVisionModel(model="gpt-5.2", reasoning_effort="none", temperature=0.5)
            model._aclient = mock_client

            test_image = Image.new("RGB", (100, 100), color="blue")
            result = await model.process_image_async(test_image, "Describe this")

            assert result == "GPT-5 instant response"

            # Verify temperature IS included when reasoning_effort="none"
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["reasoning_effort"] == "none"
            assert call_args[1]["temperature"] == 0.5

    def test_traditional_model_parameters(self, monkeypatch: MonkeyPatch) -> None:
        """Test that GPT-4 still uses traditional parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "GPT-4 response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.OpenAI", return_value=mock_client):
            model = OpenAIVisionModel(model="gpt-4.1", max_tokens=4000, temperature=0.7)
            model._client = mock_client

            test_image = Image.new("RGB", (100, 100), color="green")
            result = model.process_image(test_image, "What is this?")

            assert result == "GPT-4 response"

            # Verify traditional parameters are used for GPT-4
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4.1"
            assert call_args[1]["max_tokens"] == 4000
            assert call_args[1]["temperature"] == 0.7
            # Reasoning parameters should NOT be included
            assert "max_completion_tokens" not in call_args[1]
            assert "reasoning_effort" not in call_args[1]

    @pytest.mark.asyncio
    async def test_process_text_async_reasoning_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test process_text_async with reasoning models."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Text response"
        mock_response.usage = AsyncMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 75
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.AsyncOpenAI", return_value=mock_client):
            model = OpenAIVisionModel(model="gpt-5.1", reasoning_effort="low")
            model._aclient = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            result = await model.process_text_async(messages)

            assert result == "Text response"

            # Verify correct parameters
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["reasoning_effort"] == "low"
            assert "max_completion_tokens" in call_args[1]
            assert "temperature" not in call_args[1]


class TestAzureOpenAIReasoningModels:
    """Test cases for Azure OpenAI GPT-5 reasoning models."""

    def test_azure_inherits_reasoning_detection(self, monkeypatch: MonkeyPatch) -> None:
        """Test that Azure models inherit reasoning model detection from OpenAI."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://test.openai.azure.com")

        # Test GPT-5 models are detected as reasoning models
        model_gpt5 = AzureOpenAIVisionModel(model="gpt-5", api_key="test_key", api_base="https://test.openai.azure.com")
        assert model_gpt5.is_reasoning_model is True

        model_gpt52 = AzureOpenAIVisionModel(
            model="gpt-5.2", api_key="test_key", api_base="https://test.openai.azure.com"
        )
        assert model_gpt52.is_reasoning_model is True

        # Test GPT-4 is not a reasoning model
        model_gpt4 = AzureOpenAIVisionModel(
            model="gpt-4.1", api_key="test_key", api_base="https://test.openai.azure.com"
        )
        assert model_gpt4.is_reasoning_model is False

    def test_azure_reasoning_model_initialization(self, monkeypatch: MonkeyPatch) -> None:
        """Test Azure GPT-5 model initialization with reasoning parameters."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://test.openai.azure.com")

        model = AzureOpenAIVisionModel(
            model="gpt-5.2",
            api_key="test_key",
            api_base="https://test.openai.azure.com",
            reasoning_effort="high",
            max_completion_tokens=3000,
        )

        assert model.is_reasoning_model is True
        assert model.reasoning_effort == "high"
        assert model.max_completion_tokens == 3000
        assert model.api_version  # Should have API version

    @pytest.mark.asyncio
    async def test_azure_reasoning_model_process_image_async(self, monkeypatch: MonkeyPatch) -> None:
        """Test that Azure GPT-5 uses correct parameters in API calls."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://test.openai.azure.com")

        # Mock the async client
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Azure GPT-5 response"
        mock_response.usage = AsyncMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.AsyncAzureOpenAI", return_value=mock_client):
            model = AzureOpenAIVisionModel(
                model="gpt-5.2",
                api_key="test_key",
                api_base="https://test.openai.azure.com",
                reasoning_effort="medium",
                max_completion_tokens=4000,
            )
            model._aclient = mock_client

            test_image = Image.new("RGB", (100, 100), color="blue")
            result = await model.process_image_async(test_image, "Analyze this image")

            assert result == "Azure GPT-5 response"

            # Verify the correct parameters were passed
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-5.2"
            assert call_args[1]["max_completion_tokens"] == 4000
            assert call_args[1]["reasoning_effort"] == "medium"
            # Temperature should NOT be included for reasoning models
            assert "temperature" not in call_args[1]
            # max_tokens should NOT be used
            assert "max_tokens" not in call_args[1]

    @pytest.mark.asyncio
    async def test_azure_reasoning_model_instant_mode(self, monkeypatch: MonkeyPatch) -> None:
        """Test that Azure GPT-5 includes temperature when reasoning_effort='none'."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://test.openai.azure.com")

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Azure GPT-5 instant response"
        mock_response.usage = AsyncMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.AsyncAzureOpenAI", return_value=mock_client):
            model = AzureOpenAIVisionModel(
                model="gpt-5.2",
                api_key="test_key",
                api_base="https://test.openai.azure.com",
                reasoning_effort="none",
                temperature=0.7,
            )
            model._aclient = mock_client

            test_image = Image.new("RGB", (100, 100), color="green")
            result = await model.process_image_async(test_image, "Describe this")

            assert result == "Azure GPT-5 instant response"

            # Verify temperature IS included when reasoning_effort="none"
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["reasoning_effort"] == "none"
            assert call_args[1]["temperature"] == 0.7

    def test_azure_traditional_model_parameters(self, monkeypatch: MonkeyPatch) -> None:
        """Test that Azure GPT-4 still uses traditional parameters."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://test.openai.azure.com")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Azure GPT-4 response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        with patch("aicapture.vision_models.AzureOpenAI", return_value=mock_client):
            model = AzureOpenAIVisionModel(
                model="gpt-4.1",
                api_key="test_key",
                api_base="https://test.openai.azure.com",
                max_tokens=4000,
                temperature=0.5,
            )
            model._client = mock_client

            test_image = Image.new("RGB", (100, 100), color="red")
            result = model.process_image(test_image, "What is this?")

            assert result == "Azure GPT-4 response"

            # Verify traditional parameters are used for GPT-4
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4.1"
            assert call_args[1]["max_tokens"] == 4000
            assert call_args[1]["temperature"] == 0.5
            # Reasoning parameters should NOT be included
            assert "max_completion_tokens" not in call_args[1]
            assert "reasoning_effort" not in call_args[1]

    def test_azure_client_properties(self, monkeypatch: MonkeyPatch) -> None:
        """Test that Azure clients are created with correct parameters."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("AZURE_OPENAI_API_URL", "https://test.openai.azure.com")

        with patch("aicapture.vision_models.AzureOpenAI") as mock_azure_client:
            model = AzureOpenAIVisionModel(
                model="gpt-5.2", api_key="test_key", api_base="https://test.openai.azure.com"
            )
            _ = model.client  # Trigger client creation

            # Verify Azure-specific parameters
            mock_azure_client.assert_called_once()
            call_kwargs = mock_azure_client.call_args[1]
            assert "api_key" in call_kwargs
            assert "api_version" in call_kwargs
            assert "azure_endpoint" in call_kwargs


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

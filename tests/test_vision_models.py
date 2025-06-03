import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image
from pytest import MonkeyPatch

from aicapture.settings import VisionModelProvider
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

# Test image path
TEST_IMAGE_PATH = Path(__file__).parent / "sample" / "images" / "logic.png"


@pytest.fixture
def test_image_path() -> str:
    """Provide path to test image."""
    return str(TEST_IMAGE_PATH)


@pytest.fixture
def test_image_base64() -> str:
    """Create a base64 encoded test image."""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


@pytest.fixture
def mock_messages() -> List[Dict[str, Any]]:
    """Create mock messages for text processing."""
    return [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Test response"}
    ]


class TestVisionModel:
    """Test cases for the abstract VisionModel base class."""

    def test_vision_model_is_abstract(self) -> None:
        """Test that VisionModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VisionModel()  # type: ignore


class TestCreateDefaultVisionModel:
    """Test cases for create_default_vision_model function."""

    def test_create_openai_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating OpenAI vision model."""
        monkeypatch.setenv("USE_VISION", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAIVisionModel') as mock_model:
            create_default_vision_model()
            mock_model.assert_called_once()

    def test_create_anthropic_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Anthropic vision model."""
        monkeypatch.setenv("USE_VISION", "claude")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.AnthropicVisionModel') as mock_model:
            create_default_vision_model()
            mock_model.assert_called_once()

    def test_create_gemini_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Gemini vision model."""
        monkeypatch.setenv("USE_VISION", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.GeminiVisionModel') as mock_model:
            create_default_vision_model()
            mock_model.assert_called_once()

    def test_create_azure_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Azure OpenAI vision model."""
        monkeypatch.setenv("USE_VISION", "azure-openai")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.AzureOpenAIVisionModel') as mock_model:
            create_default_vision_model()
            mock_model.assert_called_once()

    def test_create_bedrock_model(self, monkeypatch: MonkeyPatch) -> None:
        """Test creating Anthropic AWS Bedrock vision model."""
        monkeypatch.setenv("USE_VISION", "anthropic_bedrock")
        
        with patch('aicapture.vision_models.AnthropicAWSBedrockVisionModel') as mock_model:
            create_default_vision_model()
            mock_model.assert_called_once()

    def test_unsupported_model_type(self, monkeypatch: MonkeyPatch) -> None:
        """Test error handling for unsupported model type."""
        monkeypatch.setenv("USE_VISION", "unsupported_model")
        
        with pytest.raises(ValueError, match="Unsupported vision model type"):
            create_default_vision_model()

    def test_create_model_with_exception(self, monkeypatch: MonkeyPatch) -> None:
        """Test error handling when model creation fails."""
        monkeypatch.setenv("USE_VISION", "openai")
        
        with patch('aicapture.vision_models.OpenAIVisionModel', side_effect=Exception("Model creation failed")):
            with pytest.raises(Exception, match="Model creation failed"):
                create_default_vision_model()


class TestIsVisionModelInstalled:
    """Test cases for is_vision_model_installed function."""

    def test_openai_installed(self) -> None:
        """Test checking if OpenAI is installed."""
        with patch('importlib.util.find_spec', return_value=MagicMock()):
            result = is_vision_model_installed("openai")
            assert result is True

    def test_anthropic_installed(self) -> None:
        """Test checking if Anthropic is installed."""
        with patch('importlib.util.find_spec', return_value=MagicMock()):
            result = is_vision_model_installed("anthropic")
            assert result is True

    def test_model_not_installed(self) -> None:
        """Test checking for model that's not installed."""
        with patch('importlib.util.find_spec', return_value=None):
            result = is_vision_model_installed("nonexistent_model")
            assert result is False

    def test_unknown_model(self) -> None:
        """Test checking for unknown model type."""
        result = is_vision_model_installed("unknown_model")
        assert result is False


class TestOpenAIVisionModel:
    """Test cases for OpenAIVisionModel."""

    @pytest.fixture
    def mock_openai_client(self) -> Mock:
        """Create a mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def mock_async_openai_client(self) -> AsyncMock:
        """Create a mock async OpenAI client."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Test async response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test OpenAI model initialization with defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAI') as mock_openai:
            model = OpenAIVisionModel()
            assert model.config.model == "gpt-4.1-mini"
            assert model.config.api_key == "test_key"
            mock_openai.assert_called()

    def test_init_with_custom_params(self, monkeypatch: MonkeyPatch) -> None:
        """Test OpenAI model initialization with custom parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAI'):
            model = OpenAIVisionModel(
                model="gpt-4o",
                api_key="custom_key",
                max_tokens=2000,
                temperature=0.5
            )
            assert model.config.model == "gpt-4o"
            assert model.config.api_key == "custom_key"
            assert model.max_tokens == 2000
            assert model.temperature == 0.5

    @pytest.mark.asyncio
    async def test_process_text_async(self, monkeypatch: MonkeyPatch, mock_async_openai_client: AsyncMock, mock_messages: List[Dict[str, Any]]) -> None:
        """Test async text processing."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.AsyncOpenAI', return_value=mock_async_openai_client):
            model = OpenAIVisionModel()
            result = await model.process_text_async(mock_messages)
            
            assert result == "Test async response"
            mock_async_openai_client.chat.completions.create.assert_called_once()

    def test_process_text_sync(self, monkeypatch: MonkeyPatch, mock_openai_client: Mock, mock_messages: List[Dict[str, Any]]) -> None:
        """Test synchronous text processing."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel()
            result = model.process_text(mock_messages)
            
            assert result == "Test response"
            mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_image_async(self, monkeypatch: MonkeyPatch, mock_async_openai_client: AsyncMock, test_image_path: str) -> None:
        """Test async image processing."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        # Ensure test image exists
        assert Path(test_image_path).exists(), f"Test image not found at {test_image_path}"
        
        with patch('aicapture.vision_models.AsyncOpenAI', return_value=mock_async_openai_client):
            model = OpenAIVisionModel()
            result = await model.process_image_async("Describe this image", test_image_path)
            
            assert result == "Test async response"
            mock_async_openai_client.chat.completions.create.assert_called_once()

    def test_process_image_sync(self, monkeypatch: MonkeyPatch, mock_openai_client: Mock, test_image_path: str) -> None:
        """Test synchronous image processing."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel()
            result = model.process_image("Describe this image", test_image_path)
            
            assert result == "Test response"
            mock_openai_client.chat.completions.create.assert_called_once()

    def test_encode_image(self, monkeypatch: MonkeyPatch, test_image_path: str) -> None:
        """Test image encoding to base64."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAI'):
            model = OpenAIVisionModel()
            encoded = model._encode_image(test_image_path)
            
            assert isinstance(encoded, str)
            assert len(encoded) > 0
            # Test that it's valid base64
            base64.b64decode(encoded)

    def test_process_image_nonexistent_file(self, monkeypatch: MonkeyPatch) -> None:
        """Test error handling for non-existent image file."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.OpenAI'):
            model = OpenAIVisionModel()
            with pytest.raises(FileNotFoundError):
                model.process_image("Describe", "nonexistent.jpg")


class TestAnthropicVisionModel:
    """Test cases for AnthropicVisionModel."""

    @pytest.fixture
    def mock_anthropic_client(self) -> Mock:
        """Create a mock Anthropic client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Anthropic test response"
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test Anthropic model initialization with defaults."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.anthropic.Anthropic') as mock_anthropic:
            model = AnthropicVisionModel()
            assert model.config.model == "claude-3-5-sonnet-20241022"
            assert model.config.api_key == "test_key"
            mock_anthropic.assert_called()

    def test_init_with_custom_params(self, monkeypatch: MonkeyPatch) -> None:
        """Test Anthropic model initialization with custom parameters."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.anthropic.Anthropic'):
            model = AnthropicVisionModel(
                model="claude-3-haiku-20240307",
                api_key="custom_key",
                max_tokens=1000
            )
            assert model.config.model == "claude-3-haiku-20240307"
            assert model.config.api_key == "custom_key"
            assert model.max_tokens == 1000

    @pytest.mark.asyncio
    async def test_process_text_async(self, monkeypatch: MonkeyPatch, mock_messages: List[Dict[str, Any]]) -> None:
        """Test async text processing with Anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock()]
        mock_response.content[0].text = "Anthropic async response"
        mock_client.messages.create.return_value = mock_response
        
        with patch('aicapture.vision_models.anthropic.AsyncAnthropic', return_value=mock_client):
            model = AnthropicVisionModel()
            result = await model.process_text_async(mock_messages)
            
            assert result == "Anthropic async response"
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_image_async(self, monkeypatch: MonkeyPatch, test_image_path: str) -> None:
        """Test async image processing with Anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock()]
        mock_response.content[0].text = "Anthropic image response"
        mock_client.messages.create.return_value = mock_response
        
        with patch('aicapture.vision_models.anthropic.AsyncAnthropic', return_value=mock_client):
            model = AnthropicVisionModel()
            result = await model.process_image_async("Analyze this image", test_image_path)
            
            assert result == "Anthropic image response"
            mock_client.messages.create.assert_called_once()

    def test_encode_image_anthropic(self, monkeypatch: MonkeyPatch, test_image_path: str) -> None:
        """Test image encoding for Anthropic format."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.anthropic.Anthropic'):
            model = AnthropicVisionModel()
            encoded = model._encode_image(test_image_path)
            
            assert isinstance(encoded, str)
            assert len(encoded) > 0
            # Test that it's valid base64
            base64.b64decode(encoded)


class TestGeminiVisionModel:
    """Test cases for GeminiVisionModel."""

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test Gemini model initialization with defaults."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.genai') as mock_genai:
            model = GeminiVisionModel()
            assert model.config.model == "gemini-2.5-flash-preview-04-17"
            assert model.config.api_key == "test_key"
            mock_genai.configure.assert_called_with(api_key="test_key")

    @pytest.mark.asyncio
    async def test_process_image_async(self, monkeypatch: MonkeyPatch, test_image_path: str) -> None:
        """Test async image processing with Gemini."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_key")
        
        mock_model = AsyncMock()
        mock_response = AsyncMock()
        mock_response.text = "Gemini image response"
        mock_model.generate_content_async.return_value = mock_response
        
        with patch('aicapture.vision_models.genai') as mock_genai:
            mock_genai.GenerativeModel.return_value = mock_model
            
            model = GeminiVisionModel()
            result = await model.process_image_async("Describe image", test_image_path)
            
            assert result == "Gemini image response"
            mock_model.generate_content_async.assert_called_once()


class TestAzureOpenAIVisionModel:
    """Test cases for AzureOpenAIVisionModel."""

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test Azure OpenAI model initialization with defaults."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        
        with patch('aicapture.vision_models.AzureOpenAI') as mock_azure:
            model = AzureOpenAIVisionModel()
            assert model.config.model == "gpt-4o"
            assert model.config.api_key == "test_key"
            mock_azure.assert_called()

    @pytest.mark.asyncio
    async def test_process_text_async(self, monkeypatch: MonkeyPatch, mock_messages: List[Dict[str, Any]]) -> None:
        """Test async text processing with Azure OpenAI."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_key")
        
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Azure response"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('aicapture.vision_models.AsyncAzureOpenAI', return_value=mock_client):
            model = AzureOpenAIVisionModel()
            result = await model.process_text_async(mock_messages)
            
            assert result == "Azure response"
            mock_client.chat.completions.create.assert_called_once()


class TestAnthropicAWSBedrockVisionModel:
    """Test cases for AnthropicAWSBedrockVisionModel."""

    def test_init_with_defaults(self, monkeypatch: MonkeyPatch) -> None:
        """Test Bedrock model initialization with defaults."""
        with patch('aicapture.vision_models.boto3') as mock_boto3:
            model = AnthropicAWSBedrockVisionModel()
            assert model.config.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
            mock_boto3.client.assert_called()

    @pytest.mark.asyncio
    async def test_process_text_async(self, monkeypatch: MonkeyPatch, mock_messages: List[Dict[str, Any]]) -> None:
        """Test async text processing with Bedrock."""
        mock_client = AsyncMock()
        mock_response = {
            'body': AsyncMock()
        }
        mock_body_data = {
            'content': [{'text': 'Bedrock response'}]
        }
        mock_response['body'].read.return_value = json.dumps(mock_body_data).encode()
        mock_client.invoke_model.return_value = mock_response
        
        with patch('aicapture.vision_models.boto3') as mock_boto3:
            mock_boto3.client.return_value = mock_client
            
            model = AnthropicAWSBedrockVisionModel()
            result = await model.process_text_async(mock_messages)
            
            assert result == "Bedrock response"


class TestVisionModelErrorHandling:
    """Test error handling across vision models."""

    def test_missing_api_key_openai(self, monkeypatch: MonkeyPatch) -> None:
        """Test error when OpenAI API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "")
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            OpenAIVisionModel()

    def test_missing_api_key_anthropic(self, monkeypatch: MonkeyPatch) -> None:
        """Test error when Anthropic API key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            AnthropicVisionModel()

    def test_missing_api_key_gemini(self, monkeypatch: MonkeyPatch) -> None:
        """Test error when Gemini API key is missing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "")
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            GeminiVisionModel()

    def test_missing_api_key_azure(self, monkeypatch: MonkeyPatch) -> None:
        """Test error when Azure OpenAI API key is missing."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            AzureOpenAIVisionModel()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
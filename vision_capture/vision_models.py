"""Vision model interfaces and implementations."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import anthropic
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image

from vision_capture.settings import (
    USE_VISION,
    AnthropicVisionConfig,
    AzureOpenAIVisionConfig,
    GeminiVisionConfig,
    ImageQuality,
    OpenAIVisionConfig,
    VisionModelProvider,
)


def create_default_vision_model() -> VisionModel:
    """Create a vision model instance based on environment configuration."""
    logger.info(f"Using vision model from provider: {USE_VISION}")
    if USE_VISION == VisionModelProvider.claude:
        return AnthropicVisionModel()
    elif USE_VISION == VisionModelProvider.openai:
        return OpenAIVisionModel()
    elif USE_VISION == VisionModelProvider.gemini:
        return GeminiVisionModel()
    elif USE_VISION == VisionModelProvider.azure_openai:
        return AzureOpenAIVisionModel()
    else:
        raise ValueError(f"Unsupported vision model type: {USE_VISION}")


def is_vision_model_installed() -> bool:
    """Check if a vision model is installed."""
    return USE_VISION in ["claude", "openai", "gemini", "azure-openai"]


class VisionModel(ABC):
    """Abstract base class for vision models."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        image_quality: str = ImageQuality.DEFAULT,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.image_quality = image_quality
        self._client: Any = None
        self._aclient: Any = None
        self._kwargs = kwargs
        self.last_token_usage: Dict[str, int] = {}

    def log_token_usage(self, usage_data: Dict[str, int]) -> None:
        """Log token usage statistics."""
        self.last_token_usage = usage_data
        logger.info(f"Token usage for {self.model}: {usage_data}")

    @staticmethod
    def convert_image_to_base64(image: Image.Image) -> Tuple[str, str]:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)  # High quality JPEG
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str, "image/jpeg"

    @property
    @abstractmethod
    def client(self) -> Any:
        """Synchronous client getter."""
        pass

    @property
    @abstractmethod
    def aclient(self) -> Any:
        """Asynchronous client getter."""
        pass

    @abstractmethod
    async def aprocess_image(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process one or more images asynchronously with the given prompt."""
        pass

    @abstractmethod
    def process_image(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process one or more images synchronously with the given prompt."""
        pass


class ImageSource(TypedDict):
    type: str
    media_type: str
    data: str


class ImageContent(TypedDict):
    type: str
    source: ImageSource


class TextContent(TypedDict):
    type: str
    text: str


class ImageUrlSource(TypedDict):
    type: str
    url: str
    detail: str


class ImageUrlContent(TypedDict):
    type: str
    image_url: ImageUrlSource


ContentItem = Union[ImageContent, TextContent, ImageUrlContent]


class AnthropicVisionModel(VisionModel):
    """Implementation for Anthropic Claude Vision models."""

    MAX_IMAGES_PER_REQUEST = 100
    MAX_IMAGE_SIZE = (8000, 8000)
    MAX_BATCH_IMAGE_SIZE = (2000, 2000)
    OPTIMAL_IMAGE_SIZE = 1568  # Maximum recommended dimension
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    def __init__(
        self,
        model: str = AnthropicVisionConfig.model,
        api_key: str = AnthropicVisionConfig.api_key,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)

    def _optimize_image(
        self, image: Image.Image, is_batch: bool = False
    ) -> Image.Image:
        """Optimize image size according to Anthropic's recommendations."""
        max_size = self.MAX_BATCH_IMAGE_SIZE if is_batch else self.MAX_IMAGE_SIZE
        width, height = image.size

        # Check if image exceeds maximum dimensions
        if width > max_size[0] or height > max_size[1]:
            raise ValueError(
                f"Image dimensions exceed maximum allowed size of {max_size}"
            )

        # Optimize to recommended size if larger
        if width > self.OPTIMAL_IMAGE_SIZE or height > self.OPTIMAL_IMAGE_SIZE:
            ratio = min(
                self.OPTIMAL_IMAGE_SIZE / width, self.OPTIMAL_IMAGE_SIZE / height
            )
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def _calculate_image_tokens(self, image: Image.Image) -> int:
        """Calculate approximate token usage for an image."""
        width, height = image.size
        return int((width * height) / 750)

    def _prepare_content(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str
    ) -> List[ContentItem]:
        """Prepare content for Anthropic API with proper image formatting."""
        content: List[ContentItem] = []
        images = [image] if isinstance(image, Image.Image) else image

        # Validate number of images
        if len(images) > self.MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"Maximum {self.MAX_IMAGES_PER_REQUEST} images allowed per request"
            )

        # Process each image
        is_batch = len(images) > 1
        for idx, img in enumerate(images, 1):
            # Optimize image
            optimized_img = self._optimize_image(img, is_batch)

            # Add image label for multiple images
            if is_batch:
                content.append(TextContent(type="text", text=f"Image {idx}:"))

            # Convert and validate image size
            image_data, media_type = self.convert_image_to_base64(optimized_img)
            if len(image_data) > self.MAX_FILE_SIZE:
                raise ValueError(f"Image {idx} exceeds maximum file size of 5MB")

            content.append(
                ImageContent(
                    type="image",
                    source=ImageSource(
                        type="base64",
                        media_type=media_type,
                        data=image_data,
                    ),
                )
            )

        # Add prompt text at the end
        content.append(TextContent(type="text", text=prompt))
        return content

    @property
    def client(self) -> anthropic.Client:
        if self._client is None:
            self._client = anthropic.Client(api_key=self.api_key)
        return cast(anthropic.Client, self._client)

    @property
    def aclient(self) -> anthropic.AsyncClient:
        if self._aclient is None:
            self._aclient = anthropic.AsyncClient(api_key=self.api_key)
        return cast(anthropic.AsyncClient, self._aclient)

    async def aprocess_image(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using Claude Vision asynchronously."""
        content = self._prepare_content(image, prompt)

        # Handle system parameter correctly
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
            "messages": [{"role": "user", "content": content}],
            "stream": kwargs.get("stream", False),
        }

        # Only add system if it's provided
        if "system" in kwargs and kwargs["system"] is not None:
            system = kwargs["system"]
            if not isinstance(system, list):
                system = [system]
            request_params["system"] = system

        # Add metadata if provided
        if "metadata" in kwargs:
            request_params["metadata"] = kwargs["metadata"]

        response = await self.aclient.messages.create(**request_params)

        # Log token usage
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        self.log_token_usage(usage)

        return str(response.content[0].text)

    def process_image(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using Claude Vision synchronously."""
        content = self._prepare_content(image, prompt)

        # Handle system parameter correctly
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
            "messages": [{"role": "user", "content": content}],
            "stream": kwargs.get("stream", False),
        }

        # Only add system if it's provided
        if "system" in kwargs and kwargs["system"] is not None:
            system = kwargs["system"]
            if not isinstance(system, list):
                system = [system]
            request_params["system"] = system

        # Add metadata if provided
        if "metadata" in kwargs:
            request_params["metadata"] = kwargs["metadata"]

        response = self.client.messages.create(**request_params)

        # Log token usage
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        self.log_token_usage(usage)

        return str(response.content[0].text)


class OpenAIVisionModel(VisionModel):
    """Implementation for OpenAI GPT-4 Vision models."""

    def __init__(
        self,
        model: str = OpenAIVisionConfig.model,
        api_key: str = OpenAIVisionConfig.api_key,
        api_base: str = OpenAIVisionConfig.api_base,
        image_quality: str = ImageQuality.DEFAULT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            image_quality=image_quality,
            **kwargs,
        )
        self.max_tokens = OpenAIVisionConfig.max_tokens
        self.temperature = OpenAIVisionConfig.temperature

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        return cast(OpenAI, self._client)

    @property
    def aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        return cast(AsyncOpenAI, self._aclient)

    def _prepare_content(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str
    ) -> List[ContentItem]:
        """Prepare content for OpenAI API."""
        content: List[ContentItem] = []
        images = [image] if isinstance(image, Image.Image) else image

        for img in images:
            base64_image, _ = self.convert_image_to_base64(img)
            content.append(
                ImageUrlContent(
                    type="image_url",
                    image_url=ImageUrlSource(
                        type="base64",
                        url=f"data:image/jpeg;base64,{base64_image}",
                        detail=self.image_quality,
                    ),
                )
            )

        content.append(TextContent(type="text", text=prompt))
        return content

    async def aprocess_image(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using OpenAI Vision asynchronously."""
        content = self._prepare_content(image, prompt)

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
        }
        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[message],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=kwargs.get("stream", False),
        )

        # Log token usage
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        self.log_token_usage(usage)

        return response.choices[0].message.content or ""

    def process_image(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using OpenAI Vision synchronously."""
        content = self._prepare_content(image, prompt)

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[message],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=kwargs.get("stream", False),
        )

        # Log token usage
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        self.log_token_usage(usage)

        return response.choices[0].message.content or ""


class GeminiVisionModel(OpenAIVisionModel):
    """Implementation for Google's Gemini Vision models using OpenAI compatibility."""

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        model: str = GeminiVisionConfig.model,
        api_key: str = GeminiVisionConfig.api_key,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, api_key=api_key, api_base=self.GEMINI_BASE_URL, **kwargs
        )


class AzureOpenAIVisionModel(OpenAIVisionModel):
    """Implementation for Azure OpenAI Vision models."""

    def __init__(
        self,
        model: str = AzureOpenAIVisionConfig.model,
        api_key: str = AzureOpenAIVisionConfig.api_key,
        api_base: str = AzureOpenAIVisionConfig.api_base,
        image_quality: str = ImageQuality.DEFAULT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            image_quality=image_quality,
            **kwargs,
        )
        self.api_version = AzureOpenAIVisionConfig.api_version

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=AzureOpenAIVisionConfig.api_base,
            )
        return cast(AzureOpenAI, self._client)

    @property
    def aclient(self) -> AsyncAzureOpenAI:
        if self._aclient is None:
            self._aclient = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=AzureOpenAIVisionConfig.api_base,
            )
        return cast(AsyncAzureOpenAI, self._aclient)

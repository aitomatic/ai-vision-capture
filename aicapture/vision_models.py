"""Vision model interfaces and implementations."""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import anthropic
from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image

from aicapture.settings import (
    USE_VISION,
    AnthropicAWSBedrockConfig,
    AnthropicVisionConfig,
    AzureOpenAIVisionConfig,
    GeminiVisionConfig,
    ImageQuality,
    OpenAIVisionConfig,
    VisionModelProvider,
    mask_sensitive_string,
)

# Type aliases for Responses API content items
ResponsesInputText = Dict[str, str]
ResponsesInputImage = Dict[str, str]
ResponsesContentItem = Union[ResponsesInputText, ResponsesInputImage]


def create_default_vision_model() -> VisionModel:
    """Create a vision model instance based on environment configuration.

    When OPENAI_USE_RESPONSES_API=true or AZURE_OPENAI_USE_RESPONSES_API=true,
    the corresponding Responses API model class will be used instead of the
    Chat Completions class.
    """
    logger.info(f"Creating vision model for provider: {USE_VISION}")
    try:
        if USE_VISION == VisionModelProvider.claude:
            return AnthropicVisionModel()
        elif USE_VISION == VisionModelProvider.openai:
            if OpenAIVisionConfig.use_responses_api:
                logger.info("Using OpenAI Responses API")
                return OpenAIResponsesVisionModel()
            return OpenAIVisionModel()
        elif USE_VISION == VisionModelProvider.gemini:
            return GeminiVisionModel()
        elif USE_VISION == VisionModelProvider.azure_openai:
            if AzureOpenAIVisionConfig.use_responses_api:
                logger.info("Using Azure OpenAI Responses API")
                return AzureOpenAIResponsesVisionModel()
            return AzureOpenAIVisionModel()
        elif USE_VISION == VisionModelProvider.anthropic_bedrock:
            return AnthropicAWSBedrockVisionModel()
        else:
            return AutoDetectVisionModel()
    except Exception as e:
        logger.error(f"Failed to create vision model: {str(e)}")
        raise


def is_vision_model_installed() -> bool:
    """Check if a vision model is installed."""
    return USE_VISION in [
        VisionModelProvider.claude,
        VisionModelProvider.openai,
        VisionModelProvider.gemini,
        VisionModelProvider.azure_openai,
        VisionModelProvider.anthropic_bedrock,
    ]


def AutoDetectVisionModel() -> VisionModel:
    """Auto-detect and create a vision model based on available API keys.

    Checks for API keys in the following order:
    1. Gemini (GEMINI_API_KEY)
    2. OpenAI (OPENAI_API_KEY or OPENAI_VISION_API_KEY)
    3. Azure OpenAI (AZURE_OPENAI_API_KEY)
    4. Anthropic (ANTHROPIC_API_KEY)

    Returns the first available model with its default configuration.

    Raises:
        ValueError: If no valid API key is found for any provider.
    """
    logger.info("Auto-detecting vision model based on available API keys...")

    # Check Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        logger.info("Found Gemini API key, using GeminiVisionModel")
        return GeminiVisionModel()

    # Check OpenAI
    openai_key = os.getenv("OPENAI_VISION_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        logger.info("Found OpenAI API key, using OpenAIVisionModel")
        return OpenAIVisionModel()

    # Check Azure OpenAI
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if azure_key:
        logger.info("Found Azure OpenAI API key, using AzureOpenAIVisionModel")
        return AzureOpenAIVisionModel()

    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        logger.info("Found Anthropic API key, using AnthropicVisionModel")
        return AnthropicVisionModel()

    # No API key found
    error_msg = (
        "No valid API key found for any vision model provider. "
        "Please set one of the following environment variables: "
        "GEMINI_API_KEY, OPENAI_API_KEY, AZURE_OPENAI_API_KEY, or ANTHROPIC_API_KEY"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


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

        # Log API credentials
        logger.debug(
            f"Using {self.__class__.__name__}\n"
            f"API Key: {mask_sensitive_string(self.api_key)}\n"
            f"API Base: {self.api_base}\n"
            f"Model: {self.model}\n"
        )

        if not self.api_key:
            logger.error("API key is required")
            raise ValueError("API key is required")
        # if not self.api_base:
        #     logger.error("API base is required")
        #     raise ValueError("API base is required")

    async def aclose(self) -> None:
        """Close the async HTTP client to avoid 'Event loop is closed' warnings.

        Should be called before the event loop shuts down (e.g. at the end of
        an ``asyncio.run()`` block) so that httpx can clean up its connection
        pool gracefully instead of relying on garbage-collection after the loop
        is already closed.
        """
        if self._aclient is not None:
            try:
                await self._aclient.close()
            except Exception:
                pass
            self._aclient = None

    def log_token_usage(self, usage_data: Dict[str, int]) -> None:
        """Log token usage statistics."""
        self.last_token_usage = usage_data
        logger.info(f"Token usage: {usage_data}")

    @staticmethod
    def convert_image_to_base64(image: Image.Image) -> Tuple[str, str]:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)  # High quality JPEG
        raw = buffered.getvalue()
        buffered.close()
        img_str = base64.b64encode(raw).decode()
        del raw
        return img_str, "image/jpeg"

    @property
    @abstractmethod
    def client(self) -> Any:
        """Synchronous client getter."""

    @property
    @abstractmethod
    def aclient(self) -> Any:
        """Asynchronous client getter."""

    @abstractmethod
    async def process_image_async(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process one or more images asynchronously with the given prompt."""

    @abstractmethod
    def process_image(self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any) -> str:
        """Process one or more images synchronously with the given prompt."""

    @abstractmethod
    async def process_text_async(self, messages: List[Any], **kwargs: Any) -> str:
        """Process text asynchronously with the given prompt."""


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

    def _optimize_image(self, image: Image.Image, is_batch: bool = False) -> Image.Image:
        """Optimize image size according to Anthropic's recommendations."""
        max_size = self.MAX_BATCH_IMAGE_SIZE if is_batch else self.MAX_IMAGE_SIZE
        width, height = image.size

        # Check if image exceeds maximum dimensions
        if width > max_size[0] or height > max_size[1]:
            raise ValueError(f"Image dimensions exceed maximum allowed size of {max_size}")

        # Optimize to recommended size if larger
        if width > self.OPTIMAL_IMAGE_SIZE or height > self.OPTIMAL_IMAGE_SIZE:
            ratio = min(self.OPTIMAL_IMAGE_SIZE / width, self.OPTIMAL_IMAGE_SIZE / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def _calculate_image_tokens(self, image: Image.Image) -> int:
        """Calculate approximate token usage for an image."""
        width, height = image.size
        return int((width * height) / 750)

    def _prepare_content(self, image: Union[Image.Image, List[Image.Image]], prompt: str) -> List[ContentItem]:
        """Prepare content for Anthropic API with proper image formatting."""
        content: List[ContentItem] = []
        images = [image] if isinstance(image, Image.Image) else image

        # Validate number of images
        if len(images) > self.MAX_IMAGES_PER_REQUEST:
            raise ValueError(f"Maximum {self.MAX_IMAGES_PER_REQUEST} images allowed per request")

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

    async def process_image_async(
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

    def process_image(self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any) -> str:
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

    async def process_text_async(self, messages: List[Any], **kwargs: Any) -> str:
        """Process text using Claude Vision asynchronously."""
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
            "messages": messages,
        }

        response = await self.aclient.messages.create(**request_params)

        # Log token usage
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        self.log_token_usage(usage)

        return str(response.content[0].text)


class OpenAIVisionModel(VisionModel):
    """Implementation for OpenAI GPT-4 Vision models.

    Supports both traditional models (GPT-4) and reasoning models (GPT-5 series).
    Reasoning models (gpt-5, gpt-5.1, gpt-5.2, etc.) have different parameter requirements:
    - Use max_completion_tokens instead of max_tokens
    - Don't support temperature unless reasoning_effort is "none"
    - Support reasoning_effort parameter (none, low, medium, high, xhigh)
    """

    # Reasoning model prefixes
    REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3")

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

        # Detect if this is a reasoning model
        self.is_reasoning_model = self._is_reasoning_model(model)

        # Handle max_tokens/max_completion_tokens
        if "max_completion_tokens" in kwargs:
            self.max_completion_tokens = kwargs["max_completion_tokens"]
        elif "max_tokens" in kwargs:
            self.max_completion_tokens = kwargs["max_tokens"]
        else:
            self.max_completion_tokens = OpenAIVisionConfig.max_tokens

        # Handle temperature (only for non-reasoning models or when reasoning_effort="none")
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        else:
            self.temperature = OpenAIVisionConfig.temperature

        # Handle reasoning_effort parameter (for reasoning models)
        self.reasoning_effort = kwargs.get("reasoning_effort", None)

        if self.is_reasoning_model:
            logger.info(f"Using reasoning model: {model} with reasoning_effort={self.reasoning_effort}")

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if the model is a reasoning model (GPT-5 series, o1, o3)."""
        return any(model.startswith(prefix) for prefix in OpenAIVisionModel.REASONING_MODEL_PREFIXES)

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

    def _prepare_content(self, image: Union[Image.Image, List[Image.Image]], prompt: str) -> List[ContentItem]:
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

    async def process_image_async(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using OpenAI Vision asynchronously."""
        content = self._prepare_content(image, prompt)

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
        }

        # Build request parameters based on model type
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": [message],
            "stream": kwargs.get("stream", False),
        }

        # For reasoning models, use different parameter names and skip temperature
        if self.is_reasoning_model:
            # Use max_completion_tokens for reasoning models
            request_params["max_completion_tokens"] = self.max_completion_tokens

            # Add reasoning_effort if specified
            if self.reasoning_effort is not None:
                request_params["reasoning_effort"] = self.reasoning_effort

            # Only add temperature if reasoning_effort is "none"
            if self.reasoning_effort == "none":
                request_params["temperature"] = self.temperature
        else:
            # Traditional models use max_tokens and temperature
            request_params["max_tokens"] = self.max_completion_tokens
            request_params["temperature"] = self.temperature

        response = await self.aclient.chat.completions.create(**request_params)

        # Log token usage
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        self.log_token_usage(usage)

        return response.choices[0].message.content or ""

    def process_image(self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any) -> str:
        """Process image(s) using OpenAI Vision synchronously."""
        content = self._prepare_content(image, prompt)

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
        }

        # Build request parameters based on model type
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": [message],
            "stream": kwargs.get("stream", False),
        }

        # For reasoning models, use different parameter names and skip temperature
        if self.is_reasoning_model:
            # Use max_completion_tokens for reasoning models
            request_params["max_completion_tokens"] = self.max_completion_tokens

            # Add reasoning_effort if specified
            if self.reasoning_effort is not None:
                request_params["reasoning_effort"] = self.reasoning_effort

            # Only add temperature if reasoning_effort is "none"
            if self.reasoning_effort == "none":
                request_params["temperature"] = self.temperature
        else:
            # Traditional models use max_tokens and temperature
            request_params["max_tokens"] = self.max_completion_tokens
            request_params["temperature"] = self.temperature

        response = self.client.chat.completions.create(**request_params)

        # Log token usage
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        self.log_token_usage(usage)

        return response.choices[0].message.content or ""

    async def process_text_async(self, messages: List[Any], **kwargs: Any) -> str:
        """Process text using OpenAI Vision asynchronously."""
        # Build request parameters based on model type
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }

        # For reasoning models, use different parameter names and skip temperature
        if self.is_reasoning_model:
            # Use max_completion_tokens for reasoning models
            request_params["max_completion_tokens"] = kwargs.get(
                "max_completion_tokens", kwargs.get("max_tokens", self.max_completion_tokens)
            )

            # Add reasoning_effort if specified
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                request_params["reasoning_effort"] = reasoning_effort

            # Only add temperature if reasoning_effort is "none"
            if reasoning_effort == "none":
                request_params["temperature"] = kwargs.get("temperature", self.temperature)
        else:
            # Traditional models use max_tokens and temperature
            request_params["max_tokens"] = kwargs.get("max_tokens", self.max_completion_tokens)
            request_params["temperature"] = kwargs.get("temperature", self.temperature)

        response = await self.aclient.chat.completions.create(**request_params)

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

    # Fallback prompt used when the primary prompt triggers Gemini's recitation filter.
    # Gemini blocks responses when it detects the model would reproduce verbatim content.
    # Prompts with aggressive "extract everything exactly" instructions (e.g. "DO NOT summarize",
    # "no meta-commentary") are most likely to trigger this on text-heavy document pages.
    FALLBACK_PROMPT = "Extract the document content comprehensively as markdown. Include all text, tables, charts, diagrams, and image descriptions."

    def __init__(
        self,
        model: str = GeminiVisionConfig.model,
        api_key: str = GeminiVisionConfig.api_key,
        **kwargs: Any,
    ) -> None:
        # Use Gemini-specific defaults if not explicitly provided
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
        if "temperature" not in kwargs:
            kwargs["temperature"] = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
        super().__init__(model=model, api_key=api_key, api_base=self.GEMINI_BASE_URL, **kwargs)

    async def process_image_async(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using Gemini Vision asynchronously.

        Includes retry logic for Gemini's recitation content filter. When the filter
        blocks a response (completion_tokens=0, finish_reason contains 'content_filter'),
        retries with a simplified fallback prompt that avoids triggering the filter.
        """
        content = self._prepare_content(image, prompt)

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
        }

        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[message],
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            stream=kwargs.get("stream", False),
        )

        # Log token usage
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            self.log_token_usage(usage)

        # Check for Gemini content filter (recitation block)
        finish_reason = response.choices[0].finish_reason or ""
        if "content_filter" in finish_reason or (
            response.usage and response.usage.completion_tokens == 0 and response.choices[0].message.content is None
        ):
            logger.warning(
                f"Gemini content filter triggered (finish_reason={finish_reason}). Retrying with fallback prompt."
            )
            # Retry with fallback prompt
            fallback_content = self._prepare_content(image, self.FALLBACK_PROMPT)
            fallback_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": fallback_content,  # type: ignore
            }
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=[fallback_message],
                max_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                stream=kwargs.get("stream", False),
            )
            # Log retry token usage
            if response.usage:
                retry_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                self.log_token_usage(retry_usage)

            retry_finish = response.choices[0].finish_reason or ""
            if "content_filter" in retry_finish:
                logger.warning(f"Gemini content filter triggered again on retry (finish_reason={retry_finish})")

        return response.choices[0].message.content or ""

    def process_image(self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any) -> str:
        """Process image(s) using Gemini Vision synchronously.

        Includes retry logic for Gemini's recitation content filter.
        """
        content = self._prepare_content(image, prompt)

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[message],
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            stream=kwargs.get("stream", False),
        )

        # Log token usage
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            self.log_token_usage(usage)

        # Check for Gemini content filter (recitation block)
        finish_reason = response.choices[0].finish_reason or ""
        if "content_filter" in finish_reason or (
            response.usage and response.usage.completion_tokens == 0 and response.choices[0].message.content is None
        ):
            logger.warning(
                f"Gemini content filter triggered (finish_reason={finish_reason}). Retrying with fallback prompt."
            )
            fallback_content = self._prepare_content(image, self.FALLBACK_PROMPT)
            fallback_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": fallback_content,  # type: ignore
            }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[fallback_message],
                max_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                stream=kwargs.get("stream", False),
            )
            if response.usage:
                retry_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                self.log_token_usage(retry_usage)

            retry_finish = response.choices[0].finish_reason or ""
            if "content_filter" in retry_finish:
                logger.warning(f"Gemini content filter triggered again on retry (finish_reason={retry_finish})")

        return response.choices[0].message.content or ""


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
                azure_endpoint=self.api_base,  # type: ignore
            )
        return cast(AzureOpenAI, self._client)

    @property
    def aclient(self) -> AsyncAzureOpenAI:
        if self._aclient is None:
            self._aclient = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,  # type: ignore
            )

        return cast(AsyncAzureOpenAI, self._aclient)


class OpenAIResponsesVisionModel(VisionModel):
    """OpenAI Vision using the Responses API.

    The Responses API is OpenAI's recommended API for new projects, replacing
    Chat Completions. Key differences from OpenAIVisionModel:

    - Uses ``client.responses.create()`` instead of ``client.chat.completions.create()``
    - Content types: ``input_text`` / ``input_image`` instead of ``text`` / ``image_url``
    - Parameter names: ``input`` instead of ``messages``, ``instructions`` instead of
      system message, ``max_output_tokens`` instead of ``max_tokens``
    - Response access: ``response.output_text`` instead of ``response.choices[0].message.content``
    - Supports stateful conversations via ``previous_response_id``
    - Supports native PDF input via ``input_file`` content type
    - For reasoning models, uses ``reasoning={"effort": ...}`` instead of ``reasoning_effort``

    Set ``store=False`` (default) to prevent OpenAI from retaining response data.
    Pass ``previous_response_id`` in kwargs to chain stateful conversations.
    """

    REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3")

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

        self.is_reasoning_model = self._is_reasoning_model(model)

        if "max_output_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_output_tokens"]
        elif "max_completion_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_completion_tokens"]
        elif "max_tokens" in kwargs:
            self.max_output_tokens = kwargs["max_tokens"]
        else:
            self.max_output_tokens = OpenAIVisionConfig.max_tokens

        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        else:
            self.temperature = OpenAIVisionConfig.temperature

        self.reasoning_effort = kwargs.get("reasoning_effort", None)

        # Whether to store responses server-side (default False for stateless use)
        self.store = kwargs.get("store", False)

        if self.is_reasoning_model:
            logger.info(f"Using Responses API reasoning model: {model} with reasoning_effort={self.reasoning_effort}")

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if the model is a reasoning model (GPT-5 series, o1, o3)."""
        return any(model.startswith(prefix) for prefix in OpenAIResponsesVisionModel.REASONING_MODEL_PREFIXES)

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

    def _prepare_content(self, image: Union[Image.Image, List[Image.Image]], prompt: str) -> List[ResponsesContentItem]:
        """Prepare content for Responses API using input_text / input_image types."""
        content: List[ResponsesContentItem] = []
        images = [image] if isinstance(image, Image.Image) else image

        for img in images:
            base64_image, _ = self.convert_image_to_base64(img)
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": self.image_quality,
                }
            )

        content.append({"type": "input_text", "text": prompt})
        return content

    def _build_request_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Build common request parameters for the Responses API."""
        params: Dict[str, Any] = {
            "model": self.model,
            "store": kwargs.get("store", self.store),
        }

        # Instructions (system prompt)
        if "instructions" in kwargs and kwargs["instructions"] is not None:
            params["instructions"] = kwargs["instructions"]

        # Stateful conversation chaining
        if "previous_response_id" in kwargs and kwargs["previous_response_id"] is not None:
            params["previous_response_id"] = kwargs["previous_response_id"]

        # Reasoning models
        if self.is_reasoning_model:
            params["max_output_tokens"] = self.max_output_tokens

            effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if effort is not None:
                params["reasoning"] = {"effort": effort}

                if effort == "none":
                    params["temperature"] = kwargs.get("temperature", self.temperature)
        else:
            params["max_output_tokens"] = self.max_output_tokens
            params["temperature"] = kwargs.get("temperature", self.temperature)

        return params

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract and log token usage from a Responses API response."""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        self.log_token_usage(usage)
        return usage

    async def process_image_async(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any
    ) -> str:
        """Process image(s) using OpenAI Responses API asynchronously."""
        content = self._prepare_content(image, prompt)

        request_params = self._build_request_params(**kwargs)
        request_params["input"] = [{"role": "user", "content": content}]

        response = await self.aclient.responses.create(**request_params)
        self._extract_usage(response)

        return response.output_text or ""

    def process_image(self, image: Union[Image.Image, List[Image.Image]], prompt: str, **kwargs: Any) -> str:
        """Process image(s) using OpenAI Responses API synchronously."""
        content = self._prepare_content(image, prompt)

        request_params = self._build_request_params(**kwargs)
        request_params["input"] = [{"role": "user", "content": content}]

        response = self.client.responses.create(**request_params)
        self._extract_usage(response)

        return response.output_text or ""

    async def process_text_async(self, messages: List[Any], **kwargs: Any) -> str:
        """Process text using OpenAI Responses API asynchronously.

        Converts Chat Completions-style messages to Responses API input items.
        System messages are extracted and passed as ``instructions``.
        """
        instructions = None
        input_items: List[Any] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                instructions = msg.get("content", "")
            else:
                input_items.append(msg)

        request_params = self._build_request_params(**kwargs)
        request_params["input"] = input_items
        if instructions and "instructions" not in kwargs:
            request_params["instructions"] = instructions

        response = await self.aclient.responses.create(**request_params)
        self._extract_usage(response)

        return response.output_text or ""


class AzureOpenAIResponsesVisionModel(OpenAIResponsesVisionModel):
    """Azure OpenAI Vision using the Responses API.

    Uses the Azure OpenAI v1 endpoint (``/openai/v1/responses``).
    Inherits all Responses API behavior from OpenAIResponsesVisionModel,
    but creates Azure-specific clients.

    Note: Azure's Responses API has some limitations vs. direct OpenAI:
    - Web search tool is not supported (use Bing Grounding instead)
    - PDF file upload with purpose="user_data" is not supported
    """

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
                azure_endpoint=self.api_base,  # type: ignore
            )
        return cast(AzureOpenAI, self._client)

    @property
    def aclient(self) -> AsyncAzureOpenAI:
        if self._aclient is None:
            self._aclient = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,  # type: ignore
            )
        return cast(AsyncAzureOpenAI, self._aclient)


class AnthropicAWSBedrockVisionModel(AnthropicVisionModel):
    """Implementation for Anthropic Claude Vision models via AWS Bedrock.

    Extends the standard AnthropicVisionModel but uses Bedrock client instead.
    """

    def __init__(
        self,
        model: str = AnthropicAWSBedrockConfig.model,
        api_key: str = AnthropicAWSBedrockConfig.api_key,
        **kwargs: Any,
    ) -> None:
        # Skip AnthropicVisionModel's __init__ and call VisionModel's __init__ directly
        VisionModel.__init__(self, model=model, api_key=api_key, **kwargs)
        self.aws_access_key_id = AnthropicAWSBedrockConfig.aws_access_key_id
        self.aws_secret_access_key = AnthropicAWSBedrockConfig.aws_secret_access_key
        self.aws_session_token = AnthropicAWSBedrockConfig.aws_session_token
        self.aws_region = AnthropicAWSBedrockConfig.aws_region
        self.aws_vpc_endpoint_url = AnthropicAWSBedrockConfig.aws_vpc_endpoint_url

    @property
    def client(self) -> Any:  # type: ignore
        """Get synchronous Bedrock client."""
        if self._client is None:
            if self.aws_vpc_endpoint_url:
                self._client = anthropic.AnthropicBedrock(
                    aws_access_key=self.aws_access_key_id,
                    aws_secret_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    aws_region=self.aws_region,
                    base_url=self.aws_vpc_endpoint_url,
                )
            else:
                self._client = anthropic.AnthropicBedrock(
                    aws_access_key=self.aws_access_key_id,
                    aws_secret_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    aws_region=self.aws_region,
                )
        return self._client

    @property
    def aclient(self) -> Any:  # type: ignore
        """Get asynchronous Bedrock client."""
        if self._aclient is None:
            if self.aws_vpc_endpoint_url:
                self._aclient = anthropic.AsyncAnthropicBedrock(
                    aws_access_key=self.aws_access_key_id,
                    aws_secret_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    aws_region=self.aws_region,
                    base_url=self.aws_vpc_endpoint_url,
                )
            else:
                self._aclient = anthropic.AsyncAnthropicBedrock(
                    aws_access_key=self.aws_access_key_id,
                    aws_secret_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    aws_region=self.aws_region,
                )
        return self._aclient

    # Reusing all the image processing methods from the parent class

# GPT-5 Reasoning Models Support

This document describes the upgrade to support OpenAI's latest GPT-5 series reasoning models (GPT-5, GPT-5.1, GPT-5.2).

## Overview

GPT-5 series models are **reasoning models** with different parameter requirements compared to traditional models like GPT-4. The key differences are:

1. **Use `max_completion_tokens` instead of `max_tokens`**
2. **Don't support `temperature` parameter** (unless `reasoning_effort` is set to "none")
3. **Support new `reasoning_effort` parameter** with levels: none, low, medium, high, xhigh (varies by model)

## Supported Models

The following models are now automatically detected as reasoning models:

- **GPT-5 series**: `gpt-5`, `gpt-5.1`, `gpt-5.2`, `gpt-5.2-chat-latest`, `gpt-5.2-pro`, `gpt-5-mini`
- **O-series**: `o1`, `o1-mini`, `o3`, `o3-mini`

Traditional models like `gpt-4.1`, `gpt-4-turbo` continue to work as before with `max_tokens` and `temperature`.

### Azure OpenAI Support

**Azure OpenAI** also supports GPT-5 reasoning models! All the same parameter handling applies to Azure deployments. Simply deploy a GPT-5 model in your Azure OpenAI resource and use `AzureOpenAIVisionModel` with the deployment name.

## API Changes

### Automatic Detection

The `OpenAIVisionModel` class now automatically detects reasoning models based on the model name prefix. No code changes are needed for existing GPT-4 usage.

### New Parameters

```python
from aicapture import OpenAIVisionModel, VisionParser

# GPT-5.2 with reasoning enabled
model = OpenAIVisionModel(
    model="gpt-5.2",
    reasoning_effort="medium",      # Options: none, low, medium, high, xhigh
    max_completion_tokens=5000,     # Replaces max_tokens for reasoning models
)

# GPT-5.2 in instant mode (no reasoning)
model_instant = OpenAIVisionModel(
    model="gpt-5.2",
    reasoning_effort="none",        # Disables reasoning
    max_completion_tokens=5000,
    temperature=0.0,                # Temperature only works when reasoning_effort="none"
)
```

### Environment Variables

**For OpenAI:**

New environment variable in `.env` or environment:

```bash
# Optional: Set reasoning effort level globally
OPENAI_REASONING_EFFORT=medium     # Options: none, low, medium, high, xhigh
```

Existing variables continue to work:
```bash
OPENAI_API_KEY=your_key
OPENAI_VISION_MODEL=gpt-5.2        # Model selection
OPENAI_MAX_TOKENS=5000             # Becomes max_completion_tokens for GPT-5
OPENAI_TEMPERATURE=0.0             # Only used when reasoning_effort="none"
```

**For Azure OpenAI:**

```bash
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_MODEL=gpt-5.2         # Your Azure deployment name
AZURE_OPENAI_API_URL=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-11-01-preview
AZURE_OPENAI_REASONING_EFFORT=medium  # Optional: reasoning effort level
```

## Backward Compatibility

**100% backward compatible** with existing code:

- GPT-4 and older models continue to use `max_tokens` and `temperature` parameters
- No changes required for existing implementations
- Auto-detection ensures correct parameters are used for each model type

## Examples

### Basic Usage

```python
from aicapture import OpenAIVisionModel, VisionParser

# Traditional GPT-4 usage (unchanged)
model_gpt4 = OpenAIVisionModel(model="gpt-4.1")
parser = VisionParser(vision_model=model_gpt4)
result = parser.process_image("image.png")

# GPT-5.2 with reasoning
model_gpt5 = OpenAIVisionModel(
    model="gpt-5.2",
    reasoning_effort="high",
    max_completion_tokens=5000
)
parser = VisionParser(vision_model=model_gpt5)
result = parser.process_image("image.png")
```

### Advanced Configuration

**OpenAI Examples** - See `examples/openai/parse_image_gpt5.py`:
- GPT-5.2 with reasoning enabled
- GPT-5.2 in instant mode (reasoning disabled)
- GPT-5.1 usage

**Azure OpenAI Examples** - See `examples/azure/parse_image_gpt5.py`:
- Azure GPT-5.2 with reasoning enabled
- Azure GPT-5.2 in instant mode
- Azure GPT-5.1 usage
- Using environment variables for Azure deployments

```python
from aicapture import AzureOpenAIVisionModel, VisionParser

# Azure GPT-5.2 with reasoning
model = AzureOpenAIVisionModel(
    model="gpt-5.2",  # Your Azure deployment name
    reasoning_effort="high",
    max_completion_tokens=5000
)
parser = VisionParser(vision_model=model)
result = parser.process_image("image.png")
```

## Implementation Details

### Reasoning Effort Levels

| Model | Supported Levels |
|-------|-----------------|
| GPT-5 | minimal, low, medium (default), high |
| GPT-5.1 | none (default), low, medium, high |
| GPT-5.2 | none (default), low, medium, high, **xhigh** |

### Parameter Handling

The implementation automatically handles parameters based on model type:

**For Reasoning Models (GPT-5 series, o1, o3):**
- ✅ `max_completion_tokens` - Used for token limit
- ✅ `reasoning_effort` - Controls reasoning level
- ✅ `temperature` - Only when `reasoning_effort="none"`
- ❌ `max_tokens` - Not sent to API (converted to max_completion_tokens)

**For Traditional Models (GPT-4, etc.):**
- ✅ `max_tokens` - Used for token limit
- ✅ `temperature` - Used for response randomness
- ❌ `max_completion_tokens` - Not sent to API
- ❌ `reasoning_effort` - Not sent to API

## Testing

New test suites added:
- `tests/test_vision_models.py::TestOpenAIReasoningModels` (8 tests)
- `tests/test_vision_models.py::TestAzureOpenAIReasoningModels` (6 tests)

```bash
# Run all vision model tests
uv run python -m pytest tests/test_vision_models.py -v

# Run only OpenAI GPT-5 tests
uv run python -m pytest tests/test_vision_models.py::TestOpenAIReasoningModels -v

# Run only Azure OpenAI GPT-5 tests
uv run python -m pytest tests/test_vision_models.py::TestAzureOpenAIReasoningModels -v
```

All 42 tests pass, including 14 new tests for GPT-5 functionality across OpenAI and Azure.

## References

- [GPT-5 Model Documentation](https://platform.openai.com/docs/models/gpt-5)
- [GPT-5.1 for Developers](https://openai.com/index/gpt-5-1-for-developers/)
- [GPT-5.2 Complete Guide](https://platform.openai.com/docs/models/gpt-5.2)
- [Reasoning Models Guide](https://platform.openai.com/docs/guides/reasoning)
- [Azure OpenAI Reasoning Models](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning)

## Files Modified

1. **aicapture/vision_models.py** - Added GPT-5 reasoning model support (OpenAI and Azure)
2. **aicapture/settings.py** - Added `reasoning_effort` configuration for OpenAI and Azure
3. **.env.template** - Updated with GPT-5 documentation for both providers
4. **tests/test_vision_models.py** - Added comprehensive GPT-5 tests (14 new tests)
5. **examples/openai/parse_image_gpt5.py** - New OpenAI example demonstrating GPT-5 usage
6. **examples/azure/parse_image_gpt5.py** - New Azure OpenAI example demonstrating GPT-5 usage

## Migration Guide

### If Using GPT-4 (No Changes Needed)

Your existing code continues to work without modifications:

```python
# This continues to work exactly as before
model = OpenAIVisionModel(model="gpt-4.1", temperature=0.0, max_tokens=5000)
```

### Upgrading to GPT-5

Simple upgrade - just change the model name:

```python
# Before (GPT-4)
model = OpenAIVisionModel(model="gpt-4.1", temperature=0.0, max_tokens=5000)

# After (GPT-5.2) - automatically uses correct parameters
model = OpenAIVisionModel(model="gpt-5.2", reasoning_effort="medium", max_completion_tokens=5000)
```

The library automatically:
- Detects it's a reasoning model
- Uses `max_completion_tokens` instead of `max_tokens`
- Skips `temperature` (unless `reasoning_effort="none"`)
- Adds `reasoning_effort` parameter to API calls

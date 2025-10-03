# Auto-Detection Examples

This folder contains examples demonstrating the **Auto-Detection** feature, which automatically detects and uses the first available vision model based on your environment variables.

## What is Auto-Detection?

Instead of explicitly setting `USE_VISION` to specify a provider, you can simply set any supported API key, and the library will automatically detect and use the appropriate provider.

### Detection Order

The library checks for API keys in the following order:

1. **Gemini** (`GEMINI_API_KEY`)
2. **OpenAI** (`OPENAI_API_KEY` or `OPENAI_VISION_API_KEY`)
3. **Azure OpenAI** (`AZURE_OPENAI_API_KEY`)
4. **Anthropic** (`ANTHROPIC_API_KEY`)

The first available provider will be used with its default model configuration.

## Quick Start

### 1. Set an API Key

```bash
# Choose any ONE of these
export GEMINI_API_KEY=your_key_here
# OR
export OPENAI_API_KEY=your_key_here
# OR
export AZURE_OPENAI_API_KEY=your_key_here
# OR
export ANTHROPIC_API_KEY=your_key_here
```

### 2. Run an Example

```bash
# Parse an image
python examples/auto/parse_image.py

# Parse a PDF
python examples/auto/parse_pdf.py
```

No need to set `USE_VISION`! The library will automatically detect and use your configured provider.

## Examples in This Folder

### `parse_image.py`
Demonstrates how to parse an image file using auto-detected vision model.

**Usage:**
```bash
python examples/auto/parse_image.py
```

### `parse_pdf.py`
Demonstrates how to parse PDF files using auto-detected vision model.

**Usage:**
```bash
python examples/auto/parse_pdf.py
```

## Benefits of Auto-Detection

- ✅ **Simplified Setup**: No need to set `USE_VISION` environment variable
- ✅ **Flexibility**: Easily switch providers by changing API keys
- ✅ **Smart Defaults**: Each provider uses its recommended default model
- ✅ **Error Handling**: Clear error messages if no API key is found

## When to Use Manual Provider Selection

While auto-detection is convenient, you might want to explicitly set `USE_VISION` if:

- You have multiple API keys configured and want to use a specific one
- You need to override the default detection order
- You're in a production environment with specific provider requirements

See the other example folders (`openai/`, `anthropic/`, `gemini/`, `azure/`) for manual provider selection examples.

## Default Models by Provider

When using auto-detection, the following default models are used:

- **Gemini**: `gemini-2.5-flash`
- **OpenAI**: `gpt-4.1`
- **Azure OpenAI**: `gpt-4.1`
- **Anthropic**: `claude-sonnet-4-5-20250929`

You can always create a custom model instance if you need different settings. See the [Configuration Guide](../configuration.md) for more details.


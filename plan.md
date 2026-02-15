# Migration Plan: Add OpenAI Responses API Support

## Current State Analysis

### How Chat Completions is Used Today

The library currently uses `client.chat.completions.create()` in two model classes:

1. **`OpenAIVisionModel`** (`aicapture/vision_models.py:419-641`)
   - Sync: `self.client.chat.completions.create(**request_params)` (line 589)
   - Async: `await self.aclient.chat.completions.create(**request_params)` (line 544)
   - Handles reasoning models (GPT-5, o1, o3) with `max_completion_tokens` and `reasoning_effort`
   - Uses `ChatCompletionUserMessageParam` type from `openai.types.chat`

2. **`AzureOpenAIVisionModel`** (`aicapture/vision_models.py:799-839`)
   - Inherits all methods from `OpenAIVisionModel`
   - Only overrides `client` and `aclient` properties to use `AzureOpenAI`/`AsyncAzureOpenAI`

3. **`GeminiVisionModel`** (`aicapture/vision_models.py:643-797`)
   - Inherits from `OpenAIVisionModel`, uses OpenAI-compatible endpoint
   - Adds Gemini-specific content filter retry logic

All three classes share the same request/response pattern:
```python
# Request
request_params = {
    "model": self.model,
    "messages": [{"role": "user", "content": [...]}],
    "max_tokens": ...,
    "temperature": ...,
    "stream": False,
}
response = await self.aclient.chat.completions.create(**request_params)

# Response
text = response.choices[0].message.content
usage = response.usage  # .prompt_tokens, .completion_tokens, .total_tokens
```

### Content Format (Vision/Image)
```python
# Current Chat Completions format:
content = [
    {"type": "image_url", "image_url": {"type": "base64", "url": "data:image/jpeg;base64,...", "detail": "high"}},
    {"type": "text", "text": "Extract the content..."},
]
message = {"role": "user", "content": content}
```

---

## Responses API Key Differences

### 1. Endpoint & SDK Method
| Aspect | Chat Completions | Responses API |
|--------|-----------------|---------------|
| SDK call | `client.chat.completions.create()` | `client.responses.create()` |
| Endpoint | `POST /v1/chat/completions` | `POST /v1/responses` |

### 2. Parameter Mapping
| Chat Completions | Responses API | Notes |
|-----------------|---------------|-------|
| `messages` (array) | `input` (string or array) | Can be a plain string for simple prompts |
| `system` message in `messages` | `instructions` (top-level) | Cleaner separation |
| `max_tokens` | `max_output_tokens` | Renamed |
| `max_completion_tokens` | `max_output_tokens` | Unified |
| `temperature` | `temperature` | Same |
| `stream` | `stream` | Same, but event types differ |
| `response_format` | `text.format` | Flattened schema structure |
| N/A | `previous_response_id` | **NEW**: Server-side conversation chaining |
| N/A | `store` | Whether to persist response (default: true) |
| N/A | `reasoning` | `{"effort": "medium"}` replaces `reasoning_effort` |

### 3. Content Type Changes (Vision)
| Chat Completions | Responses API |
|-----------------|---------------|
| `"type": "text"` | `"type": "input_text"` |
| `"type": "image_url"` with nested `{"image_url": {"url": "...", "detail": "..."}}` | `"type": "input_image"` with flat `{"image_url": "...", "detail": "..."}` |
| N/A | `"type": "input_file"` (NEW: native PDF support) |

### 4. Response Structure
| Chat Completions | Responses API |
|-----------------|---------------|
| `response.choices[0].message.content` | `response.output_text` (convenience) |
| `response.usage.prompt_tokens` | `response.usage.input_tokens` |
| `response.usage.completion_tokens` | `response.usage.output_tokens` |
| `response.usage.total_tokens` | `response.usage.total_tokens` |

### 5. Stateful Conversations
The biggest new capability. With `previous_response_id`, the server retains conversation context:
- No need to re-send all messages each turn
- Reasoning tokens from o-series models are preserved between turns (lost in Chat Completions)
- OpenAI reports 40-80% improvement in cache utilization
- 3% improvement on SWE-bench with reasoning models

### 6. Native PDF Support
The Responses API supports `input_file` content type for PDFs:
```python
{"type": "input_file", "filename": "doc.pdf", "file_data": "data:application/pdf;base64,..."}
```
This could potentially simplify `VisionParser` by sending PDFs directly instead of converting to images first.

### 7. Azure OpenAI Support
Azure supports the Responses API (GA since August 2025). Key differences:
- New v1 endpoint: `https://{resource}.openai.azure.com/openai/v1/`
- No longer requires `api-version` query parameter with v1
- Web search tool NOT supported (use Bing Grounding instead)
- Some limitations with PDF file uploads and background streaming

---

## Migration Plan

### Phase 1: Add `OpenAIResponsesVisionModel` (New Class)

**Approach**: Add a new model class alongside the existing one. Do NOT modify `OpenAIVisionModel` — users who depend on Chat Completions keep working unchanged.

#### 1a. New model class in `vision_models.py`

```python
class OpenAIResponsesVisionModel(VisionModel):
    """OpenAI Vision using the Responses API (recommended for new projects)."""
```

Key implementation details:
- **Client**: Reuse the same `OpenAI` / `AsyncOpenAI` client (same SDK, different method)
- **`_prepare_content()`**: Convert images to Responses API format:
  - `"type": "text"` → `"type": "input_text"`
  - `"type": "image_url"` → `"type": "input_image"` with flat structure
- **`process_image_async()`**: Call `self.aclient.responses.create()` with:
  - `input=` instead of `messages=`
  - `instructions=` for system prompt (from kwargs)
  - `max_output_tokens=` instead of `max_tokens`/`max_completion_tokens`
  - `reasoning={"effort": ...}` instead of `reasoning_effort=` for reasoning models
  - `store=False` by default (our use case is stateless per-page processing)
- **`process_image()`**: Sync version using `self.client.responses.create()`
- **`process_text_async()`**: Convert messages format to `input` items format
- **Response handling**: Use `response.output_text` and `response.usage.input_tokens`/`output_tokens`
- **Stateful support**: Accept optional `previous_response_id` in kwargs to enable multi-turn

#### 1b. New `AzureOpenAIResponsesVisionModel` class

```python
class AzureOpenAIResponsesVisionModel(OpenAIResponsesVisionModel):
    """Azure OpenAI Vision using the Responses API."""
```

- Override `client`/`aclient` properties to use Azure-specific initialization
- Azure v1 endpoint: `https://{resource}.openai.azure.com/openai/v1/`
- Note: Azure's new v1 API doesn't need `api-version`; but support both old and new patterns

#### 1c. Configuration additions in `settings.py`

```python
# New env vars:
OPENAI_USE_RESPONSES_API = os.getenv("OPENAI_USE_RESPONSES_API", "false").lower() == "true"
AZURE_OPENAI_USE_RESPONSES_API = os.getenv("AZURE_OPENAI_USE_RESPONSES_API", "false").lower() == "true"
```

#### 1d. Update `create_default_vision_model()` and `AutoDetectVisionModel()`

Add logic to select the Responses API model when the env var is set:
```python
if USE_VISION == VisionModelProvider.openai:
    if OPENAI_USE_RESPONSES_API:
        return OpenAIResponsesVisionModel()
    return OpenAIVisionModel()
```

#### 1e. Update `__init__.py` exports

Add `OpenAIResponsesVisionModel` and `AzureOpenAIResponsesVisionModel` to `__all__`.

### Phase 2: Native PDF Support (Optional Enhancement)

The Responses API supports `input_file` with PDFs. This could allow `VisionParser` to skip the PDF→image conversion step when using a Responses API model:

- Add `process_pdf_async()` method to `OpenAIResponsesVisionModel`
- Accepts a PDF file path, base64-encodes it, sends as `input_file`
- `VisionParser` checks if the model supports `process_pdf_async()` and uses it directly

This would:
- Reduce processing overhead (no PyMuPDF rendering)
- Potentially improve quality (OpenAI handles PDF rendering internally)
- Simplify the pipeline

**Risk**: OpenAI's PDF handling quality may differ from our current DPI-controlled rendering. Needs testing.

### Phase 3: Stateful Multi-Turn Support (Optional Enhancement)

Add conversation chaining for use cases that benefit from it:

- `VidCapture` chunk processing: Use `previous_response_id` so each chunk analysis has context from prior chunks
- `VisionCapture` two-step flow: The parse → template extraction flow could chain responses
- Add `session_id` / `previous_response_id` tracking to model interface

**Benefits**:
- Reasoning token preservation across chunks (significant for o-series models)
- Reduced input token costs on multi-turn (40-80% cache improvement)
- Better coherence across video chunks

### Phase 4: Tests

- Unit tests for `OpenAIResponsesVisionModel` and `AzureOpenAIResponsesVisionModel`
- Test content format conversion (`_prepare_content` output)
- Test parameter mapping (reasoning models, max tokens, etc.)
- Test stateful conversation chaining
- Mock-based tests (no API keys needed)
- Integration test with actual API (gated by env var)

### Phase 5: Update Documentation

- Update CLAUDE.md with new model classes
- Update .env.template with new env vars
- Add examples in `examples/openai/` for Responses API usage
- Add migration notes for users switching from Chat Completions

---

## File Changes Summary

| File | Change |
|------|--------|
| `aicapture/vision_models.py` | Add `OpenAIResponsesVisionModel`, `AzureOpenAIResponsesVisionModel` |
| `aicapture/settings.py` | Add `OPENAI_USE_RESPONSES_API`, `AZURE_OPENAI_USE_RESPONSES_API` env vars |
| `aicapture/__init__.py` | Export new classes |
| `tests/test_vision_models.py` | Add tests for new classes |
| `.env.template` | Add new env var documentation |
| `CLAUDE.md` | Document new models and configuration |

---

## Risks & Considerations

1. **Gemini compatibility**: Gemini's OpenAI-compatible endpoint only supports Chat Completions, NOT the Responses API. `GeminiVisionModel` must stay on Chat Completions.

2. **openai SDK version**: The `responses` API was added in openai SDK 1.66+. Current requirement is `>=1.107.0` which already includes it. No dependency change needed.

3. **Azure endpoint migration**: Azure's Responses API uses a new `/openai/v1/` base path. Existing Azure users with the old-style endpoint would need to update their `AZURE_OPENAI_API_URL`. We should support both patterns.

4. **`store=False` default**: For our use case (stateless document processing), we should default to `store=False` to avoid OpenAI retaining conversation data. Users can opt in to storage for stateful flows.

5. **Breaking changes**: None — this is purely additive. Existing `OpenAIVisionModel` and `AzureOpenAIVisionModel` continue working exactly as before.

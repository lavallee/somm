# Tool-calling — design

**Status:** in-progress (2026-05-19). Anthropic + OpenAI providers land first; Gemini/OpenRouter/MiniMax/DeepSeek/Ollama follow.

**Driving project:** Starboard's Orca orchestrator runs on `deepagents`, which mandates a tool-calling LLM. Without tool support somm can't be the substrate for any project that builds agents. See `../starboard/docs/proposal/` for the upstream context.

## Goal

A single somm-native shape for declaring tools, sending them in a request, and reading the model's tool_use response back. Per-provider adapters translate to/from native function-call formats.

## Public surface — additions only, backward-compatible

### `SommRequest`

```python
tools: list[dict] = []           # somm-neutral tool schemas
messages: list[dict] | None = None  # multi-turn; overrides `prompt` when set
tool_choice: str | dict | None = None  # "auto" | "any" | "none" | {"type":"tool","name":X}
```

### `SommResponse` and `SommResult`

```python
tool_calls: list[ToolCall] = []
stop_reason: str = ""            # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
```

### `ToolCall`

```python
@dataclass(slots=True, frozen=True)
class ToolCall:
    id: str               # provider-assigned tool_use id; needed for tool_result correlation
    name: str             # tool name from the request schema
    arguments: dict       # parsed JSON arguments
```

### `SommLLM.generate(...)` gains

```python
tools: list[dict] | None = None,
messages: list[dict] | None = None,
tool_choice: str | dict | None = None,
```

When `tools=` is passed to a provider that doesn't support tool-calling yet, the provider raises `SommBadRequest` immediately (no silent drop). Router treats this as a permanent error for the (provider, model) pair — fall through to the next provider in the chain.

## Somm-neutral tool schema

JSON-Schema shaped, lifted from Anthropic/OpenAI's near-identical formats:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {                    # JSON Schema
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City, region"},
            },
            "required": ["location"],
        },
    },
]
```

Provider translations:

| Provider | Key rename | Wrapping |
|---|---|---|
| Anthropic | `parameters` → `input_schema` | Flat list at top-level `tools` |
| OpenAI-compat | none | Wrapped: `{"type": "function", "function": {...}}` |

## Multi-turn message format

When `messages` is set, it takes precedence over `prompt`. Format mirrors Anthropic's (closest to a neutral common denominator):

```python
messages = [
    {"role": "user", "content": "What's the weather in SF?"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "I'll check."},
        {"type": "tool_use", "id": "tu_01", "name": "get_weather",
         "input": {"location": "SF"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu_01", "content": "62°F sunny"},
    ]},
]
```

`content` can be a plain string (text-only) or a list of content blocks. Block types: `text`, `image`, `tool_use` (assistant only), `tool_result` (user only).

OpenAI adapter translates:
- assistant message with `tool_use` blocks → `{"role":"assistant","tool_calls":[{"id":...,"type":"function","function":{"name":...,"arguments":json.dumps(...)}}]}`
- user message with `tool_result` blocks → split into one or more `{"role":"tool","tool_call_id":...,"content":...}` messages.

## `tool_choice` semantics

| somm value | Anthropic | OpenAI |
|---|---|---|
| `None` (default) | `tool_choice` omitted | `tool_choice` omitted |
| `"auto"` | `{"type":"auto"}` | `"auto"` |
| `"any"` | `{"type":"any"}` | `"required"` |
| `"none"` | `{"type":"none"}` | `"none"` |
| `{"type":"tool","name":"foo"}` | `{"type":"tool","name":"foo"}` | `{"type":"function","function":{"name":"foo"}}` |

## Capability advertisement

Add `tools` to the model_intel capabilities set. Workloads that need tool-calling specify `capabilities_required=["tools"]`. Router filters provider/model pairs whose `capabilities_json` doesn't include `tools`. Unknown models still fall through as capable (per existing convention).

Seeded `model_intel` rows for known tool-capable models (Claude 3+, GPT-4+, Gemini 1.5+, etc.) get the `tools` capability set in a one-off seeding pass — separate, small follow-up commit.

## Telemetry

For this round: tool calls live on `SommResult.tool_calls` and inside `raw`. No schema migration. The `calls.sqlite` `calls` row keeps its current shape; `tool_calls_count` derivable from `raw_json` if needed later.

Schema bump (proposed 0008) deferred — adds dedicated columns (`tool_calls_count`, `tools_offered_count`, `stop_reason`) once we have a real workload accumulating calls and we know what queries we want to write against them.

## Out of scope tonight

- Streaming tool calls (incremental tool-arg deltas over SSE).
- Tool-choice cost shaping (e.g., banning expensive tools when budget tight).
- Tool-result auto-truncation policy when results exceed model context.
- Parallel tool calls in a single turn — supported by passing the response through unchanged; no special handling needed.

## Test strategy

Per-provider tests in `packages/somm/tests/test_provider_adapters.py` extension:
- request shape: assert outbound payload contains tools, tool_choice, multi-turn messages translated correctly
- response parsing: assert `tool_calls` extracted from `tool_use` blocks; `stop_reason` set
- error path: `SommBadRequest` if provider doesn't support tools yet (for providers landing in tomorrow's pass)

Integration test in `test_library_ext.py` adds: `SommLLM.generate(tools=...)` end-to-end with mocked Anthropic transport.

## Hand-off plan

Tonight (2026-05-19):
- This spec
- `ToolCall`, `tools`/`messages`/`tool_choice`/`tool_calls`/`stop_reason` on the request/response dataclasses
- AnthropicProvider tool support + tests
- OpenAICompatProvider tool support + tests (covers OpenAI; minimax/openrouter/deepseek inherit)
- `SommLLM.generate(tools=...)` threading + 1 integration test
- README mention, CHANGELOG entry

Tomorrow (or later):
- GeminiProvider tool support (different format: `functionDeclarations` under `tools`)
- OllamaProvider tool support (`tools` field per Ollama 0.4+ — verify version on the box)
- model_intel seeding pass for `tools` capability
- Schema 0008 telemetry columns once a real workload exists

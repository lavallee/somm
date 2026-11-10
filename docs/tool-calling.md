# Tool-calling — design

**Status:** shipped (2026-05-19). All providers support tools; see CHANGELOG for the rollout.

**Motivation:** agent frameworks (`deepagents`, LangChain agents, and similar) mandate a tool-calling LLM. Without tool support somm can't be the substrate for any project that builds agents.

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

## Streaming

`stream(..., tools=..., events=True)` yields `somm.events.ToolCallDelta(index, id,
name, arguments)` fragments as they arrive: `id` and `name` on a call's first
fragment, then pieces of its JSON argument string. The final `StreamDone.result`
carries the assembled `tool_calls`, exactly as `generate` returns them. Adapters
that answer in one piece still emit one fragment per call.

## Telemetry

Tool calls live on `SommResult.tool_calls` and inside `raw_json`; the `calls`
row has no dedicated tool columns. Lifting `tool_calls_count`,
`tools_offered_count` and `stop_reason` into columns is deferred until a query
needs them (ROADMAP.md, "Deferred").

## Not handled

- Tool-choice cost shaping (e.g., banning expensive tools when budget is tight).
- Tool-result truncation when results exceed the model's context.
- Parallel tool calls in one turn get no special handling; they pass through
  unchanged.

## Tests

Per-provider request and response shapes are in
`packages/somm/tests/test_provider_adapters.py`; the end-to-end
`generate(tools=...)` path is in `packages/somm/tests/test_library_ext.py`.

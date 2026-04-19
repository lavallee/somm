# Multimodal prompts + capability-aware routing

`somm` accepts content-block prompts (text + image) and routes them only
to (provider, model) pairs known to serve the required capabilities.

## Two ways to send an image

```python
import somm
from somm_core.parse import image_prompt, text_prompt

llm = somm.llm(project="my_app")

# Inline bytes
png = open("chart.png", "rb").read()
result = llm.generate(
    prompt=image_prompt("Critique this chart.", png, media_type="image/png"),
    workload="critique_visual",
)

# Or by URL (providers that support it, e.g. OpenAI)
result = llm.generate(
    prompt=image_prompt("What's in this?", url="https://example.com/x.png"),
    workload="vision_test",
)

# Plain strings still work:
result = llm.generate(prompt="text only", workload="default")
```

Under the hood the prompt is passed to each provider as
`messages[0].content` — a list of blocks. Anthropic, OpenAI, and
OpenRouter all accept this shape natively.

## Capability-aware routing

When the prompt contains an image block, the router auto-adds
`"vision"` to the required capabilities and filters out providers
whose default model can't serve vision. No late 400s; no wasted
fallback to a text-only model.

```python
llm.register_workload(
    name="critique_visual",
    capabilities_required=["vision"],  # declare once; applies to every call
)
```

Explicit per-request override:

```python
result = llm.generate(
    prompt="long doc...",
    workload="summarize",
    capabilities_required=["long_context"],
)
```

If no provider in the chain can serve the required capabilities, the
router raises `SommNoCapableProvider` (`SOMM_NO_CAPABLE_PROVIDER`)
before any network call, with a list of `(provider, model, reason)`
tuples so you can see the gap at a glance.

## Capability vocabulary

| Token | Means |
|---|---|
| `vision` | Accepts image content blocks |
| `long_context` | Context window ≥ 100K (TBD threshold) |
| `tool_use` | Supports function/tool calling |
| `json_mode` | Supports strict JSON output |
| `thinking` | Supports extended/adaptive thinking |
| `streaming` | Supports SSE chunked responses |

Unknown capability values don't block — they allow the provider to
try. Only an **explicit** `False` in `model_intel.capabilities_json`
filters a model out. This preserves the "let the provider try when
unsure" default.

## What the router knows

Capabilities come from `model_intel.capabilities_json` populated by the
intel-refresh worker:

- **OpenRouter** publishes `modality` + `architecture.input_modalities`
  on every model; `vision` is derived automatically.
- **Anthropic / OpenAI** static pricing seeds use name-hint inference —
  `claude-opus-4-*`, `claude-sonnet-4-*`, `claude-haiku-4-*`, `gpt-4o*`
  all carry vision.
- **Ollama**: `llava`, `bakllava`, `llama3.2-vision` etc. inferred by
  family; unknown local models fall through as capable.
- **Minimax**: treated as capability-unknown; the adapter will surface
  a 400 if you send vision to a text-only model.

## Telemetry

Prompt hashing uses `somm_core.parse.stable_hash`, which canonicalises
list prompts to JSON before hashing. Two equivalent image prompts hash
identically. The samples table (opt-in) stores
`prompt_preview(prompt)` rather than raw base64, so enabling
sampling on a vision workload does not bloat storage.

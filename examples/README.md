# somm examples

Runnable examples showing common patterns for grafting somm into an existing
Python project. All examples use synthetic workload names (`contact_extract`,
`qa`, `medical_extract`) — swap in your own.

## Setup

```bash
uv add --editable .            # from your project root, pointing at the somm checkout
# or
pip install -e /path/to/somm
```

Set an env var for the default provider chain — at minimum, point somm at
a local ollama:

```bash
export SOMM_OLLAMA_URL=http://localhost:11434
export SOMM_OLLAMA_MODEL=gemma4:e4b
# Optional commercial providers (all opt-in, sovereign-by-default):
export OPENROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

## Examples

### `drop_in_wrapper.py` — replace a project's existing LLM class

Change **one import** to get telemetry + routing + cost tracking under your
existing call sites:

```python
# before:
from myproject.llm import FooLLM
# after:
from somm.compat import GenericLLMCompat as FooLLM
```

Shown: `.generate(prompt, system, max_tokens, provider)` drop-in,
`.probe_providers(n)` for parallel work striping, cost inspection.

### `openai_swap_in.py` — replace raw OpenAI SDK calls

Change one import + one function, keep the OpenAI-SDK response shape:

```python
# before:
from openai import OpenAI
resp = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=[...])
# after:
from somm.compat import openai_chat_completions as create
resp = create(model="openai/gpt-4o-mini", messages=[...], project="my_project")
```

Shown: provider pinning via `provider/model` prefix, routed fallback,
somm-native cost + latency on the response object.

### `private_workload.py` — privacy_class=PRIVATE for sensitive data

Use for any workload you don't want egressing to upstream providers
(PII, PHI, private comms). The router enforces the gate; shadow-eval
skips private workloads; `enable_shadow()` raises `SommPrivacyViolation`.

## After grafting

Verify telemetry is landing:

```bash
somm status --project my_project --since 1
somm tail --project my_project
```

Stand up the dashboard:

```bash
somm serve --project my_project
# open http://localhost:7878
```

Refresh model pricing:

```bash
somm-serve admin refresh-intel --project my_project
somm-serve admin list-intel --provider openrouter
```

Compare two models on a single prompt:

```bash
somm compare "Summarize this article in one sentence" \
  --models ollama/gemma4:e4b,openai/gpt-4o-mini \
  --project my_project
```

## Next step

When you're happy with the drop-in, configure shadow-eval to start
building a quality dataset:

```python
llm.enable_shadow(
    workload="contact_extract",
    gold_provider="anthropic",       # or ollama for sovereignty
    gold_model="claude-haiku-4-5-20251001",
    sample_rate=0.02,                # 2% of calls get re-run
    budget_usd_daily=0.50,
)
```

The agent worker will pick up shadow-graded data within a week and start
emitting recommendations (`switch_model`, `new_model_landed`,
`chronic_cooldown`). See them at `localhost:7878` or via `somm doctor`.

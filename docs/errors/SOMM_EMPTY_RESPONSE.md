# SOMM_EMPTY_RESPONSE

**Problem.** A provider call succeeded at the HTTP layer but returned
no usable content — empty text and no tool calls.

**Why.** Common triggers:

- A thinking/reasoning model burned its entire `max_tokens` budget on
  internal reasoning and had nothing left for the visible answer.
- A CLI-backed provider (`codex exec`) produced no output at all.
- A model refused or stopped early in a way that produced a blank
  completion rather than an explicit error.

**Behavior in the router.** Empty responses are treated as **transient**
in two different ways depending on where they're caught:

1. **Mid-chain (silent fallback).** When a provider returns text that's
   blank (and no `tool_calls`, and the request didn't set
   `allow_empty=True`), the router does *not* raise — it records a
   short cooldown (15s) on that provider and tries the next one. You
   only see this in logs/telemetry, not as an exception, unless every
   provider ends up empty.
2. **Explicit exception.** `SOMM_EMPTY_RESPONSE` is actually raised to
   the caller in two cases: a CLI-backed provider (`codex-cli`)
   produced literally no output, or you called with
   `raise_on_empty=True` and every provider in the chain ultimately
   returned empty text.

**Fix.**

1. If you set `raise_on_empty=True`, decide whether empty is really
   fatal for your use case. If not, drop the flag and handle empty
   text downstream:
   ```python
   llm.generate(prompt, workload="my_workload", raise_on_empty=False)
   ```

2. If a thinking model is the culprit, increase `max_tokens` so
   reasoning doesn't consume the whole budget, or switch to a
   non-thinking model for this workload:
   ```python
   llm.generate(prompt, workload="my_workload", max_tokens=2048)
   ```

3. If prompts genuinely produce no output sometimes (e.g. a filter
   workload that's supposed to output nothing on no-match), tell somm
   that's expected:
   ```python
   from somm.providers.base import SommRequest
   SommRequest(prompt=..., allow_empty=True)
   ```

4. For `codex-cli`, reproduce manually to see why it produced nothing:
   ```bash
   codex exec "your prompt here"
   ```

**Related.**
- [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) — every
  provider cooled, potentially from repeated empty responses.

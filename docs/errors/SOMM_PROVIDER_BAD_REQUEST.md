# SOMM_PROVIDER_BAD_REQUEST

**Problem.** A provider rejected the call as malformed (HTTP 400/404),
or somm itself refused to build the request because it's structurally
invalid for that provider.

**Why.** Common triggers:

- The requested model doesn't exist on that provider (404 — check for
  typos in `SOMM_*_MODEL` env vars).
- No model was configured or requested at all for a provider with no
  default.
- `tools` were passed to a provider/adapter that doesn't support tool
  calling yet — somm raises this locally rather than letting the
  provider fail confusingly.
- A prompt shape the provider can't parse (e.g. malformed multi-turn
  `messages`).
- `ollama` raised a provider-specific bad-request (e.g. asked for a
  model that isn't pulled locally).

**Behavior in the router.** `SOMM_PROVIDER_BAD_REQUEST` is **fatal**.
The router does not cool the provider or fall through — a malformed
request will fail identically on retry, and it will very likely fail
the same way on every other provider too, so failing loud immediately
is more useful than burning through the whole chain.

**Fix.**

1. Read the exception detail — it includes the provider's raw error
   body (truncated) which usually names the exact problem.

2. If it's a missing model, pull it or fix the model name:
   ```bash
   ollama pull qwen3:8b              # explicit local fallback
   export SOMM_OPENAI_MODEL=gpt-4o-mini   # cloud, fix typo/availability
   ```

3. If it's a tool-calling request against a provider that doesn't
   support tools yet, either drop `tools` from the request or route
   that workload to a provider that does support it (check
   `docs/tool-calling.md`).

4. If no model was configured at all, set a default:
   ```bash
   export SOMM_ANTHROPIC_MODEL=claude-haiku-4-5-20251001
   ```

**Related.**
- [`SOMM_NO_CAPABLE_PROVIDER`](./SOMM_NO_CAPABLE_PROVIDER.md) — caught
  earlier, before any network call, when no provider in the chain
  advertises the required capability at all.
- [`SOMM_PROVIDER_AUTH`](./SOMM_PROVIDER_AUTH.md) — credentials
  problem, not a malformed request; also fatal.

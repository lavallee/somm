# SOMM_PROVIDERS_EXHAUSTED

**Problem.** Every provider in somm's routing chain is currently in
cooldown, and the wait until the next one recovers exceeds
`exhausted_sleep_cap_s` (default: 300s).

**Why.** Common triggers:

- Local `ollama` is down or unreachable.
- Free-tier providers (OpenRouter) hit rate limits simultaneously.
- Consecutive failures on the only configured provider tripped its
  circuit breaker.

**Fix.**

1. Check which providers are currently cooling:
   ```bash
   somm doctor   # shows active cooldowns with time remaining
   ```

2. For local ollama issues:
   ```bash
   curl http://localhost:11434/api/tags   # should 200
   ollama serve   # if not running
   ```

3. For free-tier exhaustion, add fallback providers to your env:
   ```bash
   export ANTHROPIC_API_KEY=...   # adds anthropic to chain
   export OPENAI_API_KEY=...      # adds openai
   ```

4. To manually clear a provider's cooldown (e.g., you know it's back
   up and want to stop waiting):
   ```bash
   # From Python:
   from somm.client import SommLLM
   llm = SommLLM()
   llm._tracker.clear("openrouter")   # or specific (provider, model)
   ```

5. Tune cooldown aggressiveness in your app:
   ```python
   from somm.client import SommLLM
   from somm.routing import Router
   llm = SommLLM()
   # example: shorter-than-default circuit break
   llm.router.circuit_break_after = 10
   llm.router.circuit_break_cooldown_s = 300
   ```

**Note.** somm intentionally refuses to block forever. When every
provider is down for a long window, the call fails loud rather than
hanging — that's safer for batch jobs and servers.

**Related.**
- [`SOMM_SCHEMA_STALE`](./SOMM_SCHEMA_STALE.md) — different issue; DB
  schema needs migration.

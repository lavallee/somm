# SOMM_PROVIDER_TIMEOUT

**Problem.** A provider call didn't return before the configured request
timeout expired.

**Why.** Common triggers:

- The upstream API is slow or overloaded and didn't respond in time.
- A long prompt + high `max_tokens` pushed generation past the timeout
  window.
- A CLI-backed provider (`claude-cli`, `codex-cli`) spawned a subprocess
  that hung or took longer than its `timeout` setting.
- Local network issues (VPN, flaky wifi, DNS) between you and the
  provider.

**Behavior in the router.** `SOMM_PROVIDER_TIMEOUT` is **transient**.
The router cools the provider (default 60s) and falls through to the
next one in the chain. If every provider times out, you'll eventually
see [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) instead.

**Fix.**

1. If this is a one-off, do nothing — the router already retried the
   next provider and the call likely succeeded.

2. If it's persistent on one provider, raise its timeout:
   ```python
   from somm.providers.anthropic import AnthropicProvider
   provider = AnthropicProvider(api_key="...", timeout=120.0)
   ```

3. For CLI-backed providers, check the subprocess isn't hung:
   ```bash
   ps aux | grep -E "claude -p|codex exec"
   ```

4. Check the provider's status page if timeouts cluster across
   requests — it may be a real upstream outage rather than your
   network.

5. Add a second provider to your chain so a single slow provider
   doesn't stall every call:
   ```bash
   export OPENROUTER_API_KEY=...
   ```

**Related.**
- [`SOMM_PROVIDER_5XX`](./SOMM_PROVIDER_5XX.md) — provider responded,
  just with an error status; different failure mode, same transient
  handling.
- [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) — every
  provider cooled, including from repeated timeouts.

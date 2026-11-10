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
- The provider itself answered HTTP 408 (OpenRouter and the ChatGPT
  connection do); `status_code` is then 408.

One configured value sets two limits on each HTTP request: connecting may
take at most 10 s (or the whole value, if smaller), while reading, writing
and waiting for a pooled connection get the whole value. Connections are
kept open and reused between calls, so most calls do not connect at all.

**Behavior in the router.** `SOMM_PROVIDER_TIMEOUT` is **transient**.
The router cools the provider (default 60s) and falls through to the
next one in the chain. If every provider times out, you'll eventually
see [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) instead.

**Fix.**

1. If this is a one-off, do nothing — the router already retried the
   next provider and the call likely succeeded.

2. If one workload legitimately runs long, give its calls more time.
   `timeout=` on the call reaches every provider adapter;
   `SOMM_HTTP_TIMEOUT` (default 180 s) changes it for every HTTP
   provider, and a workload policy's `timeout_s` for one workload:
   ```python
   result = llm.generate(prompt, workload="deep_summary", timeout=600)
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

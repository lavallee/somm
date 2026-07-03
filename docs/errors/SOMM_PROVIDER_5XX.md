# SOMM_PROVIDER_5XX

**Problem.** A provider returned an HTTP 5xx (server-side error), or a
CLI-backed provider exited non-zero / returned unparseable output.

**Why.** Common triggers:

- Genuine upstream outage or degraded service on the provider's side.
- A transient overload response (Anthropic's 529 "overloaded" is folded
  into the same transient bucket, with its own short cooldown).
- A CLI-backed provider (`claude-cli`, `codex-cli`) exited with a
  non-zero code or produced output that isn't valid JSON.

**Behavior in the router.** `SOMM_PROVIDER_5XX` is **transient**. The
router cools the provider (default 30s) and falls through to the next
one in the chain.

**Fix.**

1. Usually nothing to do — the router already tried the next provider.

2. Check the provider's public status page to confirm it's a known
   outage rather than something on your end.

3. If it's a CLI-backed provider, reproduce manually to see the raw
   error:
   ```bash
   claude -p "test" --output-format json
   codex exec "test"
   ```

4. If 5xx responses cluster on one provider, temporarily drop it from
   your chain until it recovers:
   ```bash
   unset ANTHROPIC_API_KEY   # or whichever provider is flapping
   ```

**Related.**
- [`SOMM_PROVIDER_TIMEOUT`](./SOMM_PROVIDER_TIMEOUT.md) — provider
  never responded at all, rather than responding with an error status.
- [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) — every
  provider cooled, including from repeated 5xx.

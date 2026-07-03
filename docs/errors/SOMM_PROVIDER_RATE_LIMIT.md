# SOMM_PROVIDER_RATE_LIMIT

**Problem.** A provider returned HTTP 429 (or an equivalent rate-limit
signal in the response body).

**Why.** Common triggers:

- Free-tier or low-tier API keys with tight requests-per-minute caps.
- A burst of calls (batch job, retry storm) exceeded the provider's
  per-key or per-org limit.
- Shared org-level limits being consumed by other apps using the same
  key.

**Behavior in the router.** `SOMM_PROVIDER_RATE_LIMIT` is **transient**.
The router cools the provider for `retry_after_s` (the provider's
`Retry-After` header when present, otherwise a default of 120s) and
falls through to the next provider in the chain. The cooldown length is
carried on the exception as `.retry_after_s`.

**Fix.**

1. Let the router handle it — if you have more than one provider
   configured, the call already fell through and likely succeeded.

2. If it's your only provider, add a fallback so bursts don't stall:
   ```bash
   export OPENROUTER_API_KEY=...   # or ANTHROPIC_API_KEY, OPENAI_API_KEY
   ```

3. Reduce request rate — batch or throttle calls on your side, or
   spread traffic across workloads with different keys.

4. Check whether a higher API tier removes the limit (most providers
   raise RPM caps after spend history or a paid tier upgrade).

5. Inspect the actual wait time from the exception if you're handling
   it explicitly:
   ```python
   from somm.errors import SommRateLimited
   try:
       llm.generate(prompt, workload="my_workload")
   except SommRateLimited as e:
       print(f"retry after {e.retry_after_s}s")
   ```

**Related.**
- [`SOMM_PROVIDER_5XX`](./SOMM_PROVIDER_5XX.md) — upstream failure, not
  a quota limit.
- [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) — every
  provider cooled, including from simultaneous rate limits.

# SOMM_PROVIDER_RATE_LIMIT

**Problem.** A provider returned HTTP 429 (or an equivalent rate-limit
signal in the response body).

**Why.** Common triggers:

- Free-tier or low-tier API keys with tight requests-per-minute caps.
- A burst of calls (batch job, retry storm) exceeded the provider's
  per-key or per-org limit.
- Shared org-level limits being consumed by other apps using the same
  key.

Not every 429 is a rate limit. MiniMax answers an exhausted Token Plan
with 429 and code `2056`; OpenAI answers exhausted credit or a spend
limit with 429 `insufficient_quota` and a code such as
`credit_balance_exhausted`; Anthropic answers a reached monthly spend cap
with 429 and `enforced_spend_limit_reached`. somm classifies those as
[`SOMM_PROVIDER_INSUFFICIENT_CREDIT`](./SOMM_PROVIDER_INSUFFICIENT_CREDIT.md)
with its longer cooldown, so a caller that sees this error can treat it
as a genuine burst.

**Behavior in the router.** `SOMM_PROVIDER_RATE_LIMIT` is **transient**.
The router cools the provider for `retry_after_s` and falls through to
the next provider in the chain. `retry_after_s` is the provider's
`Retry-After` header when present (`0` means retry now); otherwise the
latest of its rate-limit reset headers (Anthropic's
`anthropic-ratelimit-*-reset` times, OpenAI's `x-ratelimit-reset-*`
durations such as `6m0s`); otherwise a default of 120s. It is capped at a
day. The exception also carries `status_code`, `provider_type`,
`provider_code` and `provider_request_id` (see [`SOMM_ERROR`](./SOMM_ERROR.md)).

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

# SOMM_PROVIDER_AUTH

**Problem.** A provider rejected the call as unauthenticated or
unauthorized (HTTP 401/403, or an auth-shaped error embedded in a 200
response body).

**Why.** Common triggers:

- Missing, expired, or revoked API key.
- The key is valid but lacks access to the requested model or
  endpoint.
- A typo'd env var name, so somm picked up an empty or wrong key.

**Behavior in the router.** `SOMM_PROVIDER_AUTH` is **fatal**. Unlike
timeouts, rate limits, or 5xx, the router does **not** cool the
provider and try the next one — it raises immediately and aborts the
whole call. A bad key won't fix itself by waiting, and silently falling
back could mask a misconfiguration you need to see.

**Fix.**

1. Check the relevant env var is set and non-empty:
   ```bash
   echo "${ANTHROPIC_API_KEY:0:8}..."   # sanity-check without printing it all
   ```

2. Verify the key is still valid directly against the provider (bypass
   somm to isolate the problem):
   ```bash
   curl -s https://api.anthropic.com/v1/models \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" | head -c 300
   ```

3. Rotate the key if it was revoked or expired, then re-export it and
   restart your process (env vars are read at provider construction
   time).

4. If the key is fine but the model isn't, confirm your account/tier
   actually has access to the model somm is requesting:
   ```bash
   export SOMM_ANTHROPIC_MODEL=claude-haiku-4-5-20251001
   ```

**Related.**
- [`SOMM_PROVIDER_INSUFFICIENT_CREDIT`](./SOMM_PROVIDER_INSUFFICIENT_CREDIT.md)
  — key is valid but the account is out of money/quota; transient, not
  fatal.
- [`SOMM_PROVIDER_BAD_REQUEST`](./SOMM_PROVIDER_BAD_REQUEST.md) — the
  request itself is malformed, not a credentials problem.

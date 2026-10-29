# SOMM_PROVIDER_AUTH

**Problem.** A provider rejected the call as unauthenticated or
unauthorized (HTTP 401/403, or an auth-shaped error embedded in a 200
response body).

**Why.** Common triggers:

- Missing, expired, or revoked API key.
- The key is valid but lacks access to the requested model or
  endpoint.
- A typo'd env var name, so somm picked up an empty or wrong key.

**Behavior in the router.** A bad key is that provider's problem, not the
request's. Since 0.19.2:

- **Unpinned calls** route past it. The router cools the provider for
  5 minutes (longer if the key keeps failing, on the usual escalation
  ladder), logs a warning, and tries the next provider in the chain. If
  another provider answers, the call succeeds and `on_fallback` fires with
  `pinned_provider=None` and `routed_past=["<provider>"]`, so the bad key is
  still reported. If nothing answers, the call fails with
  `SOMM_PROVIDER_AUTH` itself, not `SOMM_PROVIDERS_EXHAUSTED`, because
  waiting won't fix a key.
- **Pinned calls** (`provider=` given) stay sticky, as with every other
  error: the call fails with the auth error and the detail names the pin.
  `generate_structured` stops after the first auth failure instead of
  spending its retries on the same rejected key.

Before 0.19.2 an auth error aborted the whole call. One bad key at the
head of the chain then failed every unpinned call, even when a healthy
provider sat behind it (issue #71).

Cooldowns are stored in the ledger and survive restarts. After you rotate
a key, the provider can stay skipped for up to the remaining cooldown.
`somm doctor` lists the active cooldowns.

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

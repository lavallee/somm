# SOMM_PROVIDER_INSUFFICIENT_CREDIT

**Problem.** A provider rejected the call because its account balance
or quota is exhausted — not because the request was malformed or the
key is invalid.

**Why.** Providers surface this as an auth-ish or bad-request-ish HTTP
status (Anthropic returns 400 with "credit balance is too low", OpenAI
returns 429 with `insufficient_quota`), but somm reclassifies it: it's
neither a bad request nor a bad key, it's a *billing state* on that one
provider. The request itself is fine and would succeed on another
provider. somm matches known billing-signal phrases in the response
body (case-insensitively, e.g. `"insufficient_quota"`, `"credit
balance is too low"`, `"exceeded your current quota"`) to make this
call.

**Behavior in the router.** `SOMM_PROVIDER_INSUFFICIENT_CREDIT` is
**transient**, deliberately with a long cooldown (default 3600s / 1
hour) rather than the usual short one — a balance won't replenish in
seconds, so somm sidelines the lapsed provider instead of hammering it
on every subsequent call. The router falls through to the next provider
in the chain.

**Fix.**

1. Top up or re-enable billing on the flagged provider account.

2. In the meantime, the router is already skipping this provider for
   an hour — no action needed if you have a fallback provider
   configured.

3. If you don't want to wait out the cooldown after topping up, clear
   it manually:
   ```python
   from somm.client import SommLLM
   llm = SommLLM()
   llm._tracker.clear("anthropic")   # or whichever provider you fixed
   ```

4. If this fires on a provider that's actually fine (a false positive
   phrase match), check the raw response body in the exception message
   — false positives are rare by design (the phrase list is narrow) but
   worth confirming before assuming it's really a billing issue.

**Related.**
- [`SOMM_PROVIDER_AUTH`](./SOMM_PROVIDER_AUTH.md) — genuinely bad or
  missing credentials, not a billing state; fatal, not transient.
- [`SOMM_PROVIDER_BAD_REQUEST`](./SOMM_PROVIDER_BAD_REQUEST.md) — a
  malformed request; also fatal, also distinct from a billing state.

# SOMM_BUDGET_EXCEEDED

**Problem.** A workload's accumulated spend for today has reached its
configured daily budget cap, and the fail-closed budget gate refused
the call *before* dispatch.

**Why.** This is an opt-in gate, not a default behavior — it only
fires when both are true:

- `budget_fail_closed` is on (`SOMM_BUDGET_FAIL_CLOSED=1`), and
- the workload has a cap, either its own `budget_cap_usd_daily` or the
  project-wide `budget_default_cap_usd_daily`
  (`SOMM_BUDGET_DEFAULT_CAP_USD_DAILY`).

Enforcement is on *committed* daily spend (rows already written to
`calls`), checked with a `SELECT SUM(cost_usd) ... WHERE date(ts) =
date('now')` query. Because telemetry writes drain asynchronously
during the seconds-long LLM call, committed spend tracks Somm's recorded
amount (normally a token-times-price estimate)
closely but not instantly — the call that crosses the cap still
completes, and only the *next* call is blocked. Overshoot is bounded to
roughly one call plus whatever hadn't flushed yet.

**Behavior in the router.** `SOMM_BUDGET_EXCEEDED` is **fatal** and
deliberately not transient — the router must not cool a provider and
fall through to try another one, because that would defeat the whole
point of a ceiling. No telemetry row is written for a blocked call: a
refusal isn't a spend event.

**Fix.**

1. Raise the workload's cap if the spend is legitimate:
   ```python
   llm.register_workload(
       name="my_workload",
       budget_cap_usd_daily=5.00,   # was lower / unset
   )
   ```

2. Wait for the UTC-day reset — the cap resets at `date('now')`
   midnight UTC, not local time.

3. Turn off the gate entirely if you don't want fail-closed
   enforcement right now (spend still gets tracked and soft-warned,
   just not blocked):
   ```bash
   unset SOMM_BUDGET_FAIL_CLOSED
   ```

4. Inspect the exception for exact numbers if you're handling it in
   code:
   ```python
   from somm.errors import SommBudgetExceeded
   try:
       llm.generate(prompt, workload="my_workload")
   except SommBudgetExceeded as e:
       print(e.workload, e.spent_usd, e.cap_usd)
   ```

**Note.** This gate is shared between the `SommLLM` library path and
the `somm-service` `/v1/messages` and `/v1/chat/completions` proxies —
one ledger, one policy, regardless of which path a call comes through.

**Related.**
- [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) — a
  routing failure across providers; unrelated to spend, and always
  raised *after* an attempt was made.

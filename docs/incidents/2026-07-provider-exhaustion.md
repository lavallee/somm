# Incident: month-long SommProvidersExhausted storm (one adopter project, 2026-06-22 → 2026-07-27)

**Status:** resolved. **Severity:** high (majority of one adopter's LLM calls
failed for ~5 weeks). **Detection:** none automated — found via manual
telemetry review while building the shadow-eval judgment layer (this repo,
2026-08). No alert fired at the time.

## Summary

From late June through late July 2026, the adopter project's call-ok rate
collapsed from ~99% to single digits, with the overwhelming majority of
failures raising `SommProvidersExhausted` (`SOMM_PROVIDERS_EXHAUSTED`) — every
provider in the routed chain was cooling down simultaneously. Root cause:
`minimax/MiniMax-M2.7`, the model both somm's own default chain and the adopter's
explicit `SOMM_PROVIDER_ORDER` pin routed to, was saturated — hitting
aggressive per-window rate limits hard enough that cooldowns on that model
never fully cleared before the next batch of calls hit it again, producing a
self-sustaining failure storm. The fix was retiring the M2.7 default in favor
of `minimax/MiniMax-M3` (commit `ae4c606`, 2026-07-23), paired with a matching
change to the adopter's own `.env` the evening before. Recovery was immediate and
durable — call-ok rate jumped from ~18% to ~78% within a day and has held in
the mid-80s since.

## Timeline

All timestamps from `calls.ts` in the affected project's local telemetry DB
(project-local `.somm/calls.sqlite`, weekly rollup by `strftime('%Y-W%W', ts)`):

| Week | Date range | Calls | ok rate | exhausted rate |
|---|---|---:|---:|---:|
| W24 | 06-20 .. 06-21 | 980 | 98.9% | 1.1% |
| W25 | 06-22 .. 06-28 | 31,624 | 62.8% | 31.1% |
| W26 | 06-29 .. 07-05 | 45,052 | 70.0% | 28.7% |
| W27 | 07-06 .. 07-12 | 98,752 | 29.9% | 70.1% |
| **W28** | **07-13 .. 07-19** | **306,734** | **8.8%** | **91.1%** |
| W29 | 07-20 .. 07-26 | 169,535 | 18.2% | 81.1% |
| W30 | 07-27 .. 08-02 | 49,155 | 78.3% | 21.3% |
| W31 | 08-03 .. 08-09 | 44,166 | 87.0% | 12.5% |
| W32 | 08-10 .. 08-16 | 44,025 | 86.5% | 12.9% |

W28 is the trough: 91.1% of ~307k calls that week failed with
`SommProvidersExhausted`. Total `SommProvidersExhausted` rows across the
whole incident window: **536,862** — consistent with the "500k+" figure in
the mission brief.

Daily detail around the recovery inflection:

| Date | Calls | ok rate | exhausted rate |
|---|---:|---:|---:|
| 07-18 | 52,964 | 6.6% | 93.4% |
| 07-20 | 25,946 | 10.3% | 87.4% |
| 07-21 | 55,706 | 4.6% | 94.6% |
| 07-22 | 54,906 | 5.0% | 95.0% |
| **07-23** | **10,470** | **61.1%** | **38.5%** |
| 07-24 | 6,175 | 86.0% | 13.5% |
| 07-27 | 6,257 | 89.9% | 9.6% |

The break is sharp and lands exactly on the fix landing, not a gradual
recovery — consistent with a config/default change rather than the underlying
provider quietly restoring capacity. (Call *volume* also drops an order of
magnitude at the same point, from ~55k/day to ~6-10k/day — the retry/backoff
storm itself was inflating call counts; once providers stopped exhausting,
far fewer retries were needed per unit of real work.)

## Root cause

1. **`minimax/MiniMax-M2.7` was the routed default.** Before the fix,
   `packages/somm/src/somm/providers/registry.py` ordered built-in providers
   with `ollama` at priority 10 (first) and `minimax` at priority 40; the
   `Config` defaults pinned `minimax_model = "MiniMax-M2.7"`. The adopter's own
   `.env` additionally set `SOMM_PROVIDER_ORDER=minimax,deepseek,openrouter`
   and (at the time) a MiniMax-M2.7 pin, so nearly all of the adopter's traffic
   funneled through that one model.

2. **M2.7 had a hard, aggressive rate ceiling.** `~/.somm/plans.toml` records
   an observed 5h ceiling of 44.5M tokens_total for minimax, sourced from
   "median of 5 real 429 events (2026-07)" — i.e. the operator was watching
   real 429s from this exact incident while it was happening. Once traffic
   volume pushed the model chain past that ceiling, `somm`'s router put the
   model into cooldown; with `minimax` first (or only) in the effective
   chain and no other provider capable/authorized to pick up the slack,
   every subsequent call in the window raised `SommProvidersExhausted`
   (`packages/somm/src/somm/routing.py`).

3. **The failure mode was self-reinforcing.** `SommProvidersExhausted` is
   retryable (`packages/somm/src/somm/errors.py: retryable = isinstance(...,
   (SommTransientError, SommProvidersExhausted))`), so callers kept retrying
   into a chain that was still cooling — driving call volume *up* (W28 alone
   is 306k calls, ~3x a normal week) while ok-rate cratered. Commit
   `5753477` ("opt-in wait-with-deadline when all routed providers are
   cooling", 2026-07-08, mid-incident) explicitly cites this pattern:
   *"Cooldown storms produced failure storms: when every provider in a chain
   cooled simultaneously, each call burned an exhausted error row and batch
   jobs churned thousands of them (28% of one adopter's monthly calls)."*
   This shipped a `wait=<seconds>` opt-in for `generate()`/`stream()`/
   `extract_structured()`, but being opt-in it did not by itself resolve
   the adopter's exhaustion (the incident continued for another two weeks after
   this commit landed).

4. **`somm plans --learn` existed but wasn't enforcing.** Commit `c41492d`
   ("Learn plan limits from quota errors", 2026-07-09) added the machinery to
   persist observed 429 ceilings into `plans.toml` and hard-block once a
   *learned* limit is reached when a plan has `enforce = true`. The minimax
   plan entry in `plans.toml` has `enforce = false` — the learned ceiling was
   recorded and visible, but not used to proactively throttle before hitting
   the wall. This is a real gap: the tooling to prevent the storm before it
   started existed for two weeks before the actual fix, unused.

## What actually fixed it

Commit `ae4c606` — *"Retire Gemma defaults in favor of MiniMax M3 (#47)"*
(2026-07-23 08:31 EDT) — changed:

- `packages/somm/src/somm/providers/registry.py`: `minimax` priority
  40 → 10 (now first in the default chain), `ollama` priority 10 → 90
  (now last, explicit-fallback-only).
- `packages/somm-core/src/somm_core/config.py` and
  `packages/somm/src/somm/providers/minimax.py`: default `minimax_model`
  `"MiniMax-M2.7"` → `"MiniMax-M3"`.

the adopter's own `.env` (untracked, `mtime` 2026-07-22 20:22 EDT — the evening
before the repo commit) was updated in lockstep to
`SOMM_MINIMAX_MODEL=MiniMax-M3` and `SOMM_PROVIDER_ORDER=minimax,deepseek,openrouter`
(current state, confirmed by direct read — not modified by this work).

Moving traffic off `MiniMax-M2.7` onto `MiniMax-M3` — a different quota pool
— relieved the ceiling pressure directly. This reads as capacity
management (move off a saturated model) rather than a code-level guard
against exhaustion; no rate-limiting or backpressure logic changed in this
commit.

## Residual risk / what guards it now

- **`somm plans --learn`** (from `c41492d`) still records real 429 ceilings
  into `plans.toml` per provider/window, visible via `somm plans`. Still
  `enforce = false` for minimax — a repeat of this incident on a *new*
  saturated model would not be auto-blocked, only loggable after the fact.
- **`wait=<seconds>`** (from `5753477`) lets a caller ride out a simultaneous
  cooldown instead of hammering it with retries, but remains opt-in per call
  — nothing in the adopter's own code was confirmed to use it.
- **`somm doctor`** already surfaces active cooldowns
  (`provider_health.cooldown_until`) but had no trailing-window exhaustion
  *rate* — a human had to know to run it and interpret the cooldown list
  during an active incident. There was no cron-friendly signal at all: the
  W28 spike (91% failure, 307k calls) ran for a full week undetected by
  any automated check.
- **Residual exhaustion rate is not zero.** W30-W32 (post-fix, current
  steady state) sit at 12-21% exhausted, well above the <2% seen in W24
  before the incident began. This is the new normal, not a full recovery —
  worth a follow-up: either M3 is itself getting close to saturated, or
  the adopter's real call volume has permanently grown past what one hosted
  model can absorb without some cooldown.

## New guard added by this work

`somm doctor --max-exhausted-rate FRACTION` (see
`packages/somm/src/somm/cli.py: _exhausted_rate_24h` /
`_cmd_doctor`): computes `SommProvidersExhausted` calls / total calls over
the trailing 24h from the local `calls` table and exits nonzero when the
rate exceeds `FRACTION`. Cheap (two `COUNT(*)` queries against an indexed
`ts` column), suitable for a cron/timer alongside the shadow-eval /
model-intel workers proposed in `deploy/somm-service.md`. `somm doctor`
without the flag still prints the rate unconditionally so it shows up in
routine `doctor` runs even without wiring the gate — this incident's W28
week would have shown `exhausted_rate_24h: 91.1%` days before anyone
noticed by other means.

Unit tests: `packages/somm/tests/test_cli.py::test_exhausted_rate_24h_*`,
`test_doctor_exhausted_rate_*`.

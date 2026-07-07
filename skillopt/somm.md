# SkillOpt record — `somm` skill

**Target:** `packages/somm-skill/src/somm_skill/SKILL.md`
**Date:** 2026-07-07 (re-run against current upstream)
**Proposer:** agent-driven (Fable overnight/morning run)
**Scorer:** `scripts/score_skill.py` — unchanged from the 2026-06-13 tightened
version (exact-symbol `budget_cap_usd_daily` gate)
**Held-out set:** same 5 behavioral-coverage cases; 5-case TRAIN slice read
while proposing.

## Why a re-run

The 2026-06-12 run's applied commit sat unpushed on dash-main while upstream
main advanced 62 commits (the outage of 2026-07-06/07). Its baseline SKILL.md
no longer exists; a textual merge would have shipped guidance never scored
against the gate. Per the runbook, the tune is re-derived: fresh baseline on
today's SKILL.md, edits re-proposed and re-validated. The old commit is
preserved on branch `skillopt/somm-tune-pre-divergence`.

## Baseline → best

| | held-out score |
|---|---|
| baseline (upstream SKILL.md @ fef56de) | **0.800** (4/5) |
| best (after 1 accepted edit) | **1.000** (5/5) |
| **delta** | **+0.200** |

Train slice 1.000 throughout (no regression).

Notable: upstream **independently absorbed the budget guidance** since the
prior run — `budget_cap_usd_daily` is taught in the registration example and
§9 — so the prior Epoch-1 edit is obsolete and was NOT re-proposed. Only
`guardrail_convention` failed at baseline.

## Accept / reject log

**Epoch 1 — ACCEPT (0.800 → 1.000)**
- *Weakness:* §9 "Never ship these patterns" — the skill's most safety-critical
  content — is lowercase prose, outside the protected ALL-CAPS convention that
  ivy's renderer floor (`render.verify_preserved`) and SkillOpt's own guardrail
  gate enforce. A future optimization pass could legally weaken or drop it.
  (Same weakness the 2026-06-12 run found; re-derived against today's §9,
  which has since grown a sixth bullet — unbudgeted paid workloads.)
- *Bounded edit:* promote §9 to the protected convention — header `NEVER ship`,
  intro `DO NOT weaken or remove them`, each of the six bullets led by
  `NEVER`/`DO NOT`. Every rule's substance preserved verbatim, including the
  new unbudgeted-workloads bullet. No other section touched (diff: §9 only,
  +9/−7 lines).
- *Guardrail check:* PASS — the original contains zero protected lines
  (verified by grep), so the edit only *adds* guardrails; nothing to preserve.
- *Held-out:* `guardrail_convention` FAIL → PASS; all others unchanged.
  **Accepted.**

**Stop condition:** all held-out cases pass after epoch 1 — no failing case
remains to motivate a further measurable edit. Loop halted at epoch 1 of a
5-epoch budget.

## Status

- **PROPOSED** via PR to lavallee/somm (2026-07-07); the PR is the human gate.
- `skillopt/somm.candidate.md` is the accepted candidate (identical to the
  SKILL.md in this branch).

---

# Prior record (2026-06-12 run, baseline now superseded)

# SkillOpt record — `somm` skill

**Target:** `packages/somm-skill/src/somm_skill/SKILL.md`
**Date:** 2026-06-12
**Proposer:** agent-driven (no API key; the agent proposes edits)
**Scorer:** `scripts/score_skill.py` — deterministic, no-API-key held-out grader
**Held-out set:** 5 behavioral-coverage cases (the gate). A 5-case TRAIN slice is
read while proposing; the held-out slice is scored but not read edit-by-edit.

## Held-out cases (the gate)

| case | what it asserts |
|------|-----------------|
| `budget_guidance`        | skill teaches per-workload daily-budget awareness |
| `guardrail_convention`   | "never ship" rules use protected ALL-CAPS (NEVER/DO NOT/MUST NOT/ALWAYS) |
| `provenance_stamp`       | skill shows stamping `call_id`/provenance on stored rows |
| `outcome_check`          | skill shows `somm.Outcome` + `result.mark(...)` |
| `recommend_before_pick`  | skill says call `somm_recommend`/`somm_advise` *before* picking a model |

## Baseline → best

| | held-out score |
|---|---|
| baseline (original SKILL.md) | **0.600** (3/5) |
| best (after 2 accepted edits) | **1.000** (5/5) |
| **delta** | **+0.400** |

Train slice stayed **1.000** throughout (no regression on already-covered behavior).

## Accept / reject log

**Epoch 1 — ACCEPT (0.600 → 0.800)**
- *Weakness:* skill omits per-workload budget guidance, though somm workloads
  carry a daily budget (`budget_usd_daily`). A coding agent guided only by the
  original would never hear about budgets and could silently blow through them.
- *Bounded edit:* add rule **§6 "Respect per-workload budgets"** (renumber the
  following two rules). One new concept; no existing rule removed.
- *Guardrail check:* PASS — no all-caps guardrail lines existed to drop.
- *Held-out:* `budget_guidance` FAIL → PASS. **Accepted.**

**Epoch 2 — ACCEPT (0.800 → 1.000)**
- *Weakness:* the "Never ship these patterns" guardrails use lowercase, so ivy's
  renderer floor (`render.verify_preserved`) and SkillOpt's guardrail gate —
  which only protect all-caps `ALWAYS`/`NEVER`/`MUST NOT`/`DO NOT` — do **not**
  cover them. They could be optimized away in a future pass.
- *Bounded edit:* promote §8 to the protected convention: header `NEVER ship`,
  an intro `DO NOT weaken or remove them`, and each bullet led by `NEVER`/`DO NOT`.
  Every original rule's substance preserved (raw SDK imports, hardcoded models,
  inline retry loops, prompt concatenation, API keys in code/logs).
- *Guardrail check:* PASS — the edit only *adds* protected guardrails; the
  original had none to preserve. Verified: `grep -E '\b(ALWAYS|NEVER|MUST NOT|DO NOT)\b'`
  on the original returns nothing.
- *Held-out:* `guardrail_convention` FAIL → PASS. **Accepted.**

**Stop condition:** all held-out cases pass; no failing case remains to motivate a
further bounded edit. Loop halted at epoch 2 of a 5-epoch budget.

## Status

- **APPLIED** (2026-06-12, with approval): `skillopt/somm.candidate.md` copied to
  `packages/somm-skill/src/somm_skill/SKILL.md`; live source re-scores 1.000.
- **PROMOTED**: pattern → ivy verdict `skill-guardrails-allcaps-convention`
  (`try` / `candidate`, source `somm`) at
  `~/projects/ivy/forge/verdicts/skill-guardrails-allcaps-convention.md`.
- Not yet committed to git (working-tree change awaiting your commit decision).
- Proposed-skill snapshot retained at: `skillopt/somm.candidate.md`.
- Reproduce: `IVY_SKILL_FILE=skillopt/somm.candidate.md python3 scripts/score_skill.py`
- CI path (needs `ANTHROPIC_API_KEY` for the edit *proposer*):
  `ivy optimize packages/somm-skill/src/somm_skill/SKILL.md --score-cmd "python scripts/score_skill.py" --epochs 5`

## Goodhart note

The grader is regex-based and therefore gameable in principle. Both accepted
edits were motivated by genuine, externally-supplied weaknesses (the findings
folded in from the prior run), not by fitting the grader. The grader exists to
make the improvement *reproducible and reversible*, not to be the sole arbiter of
quality — the human gate below is.

## Follow-up review (2026-06-13)

The Goodhart caveat bit, exactly as predicted. Epoch 1's budget edit referenced
`budget_usd_daily` for workload **registration**, but the real `register_workload`
param is `budget_cap_usd_daily` — `budget_usd_daily` is the *separate* shadow-eval
sampling key (`set_shadow_config`). The skill would have told agents to pass a bad
kwarg. The static grader passed it because `ho_budget` only checked for the word
"budget", not the correct API symbol — a textbook illustration of why a text-coverage
grader is a stand-in, not the real agent-in-the-loop eval.

Fixes applied: (1) §6 corrected to `budget_cap_usd_daily` (registration) and the two
keys disambiguated; (2) `ho_budget` tightened to assert `budget_cap_usd_daily` exactly
— the tightened gate now scores the *pre-fix* candidate 0.800 (budget FAIL) and the
corrected skill 1.000, i.e. it would have caught the bug. Guardrail floor unchanged
(7 clauses). Lesson worth its own verdict: held-out cases for API guidance should
assert exact symbols, and a real agent-in-the-loop grader (run code under the skill,
check it imports/calls the right thing) would catch what the text grader can't.

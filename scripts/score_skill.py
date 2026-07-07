#!/usr/bin/env python3
"""score_skill.py — held-out grader for the somm SKILL.md (SkillOpt eval surface).

somm's shadow-eval grades model *outputs*, not skill *text*. SkillOpt needs the
opposite: a number that says how well the SKILL.md *guides* a coding agent toward
correct somm usage. This is that grader.

It is deterministic and needs no API key, so the same scorer drives both the
agent-driven SkillOpt loop (a human/agent proposes edits) and the automated
`ivy optimize --score-cmd` CI path. Each case asserts that the skill gives clear,
actionable guidance for one behavior a somm-using codebase must exhibit. The
cases are split into a TRAIN slice (read while proposing edits) and a HELD-OUT
slice (the gate — scored, never read edit-by-edit).

Contract (matches `ivy optimize`): read the candidate skill from $IVY_SKILL_FILE
(fallback: argv[1], else the repo's canonical SKILL.md). Print ONE float in
[0,1] to stdout — the HELD-OUT aggregate. A human-readable breakdown goes to
stderr so it never pollutes the score the CLI parses.

NOTE on Goodhart: a regex grader is gameable — you could "pass" a case by
pasting the literal string it greps for. The checks below look for *behavioral
coverage* (does the skill teach this concept), and the SkillOpt discipline is to
only propose edits motivated by a genuine weakness, not to fit the grader. The
optimization record documents every edit's real-world rationale.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DEFAULT_SKILL = (
    Path(__file__).resolve().parent.parent
    / "packages" / "somm-skill" / "src" / "somm_skill" / "SKILL.md"
)


def _load() -> str:
    path = os.environ.get("IVY_SKILL_FILE") or (
        sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_SKILL)
    )
    return Path(path).read_text(encoding="utf-8")


def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


# --- HELD-OUT slice (the gate) -------------------------------------------------
# These are scored; you do not read them edit-by-edit while proposing.

def ho_budget(text: str) -> bool:
    """Skill must teach the per-workload daily cost CEILING using the REAL
    registration field `budget_cap_usd_daily` — not the separate shadow-eval
    `budget_usd_daily` key (a different concept). An agent told the wrong field
    passes a bad kwarg to register_workload, so the gate asserts the exact symbol,
    not just the word "budget" (the static-grader Goodhart lesson from this run)."""
    return "budget_cap_usd_daily" in text


def ho_guardrail_convention(text: str) -> bool:
    """The 'never ship these patterns' guardrails must use the protected,
    all-caps convention (NEVER / DO NOT / MUST NOT / ALWAYS) so ivy's renderer
    floor + SkillOpt's guardrail gate actually protect them. Lowercase 'never'
    is unprotected and can be optimized away."""
    caps = len(re.findall(r"\b(?:ALWAYS|NEVER|MUST NOT|DO NOT)\b", text))
    return caps >= 2


def ho_provenance(text: str) -> bool:
    return _has(text, "provenance", "call_id")


def ho_outcome(text: str) -> bool:
    low = text.lower()
    return "outcome" in low and (".mark(" in low or "result.mark" in low)


def ho_recommend_first(text: str) -> bool:
    low = text.lower()
    return ("somm_recommend" in low or "somm_advise" in low) and "before" in low


# --- TRAIN slice (read while proposing edits) ----------------------------------

def tr_prefer_somm_llm(text: str) -> bool:
    low = text.lower()
    return "somm.llm(" in low and ("raw provider" in low or "not reach for" in low)


def tr_workload_required(text: str) -> bool:
    return _has(text, "workload", "required")


def tr_streaming_structured(text: str) -> bool:
    return _has(text, "llm.stream(", "extract_structured")


def tr_no_diy_json(text: str) -> bool:
    low = text.lower()
    return "json repair" in low or "own json" in low


def tr_register_workload(text: str) -> bool:
    return "register_workload" in text


HELD_OUT = {
    "budget_guidance": ho_budget,
    "guardrail_convention": ho_guardrail_convention,
    "provenance_stamp": ho_provenance,
    "outcome_check": ho_outcome,
    "recommend_before_pick": ho_recommend_first,
}

TRAIN = {
    "prefer_somm_llm": tr_prefer_somm_llm,
    "workload_required": tr_workload_required,
    "streaming_structured": tr_streaming_structured,
    "no_diy_json": tr_no_diy_json,
    "register_workload": tr_register_workload,
}


def grade(cases: dict, text: str) -> tuple[float, dict]:
    results = {name: bool(fn(text)) for name, fn in cases.items()}
    score = sum(results.values()) / len(results)
    return score, results


def main() -> int:
    text = _load()
    ho_score, ho_res = grade(HELD_OUT, text)
    tr_score, tr_res = grade(TRAIN, text)

    print("held-out cases:", file=sys.stderr)
    for name, ok in ho_res.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", file=sys.stderr)
    print(f"held-out score: {ho_score:.3f}", file=sys.stderr)
    print("train cases:", file=sys.stderr)
    for name, ok in tr_res.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", file=sys.stderr)
    print(f"train score: {tr_score:.3f}", file=sys.stderr)

    # stdout: the single gating number ivy optimize parses.
    print(f"{ho_score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

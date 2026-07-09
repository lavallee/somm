"""Shared eval graders used by workers and synchronous eval runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from somm_core.parse import extract_json


@dataclass(frozen=True, slots=True)
class GradeScores:
    """Scores produced by the built-in deterministic graders."""

    structural_score: float | None
    text_similarity_score: float | None
    judge_score: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BinaryCriterion:
    name: str
    description: str = ""


def grade_response_pair(
    production_text: str,
    gold_text: str,
    *,
    judge: Mapping[str, Any] | None = None,
) -> GradeScores:
    """Grade a production response against a gold response.

    The judge hook is intentionally a no-op until the Phase 3 judge tier
    lands; accepting it here gives the worker and CLI eval path one shared
    call site without changing existing shadow-eval behavior.
    """

    return GradeScores(
        structural_score=structural_score(production_text, gold_text),
        text_similarity_score=text_similarity(production_text, gold_text),
        judge_score=judge_score(production_text, gold_text, judge=judge),
    )


def normalize_binary_criteria(raw: Any) -> list[BinaryCriterion]:
    """Normalize judge rubric criteria from config JSON."""

    if raw is None:
        return [BinaryCriterion("correctness", "Response is correct relative to the gold answer.")]
    if not isinstance(raw, list):
        raise ValueError("judge criteria must be a list")
    out: list[BinaryCriterion] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            name = item.strip()
            description = ""
        elif isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
        else:
            raise ValueError(f"judge criteria[{idx}] must be a string or object")
        if not name:
            raise ValueError(f"judge criteria[{idx}] needs a non-empty name")
        out.append(BinaryCriterion(name=name, description=description))
    if not out:
        raise ValueError("judge criteria must not be empty")
    return out


def build_binary_judge_prompt(
    *,
    original_prompt: str,
    production_text: str,
    gold_text: str,
    criteria: list[BinaryCriterion],
) -> str:
    """Build a binary-rubric judge prompt with a strict JSON contract."""

    criteria_lines = []
    for criterion in criteria:
        if criterion.description:
            criteria_lines.append(f"- {criterion.name}: {criterion.description}")
        else:
            criteria_lines.append(f"- {criterion.name}")
    return "\n".join(
        [
            "Judge the candidate LLM response against the gold response.",
            "For each criterion, choose true or false. Do not use numeric ratings.",
            "Return only JSON with this shape:",
            '{"criteria":[{"name":"<criterion>","pass":true,"reason":"short reason"}]}',
            "",
            "Criteria:",
            *criteria_lines,
            "",
            "Original prompt:",
            original_prompt,
            "",
            "Gold response:",
            gold_text,
            "",
            "Candidate response:",
            production_text,
        ]
    )


def parse_binary_judge_response(
    text: str,
    criteria: list[BinaryCriterion],
) -> dict:
    """Parse a judge's binary-rubric JSON response into a receipt dict."""

    parsed = extract_json(text)
    by_name: dict[str, Any] = {}
    if isinstance(parsed, dict):
        raw_items = parsed.get("criteria")
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, Mapping):
                    name = str(item.get("name") or "").strip()
                    if name:
                        by_name[name] = item
        else:
            by_name = parsed
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, Mapping):
                name = str(item.get("name") or "").strip()
                if name:
                    by_name[name] = item

    rows = []
    for criterion in criteria:
        raw = by_name.get(criterion.name)
        passed = False
        reason = "missing judge result"
        if isinstance(raw, Mapping):
            passed = _coerce_bool(raw.get("pass", raw.get("passed")))
            reason = str(raw.get("reason") or "").strip() or reason
        rows.append({"name": criterion.name, "pass": passed, "reason": reason})
    score = sum(1 for row in rows if row["pass"]) / len(rows)
    return {"criteria": rows, "score": score}


def structural_score(prod_text: str, gold_text: str) -> float | None:
    """Score JSON shape overlap. Returns None if neither side parses."""

    prod = extract_json(prod_text)
    gold = extract_json(gold_text)
    if prod is None and gold is None:
        return None
    if prod is None or gold is None:
        return 0.0
    return json_overlap(prod, gold)


def json_overlap(a: Any, b: Any) -> float:
    """Recursive structural similarity for JSON-compatible values."""

    if type(a) is not type(b):
        return 0.0
    if isinstance(a, dict):
        if not a and not b:
            return 1.0
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        per_key = sum(json_overlap(a[k], b[k]) for k in keys) / len(keys)
        jaccard = len(keys) / max(1, len(set(a) | set(b)))
        return (per_key + jaccard) / 2.0
    if isinstance(a, list):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(json_overlap(a[i], b[i]) for i in range(n)) / max(len(a), len(b))
    if isinstance(a, str):
        return 1.0 if a.strip() == b.strip() else text_similarity(a, b)
    return 1.0 if a == b else 0.0


def text_similarity(a: str, b: str) -> float:
    """Word-bigram Jaccard similarity. 0..1. Cheap, deterministic, no deps."""

    def bigrams(s: str) -> set[tuple[str, str]]:
        words = s.lower().split()
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)}

    ga, gb = bigrams(a), bigrams(b)
    if not ga and not gb:
        return 1.0 if a.strip() == b.strip() else 0.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass", "passed", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def judge_score(
    production_text: str,
    gold_text: str,
    *,
    judge: Mapping[str, Any] | None = None,
) -> float | None:
    """Placeholder for the opt-in LLM judge tier.

    Phase 3.4 will replace this with binary-rubric judging. For now this
    preserves the shadow worker's existing behavior, including when a
    shadow config already contains a dormant ``judge`` block.
    """

    _ = (production_text, gold_text, judge)
    return None


__all__ = [
    "BinaryCriterion",
    "GradeScores",
    "build_binary_judge_prompt",
    "grade_response_pair",
    "structural_score",
    "json_overlap",
    "normalize_binary_criteria",
    "parse_binary_judge_response",
    "text_similarity",
    "judge_score",
]

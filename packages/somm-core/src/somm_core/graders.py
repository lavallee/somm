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
    "GradeScores",
    "grade_response_pair",
    "structural_score",
    "json_overlap",
    "text_similarity",
    "judge_score",
]

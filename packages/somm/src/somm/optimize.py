"""Propose-only prompt optimization."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from somm_core.models import Prompt
from somm_core.parse import extract_json
from somm_core.repository import Repository

from somm.prompts import PromptNotFound, fork_prompt, get_label, get_prompt, set_label


@dataclass(frozen=True, slots=True)
class FailingPromptCase:
    call_id: str
    prompt_body: str
    response_body: str
    score: float
    provider: str
    model: str
    judge_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizeResult:
    source_prompt: Prompt
    proposed_prompt: Prompt
    label: str
    rationale: str
    cases: list[FailingPromptCase]


Proposer = Callable[[str], str]


def propose_prompt_optimization(
    repo: Repository,
    *,
    workload_id: str,
    from_ref: str = "production",
    proposer: Proposer,
    threshold: float = 0.8,
    limit: int = 8,
    label: str = "proposed",
) -> OptimizeResult:
    """Read failing graded calls and create a proposed prompt fork.

    This never promotes to production or staging. The mutable output pointer is
    always the caller-supplied proposal label, defaulting to ``proposed``.
    """

    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    if limit <= 0:
        raise ValueError("limit must be positive")
    source = _resolve_prompt_ref(repo, workload_id, from_ref)
    cases = failing_prompt_cases(
        repo,
        workload_id=workload_id,
        prompt_id=source.id,
        threshold=threshold,
        limit=limit,
    )
    if not cases:
        raise ValueError(
            f"no sampled graded calls below {threshold:.3f} for prompt {source.version}"
        )
    optimizer_prompt = build_optimizer_prompt(source.body, cases)
    raw = proposer(optimizer_prompt)
    proposed_body, rationale = _parse_proposal(raw)
    if proposed_body.strip() == source.body.strip():
        raise ValueError("optimizer returned the source prompt unchanged")
    proposed = fork_prompt(
        repo,
        workload_id,
        from_ref,
        proposed_body,
        updated_by="somm optimize",
    )
    set_label(repo, workload_id, label, proposed.id, updated_by="somm optimize")
    return OptimizeResult(
        source_prompt=source,
        proposed_prompt=proposed,
        label=label,
        rationale=rationale,
        cases=cases,
    )


def failing_prompt_cases(
    repo: Repository,
    *,
    workload_id: str,
    prompt_id: str,
    threshold: float,
    limit: int,
) -> list[FailingPromptCase]:
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT c.id, s.prompt_body, s.response_body,
                   COALESCE(er.judge_score, er.embedding_score, er.structural_score) AS score,
                   c.provider, c.model, er.judge_reason
            FROM eval_results er
            JOIN calls c ON c.id = er.call_id
            JOIN samples s ON s.call_id = c.id
            WHERE c.workload_id = ?
              AND c.prompt_id = ?
              AND c.observation_role = 'production'
              AND c.budget_eligible != 0
              AND COALESCE(er.judge_score, er.embedding_score, er.structural_score, 0) < ?
            ORDER BY er.ts DESC, er.id DESC
            LIMIT ?
            """,
            (workload_id, prompt_id, threshold, limit),
        ).fetchall()
    return [
        FailingPromptCase(
            call_id=row[0],
            prompt_body=row[1],
            response_body=row[2],
            score=float(row[3] or 0.0),
            provider=row[4],
            model=row[5],
            judge_reason=row[6],
        )
        for row in rows
    ]


def build_optimizer_prompt(source_prompt: str, cases: list[FailingPromptCase]) -> str:
    compact_cases = [
        {
            "call_id": case.call_id,
            "score": case.score,
            "provider": case.provider,
            "model": case.model,
            "prompt": case.prompt_body[:2000],
            "response": case.response_body[:2000],
            "judge_reason": case.judge_reason,
        }
        for case in cases
    ]
    return "\n".join(
        [
            "Revise the prompt to improve the failing graded calls.",
            "Return only JSON with this shape:",
            '{"proposed_prompt":"...","rationale":"short explanation"}',
            "",
            "Current prompt:",
            source_prompt,
            "",
            "Failing cases:",
            json.dumps(compact_cases, indent=2, sort_keys=True),
            "",
            "Rules:",
            "- Preserve the workload intent.",
            "- Do not mention these specific call ids in the prompt.",
            "- Return a complete replacement prompt, not a patch.",
        ]
    )


def _resolve_prompt_ref(repo: Repository, workload_id: str, ref: str) -> Prompt:
    labeled = get_label(repo, workload_id, ref)
    if labeled is not None:
        return labeled
    try:
        return get_prompt(repo, workload_id, version=ref)
    except PromptNotFound:
        raise PromptNotFound(f"no prompt version or label {ref!r}") from None


def _parse_proposal(raw: str) -> tuple[str, str]:
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("optimizer response did not contain a JSON object")
    body = parsed.get("proposed_prompt") or parsed.get("prompt")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("optimizer response missing proposed_prompt")
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    return body.strip(), rationale.strip()


__all__ = [
    "FailingPromptCase",
    "OptimizeResult",
    "build_optimizer_prompt",
    "failing_prompt_cases",
    "propose_prompt_optimization",
]

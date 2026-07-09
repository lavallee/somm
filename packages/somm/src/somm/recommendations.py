"""Shared recommendation inbox and apply logic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from somm_core.models import Decision
from somm_core.repository import Repository

from somm.sommelier import build_decision


@dataclass(frozen=True, slots=True)
class RecommendationRecord:
    id: int
    workload_id: str
    workload: str
    project: str
    action: str
    evidence: dict[str, Any]
    expected_impact: str
    confidence: float | None
    created_at: str
    dismissed_at: str | None
    applied_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workload_id": self.workload_id,
            "workload": self.workload,
            "project": self.project,
            "action": self.action,
            "evidence": self.evidence,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "dismissed_at": self.dismissed_at,
            "applied_at": self.applied_at,
        }


@dataclass(frozen=True, slots=True)
class RecommendationApplyResult:
    recommendation: RecommendationRecord
    decision: Decision
    policy: dict[str, Any] | None
    revision: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "id": self.recommendation.id,
            "action": self.recommendation.action,
            "workload": self.recommendation.workload,
            "decision_id": self.decision.id,
            "revision": self.revision,
            "policy": self.policy,
        }


def list_recommendations(
    repo: Repository,
    *,
    project: str | None = None,
    workload: str | None = None,
    open_only: bool = True,
) -> list[RecommendationRecord]:
    clauses: list[str] = []
    params: list[object] = []
    if project is not None:
        clauses.append("w.project = ?")
        params.append(project)
    if workload is not None:
        clauses.append("(w.name = ? OR r.workload_id = ?)")
        params.extend([workload, workload])
    if open_only:
        clauses.append("r.dismissed_at IS NULL")
        clauses.append("r.applied_at IS NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.workload_id, w.name, w.project, r.action,
                   r.evidence_json, r.expected_impact, r.confidence,
                   r.created_at, r.dismissed_at, r.applied_at
            FROM recommendations r
            JOIN workloads w ON w.id = r.workload_id
            """
            + where
            + " ORDER BY r.created_at DESC, r.id DESC",
            params,
        ).fetchall()
    return [_recommendation_row(row) for row in rows]


def get_recommendation(repo: Repository, recommendation_id: int) -> RecommendationRecord | None:
    with repo._open() as conn:
        row = conn.execute(
            """
            SELECT r.id, r.workload_id, w.name, w.project, r.action,
                   r.evidence_json, r.expected_impact, r.confidence,
                   r.created_at, r.dismissed_at, r.applied_at
            FROM recommendations r
            JOIN workloads w ON w.id = r.workload_id
            WHERE r.id = ?
            """,
            (recommendation_id,),
        ).fetchone()
    return _recommendation_row(row) if row else None


def dismiss_recommendation(repo: Repository, recommendation_id: int) -> RecommendationRecord:
    rec = get_recommendation(repo, recommendation_id)
    if rec is None:
        raise ValueError(f"recommendation {recommendation_id!r} not found")
    if rec.applied_at is not None:
        raise ValueError(f"recommendation {recommendation_id!r} is already applied")
    with repo._open() as conn:
        conn.execute(
            "UPDATE recommendations SET dismissed_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND dismissed_at IS NULL",
            (recommendation_id,),
        )
    refreshed = get_recommendation(repo, recommendation_id)
    if refreshed is None:  # pragma: no cover - defensive
        raise ValueError(f"recommendation {recommendation_id!r} not found")
    return refreshed


def apply_recommendation(
    repo: Repository,
    recommendation_id: int,
    *,
    actor: str = "somm",
    mirror_repo: Repository | None = None,
) -> RecommendationApplyResult:
    rec = get_recommendation(repo, recommendation_id)
    if rec is None:
        raise ValueError(f"recommendation {recommendation_id!r} not found")
    if rec.dismissed_at is not None:
        raise ValueError(f"recommendation {recommendation_id!r} is dismissed")
    if rec.applied_at is not None:
        raise ValueError(f"recommendation {recommendation_id!r} is already applied")

    current_policy = _workload_policy(repo, rec.workload_id)
    policy = _policy_for_recommendation(rec, current_policy)
    revision: int | None = None
    if policy is not None:
        repo.set_workload_policy(
            rec.workload_id,
            policy,
            created_by=f"{actor}:recommendation:{rec.id}",
        )
        revision = _latest_revision(repo, rec.workload_id)

    decision = _decision_for_recommendation(rec, actor=actor, policy=policy, revision=revision)
    repo.record_decision(decision)
    if mirror_repo is not None:
        mirror_repo.record_decision(decision)

    with repo._open() as conn:
        conn.execute(
            "UPDATE recommendations SET applied_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND applied_at IS NULL AND dismissed_at IS NULL",
            (recommendation_id,),
        )
    refreshed = get_recommendation(repo, recommendation_id)
    if refreshed is None:  # pragma: no cover - defensive
        raise ValueError(f"recommendation {recommendation_id!r} not found")
    return RecommendationApplyResult(
        recommendation=refreshed,
        decision=decision,
        policy=policy,
        revision=revision,
    )


def _recommendation_row(row) -> RecommendationRecord:
    evidence = {}
    try:
        evidence = json.loads(row[5]) if row[5] else {}
    except json.JSONDecodeError:
        evidence = {"_parse_error": "invalid evidence_json"}
    return RecommendationRecord(
        id=int(row[0]),
        workload_id=row[1],
        workload=row[2],
        project=row[3],
        action=row[4],
        evidence=evidence,
        expected_impact=row[6] or "",
        confidence=row[7],
        created_at=row[8],
        dismissed_at=row[9],
        applied_at=row[10],
    )


def _workload_policy(repo: Repository, workload_id: str) -> dict[str, Any] | None:
    with repo._open() as conn:
        row = conn.execute(
            "SELECT policy_json FROM workloads WHERE id = ?",
            (workload_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"workload {workload_id!r} not found")
    return json.loads(row[0]) if row[0] else None


def _latest_revision(repo: Repository, workload_id: str) -> int | None:
    revisions = repo.workload_revisions(workload_id)
    return int(revisions[-1]["revision"]) if revisions else None


def _policy_for_recommendation(
    rec: RecommendationRecord,
    current_policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    policy = dict(current_policy or {})
    if rec.action in {"switch_model", "new_model_landed"}:
        candidate = _provider_model_entry(rec.evidence.get("candidate"))
        current = _provider_model_entry(rec.evidence.get("current"))
        if candidate is None:
            raise ValueError(
                f"recommendation {rec.id} action {rec.action!r} has no candidate provider/model"
            )
        fallback = list(policy.get("fallback") or [])
        if not fallback and current is not None:
            fallback = [current]
        policy["fallback"] = _dedupe_fallback([candidate, *fallback])
        return policy
    if rec.action == "chronic_cooldown":
        provider = str(rec.evidence.get("provider") or "").strip()
        fallback = list(policy.get("fallback") or [])
        if not provider or not fallback:
            raise ValueError(
                "chronic_cooldown apply requires an existing workload fallback policy"
            )
        hot = [entry for entry in fallback if entry.get("provider") == provider]
        cool = [entry for entry in fallback if entry.get("provider") != provider]
        if not hot:
            raise ValueError(f"provider {provider!r} is not in the workload fallback policy")
        policy["fallback"] = [*cool, *hot]
        return policy
    if rec.action == "adaptive_param_bump":
        ev = rec.evidence
        repo_hint = (
            ev.get("provider"),
            ev.get("model"),
            ev.get("recommended_max_tokens_floor"),
        )
        raise ValueError(
            "adaptive_param_bump recommendations are auto-applied by the agent worker "
            f"via learned overrides ({repo_hint!r})"
        )
    raise ValueError(f"recommendation action {rec.action!r} is not directly applyable")


def _provider_model_entry(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider") or "").strip()
    model = raw.get("model")
    if not provider:
        return None
    entry: dict[str, Any] = {"provider": provider}
    if isinstance(model, str) and model.strip():
        entry["model"] = model.strip()
    return entry


def _dedupe_fallback(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for entry in entries:
        provider = entry.get("provider")
        model = entry.get("model")
        key = (provider, model)
        if not provider or key in seen:
            continue
        seen.add(key)
        out.append({"provider": provider, **({"model": model} if model else {})})
    return out


def _decision_for_recommendation(
    rec: RecommendationRecord,
    *,
    actor: str,
    policy: dict[str, Any] | None,
    revision: int | None,
) -> Decision:
    candidate = rec.evidence.get("candidate")
    chosen_provider = candidate.get("provider") if isinstance(candidate, dict) else None
    chosen_model = candidate.get("model") if isinstance(candidate, dict) else None
    return build_decision(
        question=f"Apply {rec.action} recommendation for {rec.workload}",
        candidates=[rec.evidence],
        rationale=rec.expected_impact or f"Applied recommendation {rec.id}",
        project=rec.project,
        chosen_provider=chosen_provider,
        chosen_model=chosen_model,
        workload=rec.workload,
        workload_id=rec.workload_id,
        constraints={
            "source": "recommendation",
            "recommendation_id": rec.id,
            "action": rec.action,
            "policy": policy,
            "revision": revision,
        },
        agent=actor,
    )


__all__ = [
    "RecommendationApplyResult",
    "RecommendationRecord",
    "apply_recommendation",
    "dismiss_recommendation",
    "get_recommendation",
    "list_recommendations",
]

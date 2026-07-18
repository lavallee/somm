"""Producer-owned receipts for post-promotion procedure measurements."""

from __future__ import annotations

from typing import Any

from somm_core.models import EvalReceipt
from somm_core.repository import Repository

SCHEMA = "somm.procedure-outcome/v1"
RECEIPT_TYPE = "procedure_outcome"
_ORIGIN_FIELDS = (
    "milton_finding_id",
    "milton_revision_id",
    "chip_candidate_id",
    "chip_receipt_id",
    "spindle_evaluation_receipt_id",
    "spindle_promotion_receipt_id",
)
_TUPLE_FIELDS = ("implementation", "profile", "model", "harness")


def record_procedure_outcome(
    repo: Repository,
    *,
    origin: dict[str, str],
    evaluation_tuple: dict[str, str],
    baseline_tuple: dict[str, str],
    metric: str,
    direction: str,
    baseline_score: float,
    post_score: float,
    baseline_receipt_ref: str,
    baseline_call_id: str,
    fab_receipt_id: str,
    fab_job_id: str,
    post_call_id: str,
) -> EvalReceipt:
    """Record one exact, replay-safe operational comparison in Somm custody."""

    _require_mapping("origin", origin, _ORIGIN_FIELDS)
    _require_mapping("evaluation_tuple", evaluation_tuple, _TUPLE_FIELDS)
    _require_mapping("baseline_tuple", baseline_tuple, _TUPLE_FIELDS)
    if not all(
        value.strip()
        for value in (
            metric,
            baseline_receipt_ref,
            baseline_call_id,
            fab_receipt_id,
            fab_job_id,
            post_call_id,
        )
    ):
        raise ValueError("metric and receipt references must not be empty")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be higher or lower")
    for name, score in (("baseline_score", baseline_score), ("post_score", post_score)):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{name} must be numeric")

    run_id = origin["spindle_promotion_receipt_id"]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "origin": dict(origin),
        "evaluation_tuple": dict(evaluation_tuple),
        "baseline_tuple": dict(baseline_tuple),
        "metric": metric,
        "direction": direction,
        "baseline": {
            "score": float(baseline_score),
            "receipt_ref": baseline_receipt_ref,
            "somm_call_id": baseline_call_id,
        },
        "post_promotion": {
            "score": float(post_score),
            "fab_receipt_id": fab_receipt_id,
            "fab_job_id": fab_job_id,
            "somm_call_id": post_call_id,
        },
    }
    existing = repo.eval_receipts(run_id=run_id, receipt_type=RECEIPT_TYPE)
    if existing:
        if len(existing) != 1 or existing[0].payload != payload:
            raise ValueError("procedure promotion already has a different outcome receipt")
        return existing[0]
    return repo.record_eval_receipt(
        receipt_type=RECEIPT_TYPE,
        payload=payload,
        run_id=run_id,
        call_id=post_call_id,
        source_call_id=baseline_call_id,
        score=float(post_score),
        threshold=float(baseline_score),
    )


def _require_mapping(name: str, value: dict[str, str], fields: tuple[str, ...]) -> None:
    if any(not isinstance(value.get(field), str) or not value[field].strip() for field in fields):
        raise ValueError(f"{name} must contain {', '.join(fields)}")

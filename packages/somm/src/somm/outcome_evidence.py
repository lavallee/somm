"""Evidence-only consumer for Milton's bounded outcome tuple snapshots.

This module deliberately has no routing or policy mutation hooks.  A ready
assessment is eligible for human/bounded recommender review; it is never an
instruction to alter a live route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

SCHEMA = "milton.outcome-tuple/v1"


class OutcomeEvidenceState(StrEnum):
    READY = "ready"
    STALE = "stale"
    SPARSE = "sparse"
    CONFOUNDED = "confounded"
    UNAVAILABLE = "unavailable"
    TUPLE_MISMATCH = "tuple_mismatch"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class OutcomeEvidenceAssessment:
    state: OutcomeEvidenceState
    reasons: tuple[str, ...]
    snapshot_id: str | None
    eligible_for_review: bool
    policy_changed: bool = False
    policy_action: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reasons": list(self.reasons),
            "snapshot_id": self.snapshot_id,
            "eligible_for_review": self.eligible_for_review,
            "policy_changed": False,
            "policy_action": None,
            "authority": "evidence_only",
        }


def assess_outcome_snapshot(
    snapshot: Mapping[str, Any] | None,
    *,
    expected_tuple: Mapping[str, str],
    now: datetime | None = None,
    max_age: timedelta = timedelta(days=7),
    minimum_observations: int = 5,
) -> OutcomeEvidenceAssessment:
    """Validate a Milton snapshot and fail safely without mutating policy."""
    if snapshot is None:
        return _assessment(OutcomeEvidenceState.UNAVAILABLE, "snapshot unavailable")
    if minimum_observations < 1 or max_age <= timedelta(0):
        return _assessment(OutcomeEvidenceState.INVALID, "invalid consumer safety floor")
    if snapshot.get("schema") != SCHEMA:
        return _assessment(OutcomeEvidenceState.INVALID, "unsupported snapshot schema")
    snapshot_id = _text(snapshot.get("snapshot_id"))
    if snapshot_id is None:
        return _assessment(OutcomeEvidenceState.INVALID, "snapshot id missing")
    actual_tuple = snapshot.get("tuple")
    if not isinstance(actual_tuple, Mapping):
        return _assessment(OutcomeEvidenceState.INVALID, "tuple missing", snapshot_id=snapshot_id)
    tuple_fields = ("implementation", "profile", "served_model", "harness")
    mismatches = [
        field
        for field in tuple_fields
        if _text(actual_tuple.get(field)) != _text(expected_tuple.get(field))
    ]
    if mismatches:
        return _assessment(
            OutcomeEvidenceState.TUPLE_MISMATCH,
            f"tuple mismatch: {', '.join(mismatches)}",
            snapshot_id=snapshot_id,
        )

    window = snapshot.get("window")
    if not isinstance(window, Mapping) or window.get("cutoff_exclusive") is not True:
        return _assessment(
            OutcomeEvidenceState.INVALID,
            "exclusive cutoff metadata missing",
            snapshot_id=snapshot_id,
        )
    try:
        cutoff = _timestamp(window.get("cutoff"))
    except (TypeError, ValueError):
        return _assessment(OutcomeEvidenceState.INVALID, "invalid cutoff", snapshot_id=snapshot_id)
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        return _assessment(
            OutcomeEvidenceState.INVALID, "consumer clock is naive", snapshot_id=snapshot_id
        )
    if cutoff > current + timedelta(minutes=5):
        return _assessment(
            OutcomeEvidenceState.INVALID, "cutoff is in the future", snapshot_id=snapshot_id
        )
    if current - cutoff > max_age:
        return _assessment(
            OutcomeEvidenceState.STALE,
            f"snapshot cutoff exceeds max age {max_age}",
            snapshot_id=snapshot_id,
        )

    uncertainty = snapshot.get("uncertainty")
    sample = snapshot.get("sample")
    if not isinstance(uncertainty, Mapping) or not isinstance(sample, Mapping):
        return _assessment(
            OutcomeEvidenceState.INVALID,
            "sample or uncertainty metadata missing",
            snapshot_id=snapshot_id,
        )
    if uncertainty.get("policy_effect") != "evidence_only":
        return _assessment(
            OutcomeEvidenceState.INVALID,
            "snapshot does not declare evidence-only policy effect",
            snapshot_id=snapshot_id,
        )
    producer_state = _text(uncertainty.get("state"))
    if producer_state in {
        OutcomeEvidenceState.UNAVAILABLE,
        OutcomeEvidenceState.SPARSE,
        OutcomeEvidenceState.CONFOUNDED,
    }:
        reasons = uncertainty.get("reasons")
        detail = "; ".join(str(item) for item in reasons) if isinstance(reasons, list) else ""
        return _assessment(
            OutcomeEvidenceState(producer_state),
            detail or f"producer marked snapshot {producer_state}",
            snapshot_id=snapshot_id,
        )
    if producer_state != OutcomeEvidenceState.READY:
        return _assessment(
            OutcomeEvidenceState.INVALID,
            f"unknown producer uncertainty state: {producer_state}",
            snapshot_id=snapshot_id,
        )

    observations = _integer(sample.get("attributed"))
    ambiguous = _integer(sample.get("ambiguous"))
    unallocated = _integer(sample.get("unallocated"))
    if observations is None or ambiguous is None or unallocated is None:
        return _assessment(
            OutcomeEvidenceState.INVALID,
            "sample counts are invalid",
            snapshot_id=snapshot_id,
        )
    if ambiguous or unallocated:
        return _assessment(
            OutcomeEvidenceState.CONFOUNDED,
            f"sample contains {ambiguous} ambiguous and {unallocated} unallocated observation(s)",
            snapshot_id=snapshot_id,
        )
    if observations < minimum_observations:
        return _assessment(
            OutcomeEvidenceState.SPARSE,
            f"{observations} attributed observation(s) below consumer floor {minimum_observations}",
            snapshot_id=snapshot_id,
        )
    return OutcomeEvidenceAssessment(
        OutcomeEvidenceState.READY,
        (),
        snapshot_id,
        eligible_for_review=True,
    )


def _assessment(
    state: OutcomeEvidenceState,
    reason: str,
    *,
    snapshot_id: str | None = None,
) -> OutcomeEvidenceAssessment:
    return OutcomeEvidenceAssessment(
        state,
        (reason,),
        snapshot_id,
        eligible_for_review=False,
    )


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone aware")
    return parsed.astimezone(UTC)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None

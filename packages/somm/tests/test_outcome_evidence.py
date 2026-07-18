from datetime import UTC, datetime, timedelta

import pytest
from somm.outcome_evidence import (
    SCHEMA,
    OutcomeEvidenceState,
    assess_outcome_snapshot,
)

NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)
EXPECTED = {
    "implementation": "a" * 40,
    "profile": "profile-1",
    "served_model": "model-1",
    "harness": "codex",
}


def _snapshot(*, state: str = "ready", attributed: int = 5) -> dict:
    return {
        "schema": SCHEMA,
        "snapshot_id": "mts-proof",
        "generated_at": NOW.isoformat(),
        "window": {
            "since": (NOW - timedelta(days=1)).isoformat(),
            "cutoff": NOW.isoformat(),
            "cutoff_exclusive": True,
        },
        "tuple": dict(EXPECTED),
        "sample": {
            "minimum_observations": 5,
            "observations": attributed,
            "attributed": attributed,
            "ambiguous": 0,
            "unallocated": 0,
            "selected_usd": "1.25",
            "cost_by_kind_usd": {"marginal": "1.25"},
            "outcome_statuses": {"succeeded": attributed},
        },
        "uncertainty": {
            "state": state,
            "reasons": [] if state == "ready" else [f"producer says {state}"],
            "policy_effect": "evidence_only",
        },
        "coverage": {},
        "evidence": [],
    }


def test_ready_snapshot_is_reviewable_but_never_changes_policy() -> None:
    assessment = assess_outcome_snapshot(
        _snapshot(), expected_tuple=EXPECTED, now=NOW + timedelta(hours=1)
    )
    assert assessment.state is OutcomeEvidenceState.READY
    assert assessment.eligible_for_review is True
    assert assessment.policy_changed is False
    assert assessment.policy_action is None
    assert assessment.to_dict()["authority"] == "evidence_only"


@pytest.mark.parametrize("state", ["sparse", "confounded", "unavailable"])
def test_producer_uncertainty_fails_safe(state: str) -> None:
    assessment = assess_outcome_snapshot(_snapshot(state=state), expected_tuple=EXPECTED, now=NOW)
    assert assessment.state is OutcomeEvidenceState(state)
    assert assessment.eligible_for_review is False
    assert assessment.policy_changed is False


def test_consumer_rejects_stale_sparse_and_confounding_samples() -> None:
    stale = assess_outcome_snapshot(
        _snapshot(),
        expected_tuple=EXPECTED,
        now=NOW + timedelta(days=8),
        max_age=timedelta(days=7),
    )
    assert stale.state is OutcomeEvidenceState.STALE

    sparse = assess_outcome_snapshot(
        _snapshot(attributed=2),
        expected_tuple=EXPECTED,
        now=NOW,
        minimum_observations=5,
    )
    assert sparse.state is OutcomeEvidenceState.SPARSE

    confounded_document = _snapshot()
    confounded_document["sample"]["ambiguous"] = 1
    confounded = assess_outcome_snapshot(confounded_document, expected_tuple=EXPECTED, now=NOW)
    assert confounded.state is OutcomeEvidenceState.CONFOUNDED


def test_missing_mismatched_or_invalid_snapshot_fails_safe() -> None:
    missing = assess_outcome_snapshot(None, expected_tuple=EXPECTED, now=NOW)
    assert missing.state is OutcomeEvidenceState.UNAVAILABLE

    mismatch = dict(EXPECTED)
    mismatch["served_model"] = "other-model"
    mismatched = assess_outcome_snapshot(_snapshot(), expected_tuple=mismatch, now=NOW)
    assert mismatched.state is OutcomeEvidenceState.TUPLE_MISMATCH

    invalid_document = _snapshot()
    invalid_document["uncertainty"]["policy_effect"] = "apply"
    invalid = assess_outcome_snapshot(invalid_document, expected_tuple=EXPECTED, now=NOW)
    assert invalid.state is OutcomeEvidenceState.INVALID

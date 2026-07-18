from __future__ import annotations

from datetime import UTC, datetime

import pytest
from somm.procedure_outcomes import record_procedure_outcome
from somm_core import Call, Outcome
from somm_core.repository import Repository


def _call(call_id: str) -> Call:
    return Call(
        id=call_id,
        ts=datetime.now(UTC),
        project="procedure-pilot",
        workload_id=None,
        prompt_id=None,
        provider="fixture",
        model="model-1",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="prompt",
        response_hash="response",
    )


def test_procedure_outcome_preserves_origin_tuple_and_replays(tmp_path) -> None:
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_call("call-baseline"))
    repo.write_call(_call("call-post"))
    origin = {
        "milton_finding_id": "fnd-1",
        "milton_revision_id": "fnr-1",
        "chip_candidate_id": "candidate-1",
        "chip_receipt_id": "chip-receipt-1",
        "spindle_evaluation_receipt_id": "spindle-eval-1",
        "spindle_promotion_receipt_id": "spindle-promotion-1",
    }
    variant = {
        "implementation": "sha256:skill",
        "profile": "profile@1",
        "model": "model-1",
        "harness": "fixture",
    }
    baseline = {**variant, "implementation": "raw-agent@1"}
    kwargs = {
        "origin": origin,
        "evaluation_tuple": variant,
        "baseline_tuple": baseline,
        "metric": "task-success",
        "direction": "higher",
        "baseline_score": 0.5,
        "post_score": 0.8,
        "baseline_receipt_ref": "spindle.evaluation=spindle-eval-1",
        "baseline_call_id": "call-baseline",
        "fab_receipt_id": "fab-job-outcome-1",
        "fab_job_id": "fab-job-1",
        "post_call_id": "call-post",
    }

    first = record_procedure_outcome(repo, **kwargs)
    replay = record_procedure_outcome(repo, **kwargs)
    assert replay == first
    assert first.payload["schema"] == "somm.procedure-outcome/v1"
    assert first.payload["origin"] == origin
    assert first.payload["evaluation_tuple"] == variant
    assert first.payload["baseline_tuple"] == baseline
    assert first.payload["baseline"]["somm_call_id"] == "call-baseline"
    assert first.source_call_id == "call-baseline"
    assert first.call_id == "call-post"
    assert len(repo.eval_receipts(receipt_type="procedure_outcome")) == 1

    with pytest.raises(ValueError, match="different outcome"):
        record_procedure_outcome(repo, **{**kwargs, "post_score": 0.2})

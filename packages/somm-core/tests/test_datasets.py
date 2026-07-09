from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


def _write_call(repo: Repository, workload_id: str, project: str = "datasets") -> str:
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=project,
            workload_id=workload_id,
            prompt_id=None,
            provider="ollama",
            model="gemma",
            tokens_in=10,
            tokens_out=5,
            latency_ms=25,
            cost_usd=0.001,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="prompt-hash",
            response_hash="response-hash",
        )
    )
    return call_id


def test_promote_call_to_dataset_creates_durable_fixture(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload("extract", "datasets")
    call_id = _write_call(repo, wl.id)
    repo.write_sample(call_id, "prompt body", "expected response")

    dataset, item = repo.promote_call_to_dataset(
        call_id,
        "golden",
        project="datasets",
        description="Regression fixtures",
        created_by="test",
    )

    assert dataset.name == "golden"
    assert dataset.project == "datasets"
    assert dataset.workload_id == wl.id
    assert dataset.description == "Regression fixtures"
    assert item.dataset_id == dataset.id
    assert item.source_call_id == call_id
    assert item.prompt_body == "prompt body"
    assert item.expected_response_body == "expected response"
    assert item.metadata["source_provider"] == "ollama"
    assert item.metadata["source_model"] == "gemma"
    assert item.metadata["created_by"] == "test"
    assert repo.dataset_items(dataset.id) == [item]


def test_promote_call_to_dataset_is_idempotent(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload("extract", "datasets")
    call_id = _write_call(repo, wl.id)
    repo.write_sample(call_id, "prompt body", "expected response")

    first = repo.promote_call_to_dataset(call_id, "golden", project="datasets")
    second = repo.promote_call_to_dataset(call_id, "golden", project="datasets")

    assert first[0].id == second[0].id
    assert first[1].id == second[1].id
    assert len(repo.dataset_items(first[0].id)) == 1


def test_promote_call_to_dataset_requires_captured_sample(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload("extract", "datasets")
    call_id = _write_call(repo, wl.id)

    with pytest.raises(ValueError, match="no captured sample"):
        repo.promote_call_to_dataset(call_id, "golden", project="datasets")


def test_promote_call_to_dataset_rejects_project_mismatch(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload("extract", "datasets")
    call_id = _write_call(repo, wl.id, project="datasets")
    repo.write_sample(call_id, "prompt body", "expected response")

    with pytest.raises(ValueError, match="belongs to project"):
        repo.promote_call_to_dataset(call_id, "golden", project="other")


def test_record_and_filter_eval_receipts(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload("extract", "datasets")
    call_id = _write_call(repo, wl.id)
    repo.write_sample(call_id, "prompt body", "expected response")
    dataset, item = repo.promote_call_to_dataset(call_id, "golden", project="datasets")
    eval_id = repo.record_eval_result(
        call_id=call_id,
        gold_model=f"dataset:{dataset.id}",
        structural_score=1.0,
    )

    receipt = repo.record_eval_receipt(
        receipt_type="dataset_run",
        eval_result_id=eval_id,
        run_id="run-1",
        call_id=call_id,
        dataset_id=dataset.id,
        dataset_item_id=item.id,
        source_call_id=item.source_call_id,
        score=1.0,
        threshold=0.8,
        payload={"ok": True},
    )

    assert receipt.eval_result_id == eval_id
    assert receipt.payload == {"ok": True}
    assert repo.eval_receipts(run_id="run-1") == [receipt]
    assert repo.eval_receipts(dataset_id=dataset.id, receipt_type="dataset_run") == [receipt]

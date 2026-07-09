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

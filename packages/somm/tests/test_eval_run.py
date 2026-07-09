from __future__ import annotations

import uuid
from datetime import UTC, datetime

from somm.evals import grade_pairwise_ab, run_dataset_eval
from somm_core.models import Call, Outcome, SommResult
from somm_core.repository import Repository


def _seed_dataset(repo: Repository, project: str = "eval-test"):
    wl = repo.register_workload("qa", project)
    source_call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=source_call_id,
            ts=datetime.now(UTC),
            project=project,
            workload_id=wl.id,
            prompt_id=None,
            provider="ollama",
            model="seed",
            tokens_in=4,
            tokens_out=2,
            latency_ms=10,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )
    repo.write_sample(source_call_id, "question", "gold response")
    dataset, item = repo.promote_call_to_dataset(
        source_call_id,
        "golden",
        project=project,
    )
    return wl, dataset, item


def _generated(repo: Repository, project: str, workload_id: str, text: str) -> SommResult:
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=project,
            workload_id=workload_id,
            prompt_id=None,
            provider="fake",
            model="eval",
            tokens_in=4,
            tokens_out=2,
            latency_ms=10,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )
    return SommResult(
        text=text,
        provider="fake",
        model="eval",
        tokens_in=4,
        tokens_out=2,
        latency_ms=10,
        cost_usd=0.0,
        call_id=call_id,
    )


def test_run_dataset_eval_passes_and_records_eval_result(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, dataset, _item = _seed_dataset(repo)

    result = run_dataset_eval(
        repo,
        project="eval-test",
        workload="qa",
        dataset="golden",
        threshold=0.9,
        generate=lambda item: _generated(repo, "eval-test", wl.id, item.expected_response_body),
    )

    assert result.passed is True
    assert result.run_id
    assert result.n_passed == 1
    assert result.items[0].eval_result_id is not None
    assert result.mean_score == 1.0
    with repo._open() as conn:
        row = conn.execute(
            "SELECT gold_model, embedding_score, judge_reason FROM eval_results"
        ).fetchone()
    assert row[0] == f"dataset:{dataset.id}"
    assert row[1] == 1.0
    assert "somm eval run" in row[2]
    receipts = repo.eval_receipts(run_id=result.run_id)
    assert len(receipts) == 1
    assert receipts[0].eval_result_id == result.items[0].eval_result_id
    assert receipts[0].payload["dataset_item_id"] == result.items[0].item_id


def test_run_dataset_eval_fails_below_threshold(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, _dataset, _item = _seed_dataset(repo)

    result = run_dataset_eval(
        repo,
        project="eval-test",
        workload="qa",
        dataset="golden",
        threshold=0.9,
        generate=lambda _item: _generated(repo, "eval-test", wl.id, "wrong answer"),
    )

    assert result.passed is False
    assert result.n_passed == 0
    assert result.mean_score < 0.9


def test_run_dataset_eval_reports_generation_errors(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    _wl, _dataset, _item = _seed_dataset(repo)

    def fail(_item):
        raise RuntimeError("provider down")

    result = run_dataset_eval(
        repo,
        project="eval-test",
        workload="qa",
        dataset="golden",
        threshold=0.9,
        generate=fail,
    )

    assert result.passed is False
    assert result.n_errors == 1
    assert "provider down" in result.items[0].error


def test_grade_pairwise_ab_records_receipt(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, dataset, _item = _seed_dataset(repo)
    item = repo.dataset_items(dataset.id)[0]
    candidate_a = _generated(repo, "eval-test", wl.id, "gold response")
    candidate_b = _generated(repo, "eval-test", wl.id, "wrong answer")

    result = grade_pairwise_ab(
        repo,
        item=item,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )

    assert result.winner == "a"
    assert result.score_a == 1.0
    assert result.score_b < result.score_a
    receipts = repo.eval_receipts(receipt_type="pairwise_ab")
    assert len(receipts) == 1
    assert receipts[0].id == result.receipt_id
    assert receipts[0].candidate_a_call_id == candidate_a.call_id
    assert receipts[0].candidate_b_call_id == candidate_b.call_id
    assert receipts[0].payload["winner"] == "a"

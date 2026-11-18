from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from somm.evals import JudgeGrade, grade_pairwise_ab, run_dataset_eval
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
    assert result.items[0].generated_call_id is None


def test_judge_failure_keeps_the_recorded_generation(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, _, _ = _seed_dataset(repo)
    generated = _generated(repo, "eval-test", wl.id, "answer")

    def fail_judge(_item, _generated):
        raise RuntimeError("grader unavailable")

    result = run_dataset_eval(
        repo, project="eval-test", workload="qa", dataset="golden",
        generate=lambda _item: generated, judge=fail_judge,
    )

    assert not result.passed
    assert result.n_errors == 1
    assert result.items[0].generated_call_id == generated.call_id
    assert result.items[0].eval_result_id is None
    assert "grader unavailable" in result.items[0].error


def test_run_dataset_eval_uses_binary_judge_and_links_judge_calls(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, _dataset, _item = _seed_dataset(repo)
    judge_result = _generated(repo, "eval-test", wl.id, '{"criteria": []}')

    result = run_dataset_eval(
        repo,
        project="eval-test",
        workload="qa",
        dataset="golden",
        threshold=0.8,
        generate=lambda _item: _generated(
            repo, "eval-test", wl.id, "wording unlike the reference"
        ),
        judge=lambda _item, _generated_result: JudgeGrade(
            score=1.0,
            reason={"criteria": [{"name": "correctness", "pass": True}]},
            call_ids=(judge_result.call_id,),
        ),
        implementation="abc123",
    )

    assert result.passed is True
    assert result.items[0].judge_score == 1.0
    with repo._open() as conn:
        linked = conn.execute(
            "SELECT observation_role, source_call_id, eval_result_id "
            "FROM calls WHERE id = ?",
            (judge_result.call_id,),
        ).fetchone()
        reason = conn.execute("SELECT judge_reason FROM eval_results").fetchone()[0]
    assert linked[0] == "dataset_judge"
    assert linked[1] == result.items[0].generated_call_id
    assert linked[2] == result.items[0].eval_result_id
    assert "binary_judge" in reason
    receipt = repo.eval_receipts(run_id=result.run_id)[0]
    assert receipt.payload["judge_call_ids"] == [judge_result.call_id]
    assert receipt.payload["implementation"] == "abc123"


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


def test_missing_judge_call_rolls_back_the_entire_grade(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, _, _ = _seed_dataset(repo)
    judge = _generated(repo, "eval-test", wl.id, "grade")
    generated = _generated(repo, "eval-test", wl.id, "answer")

    result = run_dataset_eval(
        repo, project="eval-test", workload="qa", dataset="golden",
        generate=lambda _item: generated,
        judge=lambda _item, _result: JudgeGrade(1.0, {}, (judge.call_id, "missing-call")),
        record_timeout_s=0,
    )

    assert result.n_errors == 1
    assert result.items[0].generated_call_id == generated.call_id
    assert not repo.eval_receipts(run_id=result.run_id)
    with repo._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone() == (0,)
        assert conn.execute("SELECT eval_result_id FROM calls WHERE id = ?", (judge.call_id,)).fetchone() == (None,)


def test_grade_waits_for_judge_recorder_before_starting_transaction(tmp_path, monkeypatch):
    from somm import evals

    path = tmp_path / "calls.sqlite"
    repo, recorder = Repository(path), Repository(path)
    wl, _, _ = _seed_dataset(repo)
    generated = _generated(repo, "eval-test", wl.id, "answer")
    pending = replace(repo.get_call(generated.call_id), id="pending-judge")
    waits = []

    def publish(_delay):
        waits.append(True)
        recorder.write_call(pending)

    monkeypatch.setattr(evals.time, "sleep", publish)
    result = run_dataset_eval(
        repo, project="eval-test", workload="qa", dataset="golden",
        generate=lambda _: generated, judge=lambda *_: JudgeGrade(1.0, {}, (pending.id,)),
    )
    assert waits == [True]
    assert result.passed
    assert repo.get_call(pending.id).eval_result_id == result.items[0].eval_result_id


def test_unusable_scores_and_unsuccessful_candidates_cannot_select_a_winner(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, _, item = _seed_dataset(repo)
    generated = _generated(repo, "eval-test", wl.id, "gold response")
    pair = dict(item=item, candidate_a=generated, candidate_b=generated)
    with pytest.raises(ValueError, match="tie_margin"):
        grade_pairwise_ab(repo, **pair, tie_margin=float("nan"))
    with pytest.raises(ValueError, match="timeout"):
        grade_pairwise_ab(repo, **{**pair, "candidate_b": replace(generated, outcome=Outcome.TIMEOUT)})
    assert not repo.eval_receipts(receipt_type="pairwise_ab")
    args = dict(project="eval-test", workload="qa", dataset="golden", generate=lambda _: generated)
    with pytest.raises(ValueError, match="threshold"):
        run_dataset_eval(repo, **args, threshold=float("nan"))
    with pytest.raises(ValueError, match="record_timeout"):
        run_dataset_eval(repo, **args, record_timeout_s=float("inf"))
    bad = run_dataset_eval(repo, **args, judge=lambda *_: JudgeGrade(float("nan"), {}))
    assert bad.n_errors == 1
    assert bad.items[0].generated_call_id == generated.call_id
    assert not repo.eval_receipts(run_id=bad.run_id)


def test_provider_ids_are_scalar_and_keep_alias_precedence():
    from somm_core.parse import provider_custody

    assert provider_custody({"request_id":"primary", "id":"secondary", "invoice_id":42}) == ("primary", "42")
    assert provider_custody({"request_id":{"text":"private body"}, "requestId":" ", "id":"fallback",
                             "billing_id":True, "billingId":["bad"]}) == ("fallback", None)

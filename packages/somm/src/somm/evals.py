"""Synchronous dataset eval runner."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass

from somm_core.graders import GradeScores, grade_response_pair
from somm_core.models import DatasetItem, Outcome, SommResult
from somm_core.parse import stable_hash
from somm_core.repository import Repository


@dataclass(frozen=True, slots=True)
class EvalItemResult:
    item_id: str
    source_call_id: str | None
    generated_call_id: str | None
    eval_result_id: int | None
    score: float
    passed: bool
    structural_score: float | None
    text_similarity_score: float | None
    judge_score: float | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    run_id: str
    project: str
    workload: str
    workload_id: str
    dataset: str
    dataset_id: str
    threshold: float
    n_items: int
    n_passed: int
    n_errors: int
    mean_score: float
    passed: bool
    items: list[EvalItemResult]

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "project": self.project,
            "workload": self.workload,
            "workload_id": self.workload_id,
            "dataset": self.dataset,
            "dataset_id": self.dataset_id,
            "threshold": self.threshold,
            "n_items": self.n_items,
            "n_passed": self.n_passed,
            "n_errors": self.n_errors,
            "mean_score": self.mean_score,
            "passed": self.passed,
            "items": [asdict(item) for item in self.items],
        }


GenerateDatasetItem = Callable[[DatasetItem], SommResult]


@dataclass(frozen=True, slots=True)
class PairwiseEvalResult:
    dataset_item_id: str
    candidate_a_call_id: str
    candidate_b_call_id: str
    score_a: float
    score_b: float
    winner: str
    margin: float
    receipt_id: str


def run_dataset_eval(
    repo: Repository,
    *,
    project: str,
    workload: str,
    dataset: str,
    generate: GenerateDatasetItem,
    threshold: float = 0.8,
    record_timeout_s: float = 5.0,
) -> EvalRunResult:
    """Run a workload against a durable dataset and persist eval_results rows."""

    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    wl = repo.workload_by_name(workload, project)
    if wl is None:
        raise ValueError(f"workload {workload!r} not registered in project {project!r}")
    ds = repo.get_dataset(project=project, workload_id=wl.id, name=dataset)
    if ds is None:
        raise ValueError(
            f"dataset {dataset!r} not found for workload {workload!r} "
            f"in project {project!r}"
        )
    dataset_items = repo.dataset_items(ds.id)
    if not dataset_items:
        raise ValueError(f"dataset {dataset!r} has no items")

    run_id = str(uuid.uuid4())
    results: list[EvalItemResult] = []
    for item in dataset_items:
        try:
            generated = generate(item)
            if not _wait_for_call(repo, generated.call_id, timeout_s=record_timeout_s):
                raise RuntimeError(
                    f"generated call {generated.call_id!r} was not committed before timeout"
                )
            if generated.outcome != Outcome.OK:
                eval_result_id = _record_dataset_eval(
                    repo,
                    item=item,
                    generated=generated,
                    scores=None,
                    score=0.0,
                    passed=False,
                    error=f"outcome={generated.outcome.value}",
                    run_id=run_id,
                    threshold=threshold,
                )
                results.append(
                    EvalItemResult(
                        item_id=item.id,
                        source_call_id=item.source_call_id,
                        generated_call_id=generated.call_id,
                        eval_result_id=eval_result_id,
                        score=0.0,
                        passed=False,
                        structural_score=None,
                        text_similarity_score=None,
                        judge_score=None,
                        error=f"outcome={generated.outcome.value}",
                    )
                )
                continue
            scores = grade_response_pair(generated.text, item.expected_response_body)
            score = _combined_score(scores)
            passed = score >= threshold
            eval_result_id = _record_dataset_eval(
                repo,
                item=item,
                generated=generated,
                scores=scores,
                score=score,
                passed=passed,
                error=None,
                run_id=run_id,
                threshold=threshold,
            )
            results.append(
                EvalItemResult(
                    item_id=item.id,
                    source_call_id=item.source_call_id,
                    generated_call_id=generated.call_id,
                    eval_result_id=eval_result_id,
                    score=score,
                    passed=passed,
                    structural_score=scores.structural_score,
                    text_similarity_score=scores.text_similarity_score,
                    judge_score=scores.judge_score,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad item should not hide the run summary
            results.append(
                EvalItemResult(
                    item_id=item.id,
                    source_call_id=item.source_call_id,
                    generated_call_id=None,
                    eval_result_id=None,
                    score=0.0,
                    passed=False,
                    structural_score=None,
                    text_similarity_score=None,
                    judge_score=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    mean_score = sum(item.score for item in results) / len(results)
    n_errors = sum(1 for item in results if item.error)
    n_passed = sum(1 for item in results if item.passed)
    return EvalRunResult(
        run_id=run_id,
        project=project,
        workload=workload,
        workload_id=wl.id,
        dataset=dataset,
        dataset_id=ds.id,
        threshold=threshold,
        n_items=len(results),
        n_passed=n_passed,
        n_errors=n_errors,
        mean_score=mean_score,
        passed=mean_score >= threshold and n_errors == 0,
        items=results,
    )


def grade_pairwise_ab(
    repo: Repository,
    *,
    item: DatasetItem,
    candidate_a: SommResult,
    candidate_b: SommResult,
    tie_margin: float = 0.0,
    record_timeout_s: float = 5.0,
) -> PairwiseEvalResult:
    """Grade two candidates against one dataset item and record a receipt."""

    if tie_margin < 0:
        raise ValueError("tie_margin must be non-negative")
    for call_id in (candidate_a.call_id, candidate_b.call_id):
        if not _wait_for_call(repo, call_id, timeout_s=record_timeout_s):
            raise RuntimeError(f"candidate call {call_id!r} was not committed before timeout")

    scores_a = grade_response_pair(candidate_a.text, item.expected_response_body)
    scores_b = grade_response_pair(candidate_b.text, item.expected_response_body)
    score_a = _combined_score(scores_a)
    score_b = _combined_score(scores_b)
    margin = abs(score_a - score_b)
    winner = "tie" if margin <= tie_margin else "a" if score_a > score_b else "b"
    payload = {
        "source": "somm pairwise ab",
        "dataset_id": item.dataset_id,
        "dataset_item_id": item.id,
        "source_call_id": item.source_call_id,
        "candidate_a": _score_payload(candidate_a, scores_a, score_a),
        "candidate_b": _score_payload(candidate_b, scores_b, score_b),
        "winner": winner,
        "margin": margin,
        "tie_margin": tie_margin,
    }
    receipt = repo.record_eval_receipt(
        receipt_type="pairwise_ab",
        payload=payload,
        dataset_id=item.dataset_id,
        dataset_item_id=item.id,
        source_call_id=item.source_call_id,
        candidate_a_call_id=candidate_a.call_id,
        candidate_b_call_id=candidate_b.call_id,
        winner=winner,
        score=margin,
        threshold=tie_margin,
    )
    return PairwiseEvalResult(
        dataset_item_id=item.id,
        candidate_a_call_id=candidate_a.call_id,
        candidate_b_call_id=candidate_b.call_id,
        score_a=score_a,
        score_b=score_b,
        winner=winner,
        margin=margin,
        receipt_id=receipt.id,
    )


def _combined_score(scores: GradeScores) -> float:
    if scores.judge_score is not None:
        return float(scores.judge_score)
    if scores.structural_score is not None:
        return float(scores.structural_score)
    if scores.text_similarity_score is not None:
        return float(scores.text_similarity_score)
    return 0.0


def _wait_for_call(repo: Repository, call_id: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if repo.get_call(call_id) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _record_dataset_eval(
    repo: Repository,
    *,
    item: DatasetItem,
    generated: SommResult,
    scores: GradeScores | None,
    score: float,
    passed: bool,
    error: str | None,
    run_id: str,
    threshold: float,
) -> int:
    reason = [
        {
            "source": "somm eval run",
            "dataset_id": item.dataset_id,
            "dataset_item_id": item.id,
            "source_call_id": item.source_call_id,
            "score": score,
            "passed": passed,
            "error": error,
        }
    ]
    eval_result_id = repo.record_eval_result(
        call_id=generated.call_id,
        gold_model=f"dataset:{item.dataset_id}",
        gold_response_hash=stable_hash(item.expected_response_body),
        structural_score=scores.structural_score if scores else None,
        embedding_score=scores.text_similarity_score if scores else None,
        judge_score=scores.judge_score if scores else None,
        judge_reason=json.dumps(reason, sort_keys=True),
    )
    repo.record_eval_receipt(
        receipt_type="dataset_run",
        payload=reason[0],
        eval_result_id=eval_result_id,
        run_id=run_id,
        call_id=generated.call_id,
        dataset_id=item.dataset_id,
        dataset_item_id=item.id,
        source_call_id=item.source_call_id,
        score=score,
        threshold=threshold,
    )
    return eval_result_id


def _score_payload(result: SommResult, scores: GradeScores, score: float) -> dict:
    return {
        "call_id": result.call_id,
        "provider": result.provider,
        "model": result.model,
        "score": score,
        "structural_score": scores.structural_score,
        "text_similarity_score": scores.text_similarity_score,
        "judge_score": scores.judge_score,
    }


__all__ = [
    "EvalItemResult",
    "EvalRunResult",
    "PairwiseEvalResult",
    "GenerateDatasetItem",
    "grade_pairwise_ab",
    "run_dataset_eval",
]

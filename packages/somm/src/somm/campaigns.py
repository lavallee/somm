"""Experiment campaign harness built on durable dataset evals."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from somm_core.models import Campaign, CampaignEvent, DatasetItem, SommResult
from somm_core.repository import Repository

from somm.evals import EvalRunResult, run_dataset_eval

GenerateDatasetItem = Callable[[DatasetItem], SommResult]


@dataclass(frozen=True, slots=True)
class MetricContract:
    """Round-level metric contract for campaign decisions."""

    metric: str = "mean_score"
    threshold: float = 0.8
    direction: str = "gte"

    def __post_init__(self) -> None:
        if self.metric not in {"mean_score", "pass_rate", "error_rate"}:
            raise ValueError("metric must be one of: mean_score, pass_rate, error_rate")
        if self.direction not in {"gte", "lte"}:
            raise ValueError("direction must be 'gte' or 'lte'")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

    def value(self, run: EvalRunResult) -> float:
        if self.metric == "mean_score":
            return run.mean_score
        if self.metric == "pass_rate":
            return run.n_passed / run.n_items if run.n_items else 0.0
        if self.metric == "error_rate":
            return run.n_errors / run.n_items if run.n_items else 0.0
        raise ValueError(f"unsupported metric {self.metric!r}")

    def passes(self, score: float) -> bool:
        if self.direction == "gte":
            return score >= self.threshold
        return score <= self.threshold

    def improved(self, score: float, best_score: float | None, min_delta: float) -> bool:
        if best_score is None:
            return True
        if self.direction == "gte":
            return score >= best_score + min_delta
        return score <= best_score - min_delta


@dataclass(frozen=True, slots=True)
class RoundUsage:
    tokens_in: int
    tokens_out: int
    total_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class CampaignRunResult:
    campaign: Campaign
    events: list[CampaignEvent]
    stop_reason: str
    best_score: float | None
    total_tokens: int
    total_cost_usd: float
    passed: bool

    def as_dict(self) -> dict:
        return {
            "campaign": _campaign_dict(self.campaign),
            "stop_reason": self.stop_reason,
            "best_score": self.best_score,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "passed": self.passed,
            "events": [_event_dict(event) for event in self.events],
        }

    def jsonl(self) -> str:
        lines = [json.dumps(event.payload, sort_keys=True) for event in self.events]
        return "\n".join(lines) + ("\n" if lines else "")


def run_eval_campaign(
    repo: Repository,
    *,
    project: str,
    workload: str,
    dataset: str,
    generate: GenerateDatasetItem,
    contract: MetricContract | None = None,
    name: str | None = None,
    max_rounds: int = 5,
    token_budget: int | None = None,
    plateau_window: int = 2,
    min_delta: float = 0.0,
    eval_threshold: float | None = None,
    record_timeout_s: float = 5.0,
) -> CampaignRunResult:
    """Run repeated dataset eval rounds and persist a keep/revert campaign log."""

    contract = contract or MetricContract()
    _validate_campaign_limits(
        max_rounds=max_rounds,
        token_budget=token_budget,
        plateau_window=plateau_window,
        min_delta=min_delta,
    )
    wl = repo.workload_by_name(workload, project)
    if wl is None:
        raise ValueError(f"workload {workload!r} not registered in project {project!r}")
    ds = repo.get_dataset(project=project, workload_id=wl.id, name=dataset)
    if ds is None:
        raise ValueError(
            f"dataset {dataset!r} not found for workload {workload!r} in project {project!r}"
        )

    campaign_name = name or f"{workload}/{dataset}/{uuid.uuid4().hex[:8]}"
    campaign = repo.create_campaign(
        project=project,
        workload_id=wl.id,
        dataset_id=ds.id,
        name=campaign_name,
        metric=contract.metric,
        direction=contract.direction,
        threshold=contract.threshold,
        token_budget=token_budget,
        max_rounds=max_rounds,
        plateau_window=plateau_window,
        min_delta=min_delta,
        metadata={
            "source": "somm campaign run",
            "workload": workload,
            "dataset": dataset,
            "eval_threshold": eval_threshold if eval_threshold is not None else contract.threshold,
        },
    )

    events: list[CampaignEvent] = []
    sequence = 1
    total_tokens = 0
    total_cost_usd = 0.0
    best_score: float | None = None
    stale_rounds = 0
    stop_reason = "max_rounds"
    status = "complete"

    events.append(
        _record_event(
            repo,
            campaign,
            sequence=sequence,
            event_type="campaign_started",
            action="start",
            payload={
                "jsonl_version": 1,
                "event": "campaign_started",
                "campaign_id": campaign.id,
                "name": campaign.name,
                "project": project,
                "workload": workload,
                "workload_id": wl.id,
                "dataset": dataset,
                "dataset_id": ds.id,
                "metric": asdict(contract),
                "max_rounds": max_rounds,
                "token_budget": token_budget,
                "plateau_window": plateau_window,
                "min_delta": min_delta,
            },
        )
    )
    sequence += 1

    try:
        for round_no in range(1, max_rounds + 1):
            run = run_dataset_eval(
                repo,
                project=project,
                workload=workload,
                dataset=dataset,
                generate=generate,
                threshold=eval_threshold if eval_threshold is not None else contract.threshold,
                record_timeout_s=record_timeout_s,
            )
            usage = _usage_for_run(repo, run)
            total_tokens += usage.total_tokens
            total_cost_usd += usage.cost_usd

            score = contract.value(run)
            contract_passed = contract.passes(score)
            improved = contract.improved(score, best_score, min_delta)
            if improved:
                best_score = score
                stale_rounds = 0
            else:
                stale_rounds += 1

            action = "keep" if contract_passed and improved else "revert"
            events.append(
                _record_event(
                    repo,
                    campaign,
                    sequence=sequence,
                    run_id=run.run_id,
                    event_type="round_completed",
                    action=action,
                    metric_score=score,
                    threshold=contract.threshold,
                    tokens_in=usage.tokens_in,
                    tokens_out=usage.tokens_out,
                    total_tokens=total_tokens,
                    cost_usd=usage.cost_usd,
                    payload={
                        "jsonl_version": 1,
                        "event": "round_completed",
                        "campaign_id": campaign.id,
                        "round": round_no,
                        "run_id": run.run_id,
                        "action": action,
                        "metric": asdict(contract),
                        "metric_score": score,
                        "contract_passed": contract_passed,
                        "improved": improved,
                        "best_score": best_score,
                        "stale_rounds": stale_rounds,
                        "usage": asdict(usage),
                        "total_tokens": total_tokens,
                        "total_cost_usd": total_cost_usd,
                        "eval": run.as_dict(),
                    },
                )
            )
            sequence += 1

            if token_budget is not None and total_tokens >= token_budget:
                stop_reason = "token_budget"
                status = "budget_exhausted"
                break
            if stale_rounds >= plateau_window:
                stop_reason = "plateau"
                status = "plateau"
                break
        else:
            stop_reason = "max_rounds"
            status = "complete"
    except Exception as exc:
        stop_reason = "error"
        status = "failed"
        events.append(
            _record_event(
                repo,
                campaign,
                sequence=sequence,
                event_type="campaign_failed",
                action="stop",
                metric_score=best_score,
                threshold=contract.threshold,
                total_tokens=total_tokens,
                cost_usd=0.0,
                payload={
                    "jsonl_version": 1,
                    "event": "campaign_failed",
                    "campaign_id": campaign.id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "best_score": best_score,
                    "total_tokens": total_tokens,
                    "total_cost_usd": total_cost_usd,
                },
            )
        )
        repo.finish_campaign(
            campaign.id,
            status=status,
            best_score=best_score,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
        )
        raise

    passed = best_score is not None and contract.passes(best_score)
    events.append(
        _record_event(
            repo,
            campaign,
            sequence=sequence,
            event_type="campaign_stopped",
            action="stop",
            metric_score=best_score,
            threshold=contract.threshold,
            total_tokens=total_tokens,
            cost_usd=0.0,
            payload={
                "jsonl_version": 1,
                "event": "campaign_stopped",
                "campaign_id": campaign.id,
                "status": status,
                "stop_reason": stop_reason,
                "best_score": best_score,
                "passed": passed,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
            },
        )
    )
    campaign = repo.finish_campaign(
        campaign.id,
        status=status,
        best_score=best_score,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
    )
    return CampaignRunResult(
        campaign=campaign,
        events=events,
        stop_reason=stop_reason,
        best_score=best_score,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        passed=passed,
    )


def write_campaign_jsonl(result: CampaignRunResult, path: str | Path) -> None:
    Path(path).write_text(result.jsonl(), encoding="utf-8")


def _record_event(
    repo: Repository,
    campaign: Campaign,
    *,
    sequence: int,
    event_type: str,
    action: str,
    payload: dict,
    run_id: str | None = None,
    metric_score: float | None = None,
    threshold: float | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
) -> CampaignEvent:
    return repo.record_campaign_event(
        campaign.id,
        sequence=sequence,
        run_id=run_id,
        event_type=event_type,
        action=action,
        metric_score=metric_score,
        threshold=threshold,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        payload=payload,
    )


def _usage_for_run(repo: Repository, run: EvalRunResult) -> RoundUsage:
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    seen: set[str] = set()
    for item in run.items:
        if item.generated_call_id is None or item.generated_call_id in seen:
            continue
        seen.add(item.generated_call_id)
        call = repo.get_call(item.generated_call_id)
        if call is None:
            continue
        tokens_in += int(call.tokens_in or 0)
        tokens_out += int(call.tokens_out or 0)
        cost_usd += float(call.cost_usd or 0.0)
    return RoundUsage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cost_usd=cost_usd,
    )


def _validate_campaign_limits(
    *,
    max_rounds: int,
    token_budget: int | None,
    plateau_window: int,
    min_delta: float,
) -> None:
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if token_budget is not None and token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if plateau_window <= 0:
        raise ValueError("plateau_window must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")


def _campaign_dict(campaign: Campaign) -> dict:
    out = asdict(campaign)
    for key in ("created_at", "updated_at", "completed_at"):
        value = out.get(key)
        if value is not None:
            out[key] = value.isoformat()
    return out


def _event_dict(event: CampaignEvent) -> dict:
    out = asdict(event)
    if event.created_at is not None:
        out["created_at"] = event.created_at.isoformat()
    return out


__all__ = [
    "CampaignRunResult",
    "GenerateDatasetItem",
    "MetricContract",
    "RoundUsage",
    "run_eval_campaign",
    "write_campaign_jsonl",
]

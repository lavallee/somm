"""workload_frontier — adequacy rollup per (provider, model) for a workload.

Verifies the v_calls_classified-backed rollup correctly splits capability vs
detractor failures and emits fitness flags only when the workload sets the
matching constraint.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm_core.models import Call, Outcome
from somm_core.repository import Repository, _percentiles


def _mk_call(
    *,
    workload_id: str,
    provider: str,
    model: str,
    outcome: Outcome,
    latency_ms: int = 100,
    cost_usd: float = 0.0,
    project: str = "p",
    ts: datetime | None = None,
) -> Call:
    return Call(
        id=str(uuid.uuid4()),
        ts=ts or datetime.now(UTC),
        project=project,
        workload_id=workload_id,
        prompt_id=None,
        provider=provider,
        model=model,
        tokens_in=10,
        tokens_out=10,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        outcome=outcome,
        error_kind=None,
        prompt_hash="h",
        response_hash="h",
    )


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "somm.sqlite")


def test_frontier_splits_capability_from_detractor(repo: Repository) -> None:
    wl = repo.register_workload(name="w1", project="p")
    # 4 calls on (ollama, qwen): 2 ok, 1 bad_json (capability), 1 rate_limit (detractor)
    repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="qwen", outcome=Outcome.OK, latency_ms=100, cost_usd=0.0))
    repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="qwen", outcome=Outcome.OK, latency_ms=200, cost_usd=0.0))
    repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="qwen", outcome=Outcome.BAD_JSON, latency_ms=150))
    repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="qwen", outcome=Outcome.RATE_LIMIT, latency_ms=50))

    frontier = repo.workload_frontier(wl.id)
    assert len(frontier) == 1
    row = frontier[0]
    assert row["provider"] == "ollama"
    assert row["model"] == "qwen"
    assert row["n_calls"] == 4
    assert row["n_ok"] == 2
    assert row["n_capability_failures"] == 1
    assert row["n_detractors"] == 1
    assert row["capability_failure_rate"] == 0.25
    assert row["detractor_rate"] == 0.25


def test_frontier_fitness_unset_when_constraint_unset(repo: Repository) -> None:
    wl = repo.register_workload(name="w1", project="p")
    repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="qwen", outcome=Outcome.OK, cost_usd=0.001))
    row = repo.workload_frontier(wl.id)[0]
    assert row["fitness"]["exceeds_max_capability_failure_rate"] is None
    assert row["fitness"]["exceeds_max_p95_latency_ms"] is None
    assert row["fitness"]["exceeds_max_cost_per_call_usd"] is None


def test_frontier_fitness_flags_on_violation(repo: Repository) -> None:
    wl = repo.register_workload(
        name="w1",
        project="p",
        max_p95_latency_ms=100,
        max_capability_failure_rate=0.10,
        max_cost_per_call_usd=0.0001,
    )
    # 10 ok, 2 bad_json → cap rate 16.7% (> 10%)
    for _ in range(10):
        repo.write_call(_mk_call(workload_id=wl.id, provider="paid", model="big", outcome=Outcome.OK, latency_ms=300, cost_usd=0.0005))
    for _ in range(2):
        repo.write_call(_mk_call(workload_id=wl.id, provider="paid", model="big", outcome=Outcome.BAD_JSON, latency_ms=300))

    row = repo.workload_frontier(wl.id)[0]
    assert row["fitness"]["exceeds_max_capability_failure_rate"] is True
    assert row["fitness"]["exceeds_max_p95_latency_ms"] is True
    assert row["fitness"]["exceeds_max_cost_per_call_usd"] is True


def test_frontier_orders_fittest_first(repo: Repository) -> None:
    wl = repo.register_workload(name="w1", project="p")
    # ollama/cheap: 100% capability fail
    for _ in range(3):
        repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="cheap", outcome=Outcome.BAD_JSON))
    # ollama/good: 0% fail, $0.001/call
    for _ in range(3):
        repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="good", outcome=Outcome.OK, cost_usd=0.001))
    # paid/best: 0% fail, $0.005/call
    for _ in range(3):
        repo.write_call(_mk_call(workload_id=wl.id, provider="paid", model="best", outcome=Outcome.OK, cost_usd=0.005))

    rows = repo.workload_frontier(wl.id)
    # cap-fail-rate ascending, then mean_cost ascending — good < best < cheap
    assert [(r["provider"], r["model"]) for r in rows] == [
        ("ollama", "good"),
        ("paid", "best"),
        ("ollama", "cheap"),
    ]


def test_frontier_returns_empty_for_unknown_workload(repo: Repository) -> None:
    assert repo.workload_frontier("nonexistent") == []


def test_frontier_excludes_calls_outside_window(repo: Repository) -> None:
    wl = repo.register_workload(name="w1", project="p")
    old_ts = datetime.fromisoformat("2020-01-01T00:00:00+00:00")
    repo.write_call(_mk_call(workload_id=wl.id, provider="ollama", model="m", outcome=Outcome.OK, ts=old_ts))
    assert repo.workload_frontier(wl.id, since_days=7) == []


def test_set_workload_constraints_partial_update(repo: Repository) -> None:
    wl = repo.register_workload(
        name="w1",
        project="p",
        max_p95_latency_ms=500,
        max_capability_failure_rate=0.05,
    )
    repo.set_workload_constraints(wl.id, max_cost_per_call_usd=0.01)
    refreshed = repo.workload_by_name("w1", "p")
    assert refreshed is not None
    assert refreshed.max_p95_latency_ms == 500
    assert refreshed.max_capability_failure_rate == 0.05
    assert refreshed.max_cost_per_call_usd == 0.01


def test_set_workload_constraints_clear(repo: Repository) -> None:
    wl = repo.register_workload(
        name="w1",
        project="p",
        max_p95_latency_ms=500,
        max_capability_failure_rate=0.05,
    )
    repo.set_workload_constraints(wl.id, clear=True)
    refreshed = repo.workload_by_name("w1", "p")
    assert refreshed is not None
    assert refreshed.max_p95_latency_ms is None
    assert refreshed.max_capability_failure_rate is None
    assert refreshed.max_cost_per_call_usd is None


@pytest.mark.parametrize(
    ("csv", "expected_p50", "expected_p95"),
    [
        (None, None, None),
        ("", None, None),
        ("100", 100, 100),
        ("100,200", 100, 200),
        ("100,150,200,250,300,350,400,450,500,1000", 300, 1000),
    ],
)
def test_percentiles_helper(csv: str | None, expected_p50: int | None, expected_p95: int | None) -> None:
    p50, p95 = _percentiles(csv)
    assert p50 == expected_p50
    assert p95 == expected_p95

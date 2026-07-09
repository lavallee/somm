from __future__ import annotations

import pytest
from somm_core.repository import Repository


def test_register_workload_records_initial_config_revision(tmp_path):
    repo = Repository(tmp_path / "somm.sqlite")

    wl = repo.register_workload(
        name="w1",
        project="p",
        budget_cap_usd_daily=5.0,
        max_p95_latency_ms=500,
        max_capability_failure_rate=0.05,
    )

    assert repo.current_workload_revision(wl.id) == {
        "max_p95_latency_ms": 500,
        "max_capability_failure_rate": 0.05,
        "max_cost_per_call_usd": None,
        "budget_cap_usd_daily": 5.0,
        "shadow_config": None,
        "policy": None,
    }

    repo.register_workload(name="w1", project="p")
    revisions = repo.workload_revisions(wl.id)
    assert [row["revision"] for row in revisions] == [1]


def test_set_workload_constraints_records_snapshots_and_keeps_live_row(tmp_path):
    repo = Repository(tmp_path / "somm.sqlite")
    wl = repo.register_workload(
        name="w1",
        project="p",
        budget_cap_usd_daily=2.0,
        max_p95_latency_ms=500,
    )

    repo.set_workload_constraints(wl.id, max_cost_per_call_usd=0.01)
    repo.set_workload_constraints(
        wl.id,
        max_p95_latency_ms=250,
        max_capability_failure_rate=0.1,
    )

    revisions = repo.workload_revisions(wl.id)
    assert [row["revision"] for row in revisions] == [1, 2, 3]
    assert revisions[1]["config"] == {
        "max_p95_latency_ms": 500,
        "max_capability_failure_rate": None,
        "max_cost_per_call_usd": 0.01,
        "budget_cap_usd_daily": 2.0,
        "shadow_config": None,
        "policy": None,
    }
    assert revisions[2]["config"] == {
        "max_p95_latency_ms": 250,
        "max_capability_failure_rate": 0.1,
        "max_cost_per_call_usd": 0.01,
        "budget_cap_usd_daily": 2.0,
        "shadow_config": None,
        "policy": None,
    }
    assert repo.current_workload_revision(wl.id) == revisions[2]["config"]

    refreshed = repo.workload_by_name("w1", "p")
    assert refreshed is not None
    assert refreshed.max_p95_latency_ms == 250
    assert refreshed.max_capability_failure_rate == 0.1
    assert refreshed.max_cost_per_call_usd == 0.01


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "policy",
    [
        {"timeout_s": None},
        {"retry": {"backoff_s": None}},
        {"retry": {"deadline_s": None}},
        {"retry": {"max": None}},
    ],
)
def test_set_workload_policy_rejects_non_finite_numbers(tmp_path, policy, bad_value):
    repo = Repository(tmp_path / "somm.sqlite")
    wl = repo.register_workload(name="w1", project="p")

    if "timeout_s" in policy:
        candidate = {"timeout_s": bad_value}
    else:
        key = next(iter(policy["retry"]))
        candidate = {"retry": {key: bad_value}}

    with pytest.raises(ValueError, match="finite|integer"):
        repo.set_workload_policy(wl.id, candidate)


def test_shadow_config_diff_and_forward_only_rollback(tmp_path):
    repo = Repository(tmp_path / "somm.sqlite")
    wl = repo.register_workload(
        name="w1",
        project="p",
        budget_cap_usd_daily=3.0,
        max_cost_per_call_usd=0.05,
    )
    shadow_config = {
        "gold_provider": "openai",
        "gold_model": "gpt-test",
        "sample_rate": 0.25,
    }

    repo.set_shadow_config(wl.id, shadow_config)
    repo.set_workload_constraints(wl.id, max_cost_per_call_usd=0.01)

    revisions = repo.workload_revisions(wl.id)
    assert revisions[1]["config"]["shadow_config"] == shadow_config
    assert repo.workload_revision_diff(wl.id, 2, 3) == {
        "max_cost_per_call_usd": {"old": 0.05, "new": 0.01}
    }

    new_revision = repo.rollback_workload(wl.id, 2, created_by="test")

    assert new_revision == 4
    revisions = repo.workload_revisions(wl.id)
    assert [row["revision"] for row in revisions] == [1, 2, 3, 4]
    assert revisions[-1]["created_by"] == "test"
    assert revisions[-1]["config"] == revisions[1]["config"]
    assert repo.current_workload_revision(wl.id) == revisions[1]["config"]
    assert repo.get_shadow_config(wl.id) == shadow_config

    refreshed = repo.workload_by_name("w1", "p")
    assert refreshed is not None
    assert refreshed.max_cost_per_call_usd == 0.05
    assert refreshed.budget_cap_usd_daily == 3.0


def test_revision_sequences_are_independent_per_workload(tmp_path):
    repo = Repository(tmp_path / "somm.sqlite")
    first = repo.register_workload(name="first", project="p")
    second = repo.register_workload(name="second", project="p")

    repo.set_workload_constraints(first.id, max_p95_latency_ms=100)
    repo.set_workload_constraints(first.id, max_cost_per_call_usd=0.01)
    repo.set_shadow_config(second.id, {"gold_provider": "local"})

    assert [row["revision"] for row in repo.workload_revisions(first.id)] == [1, 2, 3]
    assert [row["revision"] for row in repo.workload_revisions(second.id)] == [1, 2]


def test_first_mutation_backfills_initial_revision_for_existing_workload(tmp_path):
    repo = Repository(tmp_path / "somm.sqlite")
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO workloads "
            "(id, name, project, max_p95_latency_ms) "
            "VALUES ('legacy', 'legacy', 'p', 700)"
        )

    repo.set_workload_constraints("legacy", max_p95_latency_ms=300)

    revisions = repo.workload_revisions("legacy")
    assert [row["revision"] for row in revisions] == [1, 2]
    assert revisions[0]["config"]["max_p95_latency_ms"] == 700
    assert revisions[1]["config"]["max_p95_latency_ms"] == 300

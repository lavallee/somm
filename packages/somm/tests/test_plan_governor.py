from __future__ import annotations

from datetime import UTC, datetime

from somm.plan_governor import PlanGovernor
from somm_core.models import Call, Outcome
from somm_core.plans import Plan, PlanLimit
from somm_core.repository import Repository


def test_governor_blocks_enforced_learned_exhausted_limit(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(
        Call(
            id="c1",
            ts=datetime.now(UTC),
            project="plans",
            workload_id=None,
            prompt_id=None,
            provider="claude-cli",
            model="sonnet",
            tokens_in=1,
            tokens_out=0,
            latency_ms=1,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )
    plans = {
        "claude-cli": Plan(
            provider="claude-cli",
            mode="metered",
            enforce=True,
            limits=[
                PlanLimit(
                    window="5h",
                    quota=1,
                    unit="requests",
                    learned=True,
                    source="observed_429",
                )
            ],
        )
    }
    governor = PlanGovernor(plans, lambda: [repo.db_path], ttl_s=0)

    assert governor.decision("claude-cli") == "block"

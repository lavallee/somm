from __future__ import annotations

from datetime import UTC, datetime, timedelta

from somm_core.models import Call, Outcome
from somm_core.plans import (
    learn_observed_limits,
    load_plans,
    observed_ceilings,
)
from somm_core.repository import Repository


def _call(call_id: str, ts: datetime, *, tokens: int, outcome: Outcome) -> Call:
    return Call(
        id=call_id,
        ts=ts,
        project="plans",
        workload_id=None,
        prompt_id=None,
        provider="claude-cli",
        model="sonnet",
        tokens_in=tokens,
        tokens_out=0,
        latency_ms=1,
        cost_usd=0.0,
        outcome=outcome,
        error_kind="RateLimitError" if outcome == Outcome.RATE_LIMIT else None,
        error_detail="quota exceeded" if outcome == Outcome.RATE_LIMIT else None,
        prompt_hash="p",
        response_hash="r",
    )


def test_observed_ceilings_do_not_count_calls_after_quota_event(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    event_at = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
    repo.write_call(_call("before", event_at - timedelta(minutes=10), tokens=100, outcome=Outcome.OK))
    repo.write_call(_call("event", event_at, tokens=0, outcome=Outcome.RATE_LIMIT))
    repo.write_call(_call("after", event_at + timedelta(minutes=10), tokens=900, outcome=Outcome.OK))

    ceilings = observed_ceilings(
        [repo.db_path],
        "claude-cli",
        windows=("5h",),
        units=("tokens_total",),
        now=event_at + timedelta(hours=1),
    )

    assert len(ceilings) == 1
    assert ceilings[0].estimate == 100


def test_learn_observed_limits_writes_parseable_plans_toml(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    event_at = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
    repo.write_call(_call("before", event_at - timedelta(minutes=10), tokens=100, outcome=Outcome.OK))
    repo.write_call(_call("event", event_at, tokens=0, outcome=Outcome.RATE_LIMIT))
    plans_path = tmp_path / "plans.toml"

    plans = {}
    updates = learn_observed_limits([repo.db_path], plans, path=plans_path)

    assert plans_path.exists()
    assert any(
        u.provider == "claude-cli"
        and u.window == "5h"
        and u.unit == "tokens_total"
        and u.new_quota == 100
        and u.action == "added"
        for u in updates
    )
    loaded = load_plans(path=plans_path)
    learned = [
        limit
        for limit in loaded["claude-cli"].limits
        if limit.window == "5h" and limit.unit == "tokens_total"
    ][0]
    assert learned.quota == 100
    assert learned.learned is True
    assert learned.source == "observed_429"
    assert learned.n_events == 1

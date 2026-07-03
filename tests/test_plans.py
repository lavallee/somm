"""Tests for billing plans: parsing, window math, pacing states,
fleet-wide usage, the registry, and the pace-aware router gate."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from somm_core.models import Call, Outcome
from somm_core.plans import (
    LimitStatus,
    PlanLimit,
    limit_statuses,
    load_plans,
    plan_for,
    usage_in_window,
)
from somm_core.registry import fleet_db_paths, register_project
from somm_core.repository import Repository

NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# plans.toml parsing


def _write_plans(tmp_path: Path, text: str, monkeypatch) -> None:
    p = tmp_path / "plans.toml"
    p.write_text(text)
    monkeypatch.setenv("SOMM_PLANS_PATH", str(p))


def test_load_plans_full(tmp_path, monkeypatch):
    _write_plans(
        tmp_path,
        """
        [minimax]
        mode = "metered"
        plan = "coding-pro"
        soft_target_pct = 75
        enforce = true
        [[minimax.limits]]
        window = "month"
        anchor_day = 12
        quota = 40.0
        unit = "usd_equiv"
        [[minimax.limits]]
        window = "5h"
        quota = 500
        unit = "requests"

        [gemini]
        mode = "payg"
        """,
        monkeypatch,
    )
    plans = load_plans()
    mm = plans["minimax"]
    assert mm.mode == "metered" and mm.enforce and mm.soft_target_pct == 75
    assert len(mm.limits) == 2
    assert mm.limits[0].anchor_day == 12
    assert plans["gemini"].mode == "payg"


def test_load_plans_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMM_PLANS_PATH", str(tmp_path / "absent.toml"))
    assert load_plans() == {}


@pytest.mark.parametrize(
    "body",
    [
        '[x]\nmode = "subscription"',  # bad mode
        '[x]\nmode = "metered"\n[[x.limits]]\nquota = 1\nunit = "calls"',  # bad unit
        '[x]\nmode = "metered"\n[[x.limits]]\nquota = 1\nwindow = "5x"',  # bad window
    ],
)
def test_load_plans_rejects_bad_config_loudly(tmp_path, monkeypatch, body):
    _write_plans(tmp_path, body, monkeypatch)
    with pytest.raises(ValueError):
        load_plans()


def test_plan_for_defaults():
    assert plan_for("ollama", {}).mode == "free"
    assert plan_for("claude-cli", {}).mode == "metered"
    assert plan_for("gemini", {}).mode == "payg"


# ---------------------------------------------------------------------------
# window math


def test_calendar_bounds_after_anchor():
    lim = PlanLimit(window="month", quota=1, anchor_day=1)
    start, end = lim.bounds(NOW)  # Jul 3, anchor day 1
    assert (start.month, start.day) == (7, 1)
    assert (end.month, end.day) == (8, 1)


def test_calendar_bounds_before_anchor_uses_previous_month():
    lim = PlanLimit(window="month", quota=1, anchor_day=12)
    start, end = lim.bounds(NOW)  # Jul 3 < Jul 12 → window began Jun 12
    assert (start.month, start.day) == (6, 12)
    assert (end.month, end.day) == (7, 12)


def test_rolling_bounds():
    lim = PlanLimit(window="5h", quota=1)
    start, end = lim.bounds(NOW)
    assert end == NOW and (end - start) == timedelta(hours=5)
    assert PlanLimit(window="1w", quota=1).window_seconds() == 7 * 86400


# ---------------------------------------------------------------------------
# pacing states


def _status(used_pct, elapsed_frac, rolling=False, soft=80.0):
    quota = 100.0
    if rolling:
        lim = PlanLimit(window="5h", quota=quota)
        start, end = NOW - timedelta(hours=5), NOW
    else:
        lim = PlanLimit(window="month", quota=quota, anchor_day=1)
        span = timedelta(days=30)
        start = datetime.now(UTC) - span * elapsed_frac
        end = start + span
    return LimitStatus(
        provider="p", plan_name="t", limit=lim, used=used_pct,
        window_start=start, window_end=end, soft_target_pct=soft,
    )


def test_calendar_over_pace_needs_both_burn_and_soft_target():
    # burning 2x but only 30% used → still ok (early-window burst)
    assert _status(30, 0.15).state == "ok"
    # past soft target AND ahead of the calendar → over_pace
    assert _status(85, 0.5).state == "over_pace"
    # past soft target but UNDER pace (85% used, 95% elapsed) → ok
    assert _status(85, 0.95).state == "ok"
    assert _status(101, 0.5).state == "exhausted"


def test_rolling_over_pace_is_soft_target_utilization():
    assert _status(79, 1, rolling=True).state == "ok"
    assert _status(81, 1, rolling=True).state == "over_pace"
    assert _status(100, 1, rolling=True).state == "exhausted"


# ---------------------------------------------------------------------------
# usage across fleet DBs + registry


def _db_with_calls(tmp_path: Path, name: str, provider: str, n: int, cost: float):
    repo = Repository(tmp_path / name)
    wl = repo.register_workload(name="w", project=name)
    for _ in range(n):
        repo.write_call(Call(
            id=str(uuid.uuid4()), ts=datetime.now(UTC), project=name,
            workload_id=wl.id, prompt_id=None, provider=provider, model="m",
            tokens_in=10, tokens_out=5, latency_ms=1, cost_usd=cost,
            outcome=Outcome.OK, error_kind=None, prompt_hash="h", response_hash="h",
        ))
    return tmp_path / name


def test_usage_sums_across_dbs_and_units(tmp_path):
    db1 = _db_with_calls(tmp_path, "a.sqlite", "minimax", 3, 0.5)
    db2 = _db_with_calls(tmp_path, "b.sqlite", "minimax", 2, 0.25)
    lim_req = PlanLimit(window="1d", quota=100, unit="requests")
    lim_usd = PlanLimit(window="1d", quota=100, unit="usd_equiv")
    lim_tok = PlanLimit(window="1d", quota=100, unit="tokens_total")
    assert usage_in_window([db1, db2], "minimax", lim_req) == 5
    assert usage_in_window([db1, db2], "minimax", lim_usd) == pytest.approx(2.0)
    assert usage_in_window([db1, db2], "minimax", lim_tok) == 75
    assert usage_in_window([db1, db2], "gemini", lim_req) == 0


def test_registry_roundtrip_and_pruning(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(tmp_path / "registry.json"))
    db = _db_with_calls(tmp_path, "c.sqlite", "minimax", 1, 0.0)
    register_project("proj_c", db)
    register_project("gone", tmp_path / "never-existed.sqlite")
    paths = fleet_db_paths()
    assert db.resolve() in [p.resolve() for p in paths]
    assert len(paths) == 1  # vanished DB pruned


def test_limit_statuses_end_to_end(tmp_path, monkeypatch):
    _write_plans(
        tmp_path,
        '[minimax]\nmode = "metered"\nsoft_target_pct = 50\n'
        '[[minimax.limits]]\nwindow = "1d"\nquota = 4\nunit = "requests"',
        monkeypatch,
    )
    db = _db_with_calls(tmp_path, "d.sqlite", "minimax", 3, 0.0)
    (st,) = limit_statuses([db], load_plans())
    assert st.used == 3 and st.used_pct == 75.0
    assert st.state == "over_pace"  # rolling window past 50% soft target


# ---------------------------------------------------------------------------
# router gate


class _P:
    def __init__(self, name):
        self.name = name


def _governor(decisions: dict):
    class G:
        def decision(self, name):
            return decisions.get(name, "ok")
    return G()


def test_router_gate_orders_and_drops():
    from somm.routing import Router

    r = Router.__new__(Router)
    r.plan_governor = _governor({"minimax": "defer", "claude-cli": "block"})
    chain = [_P("minimax"), _P("ollama"), _P("claude-cli"), _P("gemini")]
    out = r._apply_plan_governor(chain)
    assert [p.name for p in out] == ["ollama", "gemini", "minimax"]


def test_router_gate_none_is_passthrough():
    from somm.routing import Router

    r = Router.__new__(Router)
    r.plan_governor = None
    chain = [_P("a"), _P("b")]
    assert r._apply_plan_governor(chain) == chain


def test_router_gate_survives_broken_governor():
    from somm.routing import Router

    class Boom:
        def decision(self, name):
            raise RuntimeError("governor bug")

    r = Router.__new__(Router)
    r.plan_governor = Boom()
    chain = [_P("a")]
    assert r._apply_plan_governor(chain) == chain

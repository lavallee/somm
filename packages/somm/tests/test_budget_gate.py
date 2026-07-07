"""Fail-closed budget gate: refuse calls once a workload's daily cap is reached.

The soft-warn (test_pricing_safety.py) only *warns* after the spend. This gate
blocks BEFORE dispatch when ``budget_fail_closed`` is on — the safety floor under
autonomous spend.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm.client import SommLLM, _warned_budget_near_cap, enforce_workload_budget
from somm.errors import SommBudgetExceeded
from somm.providers.base import ProviderHealth, SommResponse
from somm_core import Outcome
from somm_core.config import Config
from somm_core.pricing import write_intel


class CountingProvider:
    """FakeProvider that records how many times it was actually dispatched."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return SommResponse(
            text="ok",
            model=request.model or "fake-model",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            raw=None,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path, *, fail_closed: bool) -> Config:
    cfg = Config()
    cfg.project = "bg"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.budget_fail_closed = fail_closed
    return cfg


def test_blocks_when_cap_already_reached(tmp_path):
    """cap=0 with fail_closed on refuses immediately, before any provider call."""
    cfg = _cfg(tmp_path, fail_closed=True)
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        llm.repo.register_workload(
            name="capped", project=cfg.project, budget_cap_usd_daily=0.0
        )
        with pytest.raises(SommBudgetExceeded) as ei:
            llm.generate("p", workload="capped", provider="fake", model="m")
        assert ei.value.code == "SOMM_BUDGET_EXCEEDED"
        assert ei.value.workload == "capped"
        assert prov.calls == 0  # blocked before dispatch — no spend
    finally:
        llm.close()


def test_blocks_after_accumulated_spend(tmp_path):
    """The crossing call completes; the next is refused (bounded overshoot)."""
    cfg = _cfg(tmp_path, fail_closed=True)
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        # Price the fake (provider, model) so one call costs ~$2, over the cap.
        write_intel(llm.repo, "fake", "m", 1_000_000.0, 1_000_000.0, None, None, "test")
        llm.repo.register_workload(
            name="capped", project=cfg.project, budget_cap_usd_daily=0.50
        )
        # First call: spend starts at 0 → allowed, accumulates over the cap.
        r1 = llm.generate("p", workload="capped", provider="fake", model="m")
        assert r1.outcome == Outcome.OK and prov.calls == 1
        assert r1.cost_usd > 0.50
        # Commit the telemetry row. In production the async writer drains during
        # the seconds-long real LLM call; here the fake provider is instant.
        llm._writer.flush()
        # Second call: committed spend >= cap → refused before dispatch.
        with pytest.raises(SommBudgetExceeded):
            llm.generate("p", workload="capped", provider="fake", model="m")
        assert prov.calls == 1  # provider was not called the second time
    finally:
        llm.close()


def test_inert_when_disabled(tmp_path):
    """budget_fail_closed=False → the gate is a no-op even at/over the cap."""
    cfg = _cfg(tmp_path, fail_closed=False)
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        llm.repo.register_workload(
            name="capped", project=cfg.project, budget_cap_usd_daily=0.0
        )
        r = llm.generate("p", workload="capped", provider="fake", model="m")
        assert r.outcome == Outcome.OK and prov.calls == 1
    finally:
        llm.close()


def test_inert_when_no_cap(tmp_path):
    """fail_closed on but the workload has no cap → no-op (most workloads)."""
    cfg = _cfg(tmp_path, fail_closed=True)
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        # observe-mode auto-registers "uncapped" with budget_cap_usd_daily=None
        r = llm.generate("p", workload="uncapped", provider="fake", model="m")
        assert r.outcome == Outcome.OK and prov.calls == 1
    finally:
        llm.close()


def test_default_cap_applies_when_workload_uncapped(tmp_path):
    """A config-level default cap covers workloads with no explicit cap."""
    cfg = _cfg(tmp_path, fail_closed=True)
    cfg.budget_default_cap_usd_daily = 0.0  # default ceiling → blocks at 0
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        with pytest.raises(SommBudgetExceeded):
            llm.generate("p", workload="uncapped", provider="fake", model="m")
        assert prov.calls == 0
    finally:
        llm.close()


def test_explicit_cap_beats_default(tmp_path):
    """An explicit per-workload cap overrides the config default."""
    cfg = _cfg(tmp_path, fail_closed=True)
    cfg.budget_default_cap_usd_daily = 0.0  # default would block everything
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        llm.repo.register_workload(
            name="vip", project=cfg.project, budget_cap_usd_daily=1000.0
        )
        r = llm.generate("p", workload="vip", provider="fake", model="m")
        assert r.outcome == Outcome.OK and prov.calls == 1
    finally:
        llm.close()


def test_stream_blocks_when_cap_already_reached(tmp_path):
    """stream() gate fires before first yield — provider never dispatched."""
    cfg = _cfg(tmp_path, fail_closed=True)
    prov = CountingProvider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        llm.repo.register_workload(
            name="capped", project=cfg.project, budget_cap_usd_daily=0.0
        )
        with pytest.raises(SommBudgetExceeded) as ei:
            list(llm.stream("p", workload="capped", provider="fake", model="m"))
        assert ei.value.code == "SOMM_BUDGET_EXCEEDED"
        assert ei.value.workload == "capped"
        assert prov.calls == 0  # gate fires pre-dispatch, provider never reached
    finally:
        llm.close()


def _seed_spend(repo, workload_id: str, project: str, cost_usd: float) -> None:
    """Insert a synthetic spend row dated today so enforce_workload_budget sees it."""
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO calls (id, ts, project, workload_id, provider, model, "
            "tokens_in, tokens_out, latency_ms, cost_usd, outcome, prompt_hash, response_hash) "
            "VALUES (?, datetime('now'), ?, ?, 'fake', 'fake-model', 1, 1, 1, ?, 'ok', 'ph', 'rh')",
            (str(uuid.uuid4()), project, workload_id, cost_usd),
        )


def test_near_cap_warning_fires_at_85_pct(tmp_path, capsys):
    """Warning prints to stderr at 85% spend; no SommBudgetExceeded raised."""
    _warned_budget_near_cap.clear()
    cfg = _cfg(tmp_path, fail_closed=True)
    llm = SommLLM(config=cfg, providers=[])
    try:
        cap = 1.00
        workload = llm.repo.register_workload(
            name="nearwarn", project=cfg.project, budget_cap_usd_daily=cap
        )
        _seed_spend(llm.repo, workload.id, cfg.project, 0.85)
        # Must not raise — we're under the hard gate.
        enforce_workload_budget(
            llm.repo, workload, fail_closed=True, default_cap_usd_daily=None
        )
        err = capsys.readouterr().err
        assert "[somm] WARNING" in err
        assert "nearwarn" in err
        assert "Hard gate fires at 100%" in err
    finally:
        llm.close()


def test_near_cap_warning_suppressed_below_threshold(tmp_path, capsys):
    """No warning at 79% spend — below the 80% threshold."""
    _warned_budget_near_cap.clear()
    cfg = _cfg(tmp_path, fail_closed=True)
    llm = SommLLM(config=cfg, providers=[])
    try:
        cap = 1.00
        workload = llm.repo.register_workload(
            name="safe", project=cfg.project, budget_cap_usd_daily=cap
        )
        _seed_spend(llm.repo, workload.id, cfg.project, 0.79)
        enforce_workload_budget(
            llm.repo, workload, fail_closed=True, default_cap_usd_daily=None
        )
        err = capsys.readouterr().err
        assert "[somm] WARNING" not in err
    finally:
        llm.close()

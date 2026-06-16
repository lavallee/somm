"""Tests for somm self-healing: adaptive_param_bump.

Covers the batch auto-heal path — the agent worker detecting a recurring
`stripped_empty` capability failure for a (workload, provider, model) and
writing a learned_param_override that raises the max_tokens floor.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository
from somm_service.workers.agent import AgentWorker


def _tmp_setup(tmp_path: Path):
    cfg = Config()
    cfg.project = "selfheal"
    cfg.db_dir = tmp_path / ".somm"
    return cfg, Repository(cfg.db_path)


def _empty_call(repo, wl_id, project, provider, model, *, max_tokens=4096, stripped=True):
    """A reasoning-model overflow: HTTP 200 but empty after the think-block is
    stripped. tokens_out high (budget spent on thinking), outcome EMPTY."""
    detail = (
        f"EmptyResponse: {'stripped_empty' if stripped else 'no_content'} "
        f"| out_tokens=12288 | provider={provider} | model={model}"
    )
    repo.write_call(
        Call(
            id=str(uuid.uuid4()),
            ts=datetime.now(UTC),
            project=project,
            workload_id=wl_id,
            prompt_id=None,
            provider=provider,
            model=model,
            tokens_in=2000,
            tokens_out=12288,
            latency_ms=80000,
            cost_usd=0.0,
            outcome=Outcome.EMPTY,
            error_kind="EmptyResponse",
            prompt_hash="p",
            response_hash="r",
            error_detail=detail,
            max_tokens=max_tokens,
        )
    )


# ---------------------------------------------------------------------------
# detection


def test_detect_capability_overflow_flags_stripped_empty(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="prioritize", project=cfg.project)
    for _ in range(10):
        _empty_call(repo, wl.id, cfg.project, "minimax", "MiniMax-M2.7")

    cands = repo.detect_capability_overflow(min_calls=5)
    assert len(cands) == 1
    c = cands[0]
    assert (c["provider"], c["model"]) == ("minimax", "MiniMax-M2.7")
    assert c["empty_rate"] == 1.0
    # base requested 4096 → floor = max(4096*2, 8192) = 8192
    assert c["recommended_max_tokens_floor"] == 8192


def test_detect_ignores_no_content_and_low_volume(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w", project=cfg.project)
    # no_content (not stripped_empty) shouldn't match the thinking-overflow signature
    for _ in range(10):
        _empty_call(repo, wl.id, cfg.project, "minimax", "MiniMax-M2.7", stripped=False)
    assert repo.detect_capability_overflow(min_calls=5) == []
    # too few stripped_empty calls
    for _ in range(2):
        _empty_call(repo, wl.id, cfg.project, "minimax", "MiniMax-M2.7")
    assert repo.detect_capability_overflow(min_calls=5) == []


# ---------------------------------------------------------------------------
# auto-apply via the agent worker


def test_agent_auto_applies_and_records(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="prioritize", project=cfg.project)
    for _ in range(10):
        _empty_call(repo, wl.id, cfg.project, "minimax", "MiniMax-M2.7")

    summary = AgentWorker(repo, min_calls_for_consideration=5).run_once()
    assert summary["auto_applied"] >= 1
    assert summary["by_action"].get("adaptive_param_bump", 0) >= 1

    # the heal is written and readable on the hot path
    ov = repo.lookup_learned_override(wl.id, "MiniMax-M2.7", "minimax")
    assert ov is not None and ov["max_tokens_floor"] == 8192

    # recorded as an already-applied recommendation (not an open suggestion)
    with repo._open() as conn:
        row = conn.execute(
            "SELECT applied_at FROM recommendations WHERE action = 'adaptive_param_bump'"
        ).fetchone()
    assert row is not None and row[0] is not None


def test_adaptive_bump_dedups_across_runs(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="prioritize", project=cfg.project)
    for _ in range(10):
        _empty_call(repo, wl.id, cfg.project, "minimax", "MiniMax-M2.7")

    worker = AgentWorker(repo, min_calls_for_consideration=5)
    worker.run_once()
    second = worker.run_once()  # override already at floor → no re-apply
    assert second["auto_applied"] == 0

    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM learned_param_overrides").fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# override store round-trip


def test_lookup_override_round_trip(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w", project=cfg.project)
    repo.upsert_learned_override(
        workload_id=wl.id, provider="minimax", model="MiniMax-M2.7",
        max_tokens_floor=8192, failure_signature="capability_empty:stripped_empty",
        evidence={"empty_rate": 1.0}, confidence=0.8,
    )
    # provider-specific and provider-agnostic lookups both resolve
    assert repo.lookup_learned_override(wl.id, "MiniMax-M2.7", "minimax")["max_tokens_floor"] == 8192
    assert repo.lookup_learned_override(wl.id, "MiniMax-M2.7")["max_tokens_floor"] == 8192
    # unknown model → None
    assert repo.lookup_learned_override(wl.id, "other-model") is None

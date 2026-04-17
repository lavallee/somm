"""Tests for AgentWorker — switch_model, new_model_landed, chronic_cooldown."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.pricing import write_intel
from somm_core.repository import Repository
from somm_service.workers.agent import AgentWorker, _impact_str


def _tmp_setup(tmp_path: Path):
    cfg = Config()
    cfg.project = "agent"
    cfg.db_dir = tmp_path / ".somm"
    repo = Repository(cfg.db_path)
    return cfg, repo


def _write_call(
    repo: Repository,
    workload_id: str,
    project: str,
    provider: str,
    model: str,
    latency_ms: int = 100,
    cost_usd: float = 0.0,
    ts: datetime | None = None,
) -> str:
    call_id = str(uuid.uuid4())
    ts = ts or datetime.now(UTC)
    repo.write_call(
        Call(
            id=call_id,
            ts=ts,
            project=project,
            workload_id=workload_id,
            prompt_id=None,
            provider=provider,
            model=model,
            tokens_in=100,
            tokens_out=50,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="x",
            response_hash="y",
        )
    )
    return call_id


def _write_eval(repo: Repository, call_id: str, score: float, gold_model: str = "gold-m") -> None:
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO eval_results (call_id, gold_model, structural_score, "
            "embedding_score, judge_score) VALUES (?, ?, ?, ?, ?)",
            (call_id, gold_model, score, score, None),
        )


# ---------------------------------------------------------------------------


def test_impact_str_formatting():
    current = {"score": 0.5, "cost_usd": 0.001, "latency_ms": 100}
    best = {"score": 0.7, "cost_usd": 0.0004, "latency_ms": 50}
    s = _impact_str(current, best)
    assert "+20% quality" in s
    assert "cost" in s
    assert "latency" in s


# ---------------------------------------------------------------------------
# switch_model


def test_switch_model_recommended_when_shadow_shows_better_cheaper(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="extract", project=cfg.project)

    # Current production model: ollama/slow-m (lots of calls + low score)
    for _ in range(20):
        cid = _write_call(
            repo, wl.id, cfg.project, "ollama", "slow-m", latency_ms=500, cost_usd=0.0
        )
        _write_eval(repo, cid, score=0.4)
    # Candidate: ollama/fast-m (fewer calls, but high score + low cost)
    for _ in range(8):
        cid = _write_call(
            repo, wl.id, cfg.project, "ollama", "fast-m", latency_ms=100, cost_usd=0.0
        )
        _write_eval(repo, cid, score=0.85)

    worker = AgentWorker(repo, min_evals_for_switch=5, quality_threshold=0.1)
    summary = worker.run_once()
    assert summary["by_action"].get("switch_model", 0) >= 1

    with repo._open() as conn:
        recs = conn.execute(
            "SELECT action, evidence_json, expected_impact FROM recommendations"
        ).fetchall()
    actions = [r[0] for r in recs]
    assert "switch_model" in actions

    import json

    evidence = next(json.loads(r[1]) for r in recs if r[0] == "switch_model")
    assert evidence["candidate"]["model"] == "fast-m"
    assert evidence["current"]["model"] == "slow-m"


def test_switch_model_not_recommended_when_candidate_more_expensive(tmp_path):
    """Better quality but 5x more expensive — no switch unless also faster."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="extract", project=cfg.project)

    for _ in range(20):
        cid = _write_call(
            repo, wl.id, cfg.project, "ollama", "cheap", latency_ms=100, cost_usd=0.001
        )
        _write_eval(repo, cid, score=0.6)
    for _ in range(8):
        cid = _write_call(
            repo,
            wl.id,
            cfg.project,
            "anthropic",
            "opus",
            latency_ms=95,  # slightly faster, not 20% faster
            cost_usd=0.05,
        )
        _write_eval(repo, cid, score=0.9)

    worker = AgentWorker(repo, min_evals_for_switch=5)
    summary = worker.run_once()
    # Not recommended — more expensive AND not materially faster
    assert summary["by_action"].get("switch_model", 0) == 0


def test_switch_model_dedup_against_existing_open_recs(tmp_path):
    """Two runs shouldn't produce duplicate recs."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w", project=cfg.project)

    for _ in range(20):
        cid = _write_call(repo, wl.id, cfg.project, "ollama", "slow")
        _write_eval(repo, cid, score=0.4)
    for _ in range(8):
        cid = _write_call(repo, wl.id, cfg.project, "ollama", "fast")
        _write_eval(repo, cid, score=0.85)

    worker = AgentWorker(repo, min_evals_for_switch=5)
    worker.run_once()
    worker.run_once()  # second run

    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# new_model_landed


def test_new_model_landed_recommended(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_nml", project=cfg.project)

    # Seed: openai/gpt-4o is the current prod model
    write_intel(repo, "openai", "gpt-4o", 2.5, 10.0, 128_000, None, "static")
    # Seed: a cheaper openai model landed
    write_intel(repo, "openai", "gpt-4o-mini", 0.15, 0.6, 128_000, None, "static")

    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "openai", "gpt-4o", latency_ms=300, cost_usd=0.01)

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    summary = worker.run_once()
    assert summary["by_action"].get("new_model_landed", 0) >= 1


def test_new_model_landed_skips_free_local_models(tmp_path):
    """Don't recommend a "new cheaper" when current is already free."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_local", project=cfg.project)

    write_intel(repo, "ollama", "gemma4:e4b", 0.0, 0.0, None, None, "ollama-local")
    write_intel(repo, "ollama", "qwen2.5:7b", 0.0, 0.0, None, None, "ollama-local")

    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "ollama", "gemma4:e4b", latency_ms=300, cost_usd=0.0)

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    summary = worker.run_once()
    assert summary["by_action"].get("new_model_landed", 0) == 0


# ---------------------------------------------------------------------------
# chronic_cooldown


def test_chronic_cooldown_flagged(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_cool", project=cfg.project)

    # Simulate a chronically-failing provider
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO provider_health (provider, model, consecutive_failures) VALUES (?, '', 8)",
            ("openrouter",),
        )

    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "openrouter", "gemma-3-27b:free")

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    summary = worker.run_once()
    assert summary["by_action"].get("chronic_cooldown", 0) >= 1


def test_no_recs_on_empty_repo(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    worker = AgentWorker(repo)
    summary = worker.run_once()
    assert summary["written"] == 0

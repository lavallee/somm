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
    outcome: Outcome = Outcome.OK,
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
            outcome=outcome,
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


def _landed_days_ago(repo: Repository, provider: str, model: str, days: int) -> None:
    """Make a model_intel row look like it has been in the catalog for a while."""
    with repo._open() as conn:
        conn.execute(
            "UPDATE model_intel SET first_seen = datetime('now', ?) WHERE provider = ? AND model = ?",
            (f"-{days} day", provider, model),
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

    # Seed: openai/gpt-4o is the current prod model, in the catalog for a month
    write_intel(repo, "openai", "gpt-4o", 2.5, 10.0, 128_000, None, "static")
    _landed_days_ago(repo, "openai", "gpt-4o", 30)
    # Seed: a cheaper openai model landed today
    write_intel(repo, "openai", "gpt-4o-mini", 0.15, 0.6, 128_000, None, "static")

    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "openai", "gpt-4o", latency_ms=300, cost_usd=0.01)

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    summary = worker.run_once()
    assert summary["by_action"].get("new_model_landed", 0) >= 1


def test_new_model_landed_ignores_models_that_predate_the_current_one(tmp_path):
    """A cheaper sibling that was already in the catalog when the current model
    arrived did not "land" — it was passed over. Regression: the rule read
    last_seen, which every intel pass refreshes, and proposed older generations
    of the same family once per workload, forever."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_old", project=cfg.project)

    write_intel(repo, "minimax", "MiniMax-M3", 0.3, 1.2, 200_000, None, "models_dev")
    write_intel(repo, "minimax", "MiniMax-M2", 0.0, 0.0, 200_000, None, "stale_seed")
    # Both have been in the catalog equally long (the migration backfill case).
    _landed_days_ago(repo, "minimax", "MiniMax-M3", 3)
    _landed_days_ago(repo, "minimax", "MiniMax-M2", 3)

    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "minimax", "MiniMax-M3", cost_usd=0.01)

    summary = AgentWorker(repo, min_calls_for_consideration=10).run_once()
    assert summary["by_action"].get("new_model_landed", 0) == 0


def test_new_model_landed_is_proposed_once_even_after_dismissal(tmp_path):
    """A landing is an event. Dismissing the recommendation is the answer to
    it; the next weekly pass must not ask again."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_once", project=cfg.project)
    write_intel(repo, "openai", "gpt-4o", 2.5, 10.0, 128_000, None, "static")
    _landed_days_ago(repo, "openai", "gpt-4o", 30)
    write_intel(repo, "openai", "gpt-4o-mini", 0.15, 0.6, 128_000, None, "static")
    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "openai", "gpt-4o", cost_usd=0.01)

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    assert worker.run_once()["by_action"].get("new_model_landed", 0) == 1
    with repo._open() as conn:
        conn.execute("UPDATE recommendations SET dismissed_at = CURRENT_TIMESTAMP")
    assert worker.run_once()["by_action"].get("new_model_landed", 0) == 0


def test_new_model_landed_skips_free_local_models(tmp_path):
    """Don't recommend a "new cheaper" when current is already free."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_local", project=cfg.project)

    write_intel(repo, "ollama", "qwen3:8b", 0.0, 0.0, None, None, "ollama-local")
    write_intel(repo, "ollama", "qwen2.5:7b", 0.0, 0.0, None, None, "ollama-local")

    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "ollama", "qwen3:8b", latency_ms=300, cost_usd=0.0)

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    summary = worker.run_once()
    assert summary["by_action"].get("new_model_landed", 0) == 0


def test_new_model_landed_skips_unknown_pricing_sentinel(tmp_path):
    """Rows with NULL or negative pricing (OpenRouter's '-1' dynamic-pricing
    sentinel on meta-models like openrouter/auto) must not rank as cheapest.
    Regression: the agent used to pick these as "~6666767% lower cost"."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_sentinel", project=cfg.project)

    # Current prod model — Opus 4.7 via OpenRouter, in the catalog for a month
    write_intel(repo, "openrouter", "anthropic/claude-opus-4.7", 5.0, 25.0, 200_000, None, "openrouter")
    _landed_days_ago(repo, "openrouter", "anthropic/claude-opus-4.7", 30)
    # Sentinel meta-model: pricing unknown (NULL after the bugfix; would
    # have been -1M before). Must NOT be picked.
    write_intel(repo, "openrouter", "openrouter/auto", None, None, 2_000_000, None, "openrouter")
    # A genuine cheaper model — should be picked instead.
    write_intel(repo, "openrouter", "anthropic/claude-haiku-4.5", 1.0, 5.0, 200_000, None, "openrouter")

    for _ in range(15):
        _write_call(
            repo, wl.id, cfg.project,
            "openrouter", "anthropic/claude-opus-4.7", latency_ms=300, cost_usd=0.01,
        )

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    worker.run_once()
    # Should emit exactly one rec — picking Haiku, not openrouter/auto.
    [r for r in worker.last_run_recs if r.action == "new_model_landed"] \
        if hasattr(worker, "last_run_recs") else []
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT evidence_json FROM recommendations WHERE action = 'new_model_landed' "
            "AND workload_id = ? ORDER BY created_at DESC",
            (wl.id,),
        ).fetchall()
    assert rows, "expected a new_model_landed recommendation"
    import json
    ev = json.loads(rows[0][0])
    assert ev["candidate"]["model"] == "anthropic/claude-haiku-4.5", (
        f"agent picked {ev['candidate']['model']} — must not be openrouter/auto"
    )


def test_parse_price_per_token_handles_sentinels():
    """The OpenRouter price parser must return None for -1/negative/malformed
    values so they propagate as NULL, not as literal cheap prices."""
    from somm_service.workers.model_intel import _parse_price_per_token

    assert _parse_price_per_token("0.0000006") == 0.0000006
    assert _parse_price_per_token("0") == 0.0
    assert _parse_price_per_token("") is None
    assert _parse_price_per_token(None) is None
    assert _parse_price_per_token("-1") is None    # OpenRouter dynamic-pricing sentinel
    assert _parse_price_per_token("-0.0001") is None
    assert _parse_price_per_token("abc") is None


# ---------------------------------------------------------------------------
# chronic_cooldown


def test_chronic_cooldown_flagged(tmp_path):
    """Three of fifteen calls to the provider failed to deliver: 20%, chronic."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_cool", project=cfg.project)

    for _ in range(12):
        _write_call(repo, wl.id, cfg.project, "openrouter", "gemma-4-31b-it:free")
    for outcome in (Outcome.UPSTREAM_ERROR, Outcome.EXHAUSTED, Outcome.TIMEOUT):
        _write_call(repo, wl.id, cfg.project, "openrouter", "*roster_exhausted", outcome=outcome)

    worker = AgentWorker(repo, min_calls_for_consideration=10)
    summary = worker.run_once()
    assert summary["by_action"].get("chronic_cooldown", 0) == 1
    with repo._open() as conn:
        (evidence,) = conn.execute(
            "SELECT evidence_json FROM recommendations WHERE action = 'chronic_cooldown'"
        ).fetchone()
    import json

    ev = json.loads(evidence)
    assert ev["provider"] == "openrouter"
    assert (ev["n_calls"], ev["n_failed"], ev["failure_rate"]) == (15, 3, 0.2)


def test_chronic_cooldown_reads_the_calls_not_stale_health_rows(tmp_path):
    """A provider_health entry with a long failure streak is history, not a
    verdict on a workload the provider is serving fine today. Regression: the
    rule used to indict every workload that had ever used the provider."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="w_fine", project=cfg.project)
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO provider_health (provider, model, consecutive_failures) "
            "VALUES ('openrouter', 'openrouter/elephant-alpha', 68166)"
        )
    for _ in range(15):
        _write_call(repo, wl.id, cfg.project, "openrouter", "gemma-4-31b-it:free")

    summary = AgentWorker(repo, min_calls_for_consideration=10).run_once()
    assert summary["by_action"].get("chronic_cooldown", 0) == 0


def test_no_recs_on_empty_repo(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    worker = AgentWorker(repo)
    summary = worker.run_once()
    assert summary["written"] == 0

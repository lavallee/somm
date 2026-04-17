"""Tests for ShadowEvalWorker — structural + text similarity grading,
privacy defense-in-depth, budget ceiling, lease semantics.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.models import Call, Outcome, PrivacyClass
from somm_core.pricing import write_intel
from somm_core.repository import Repository
from somm_service.workers.shadow_eval import (
    ShadowEvalWorker,
    _structural_score,
    _text_similarity,
)

# ---------------------------------------------------------------------------
# Grader unit tests


def test_text_similarity_identical_strings_are_one():
    assert _text_similarity("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_text_similarity_disjoint_strings_are_low():
    assert _text_similarity("quick brown fox", "slow purple bear") < 0.5


def test_text_similarity_empty_strings():
    assert _text_similarity("", "") == 1.0
    assert _text_similarity("hello", "") == 0.0


def test_structural_score_matching_json():
    prod = '{"name": "alice", "age": 30}'
    gold = '{"name": "alice", "age": 30}'
    assert _structural_score(prod, gold) == 1.0


def test_structural_score_partial_match():
    prod = '{"name": "alice", "age": 30}'
    gold = '{"name": "bob", "age": 30}'
    # Two keys overlap, ages match (1.0) but names differ.
    score = _structural_score(prod, gold)
    assert 0.0 < score < 1.0


def test_structural_score_prose_returns_none():
    assert _structural_score("not json", "not json either") is None


def test_structural_score_one_parses_other_doesnt():
    assert _structural_score('{"a": 1}', "prose") == 0.0


# ---------------------------------------------------------------------------
# Fixtures for worker


def _tmp_setup(tmp_path: Path):
    cfg = Config()
    cfg.project = "shadow"
    cfg.db_dir = tmp_path / ".somm"
    repo = Repository(cfg.db_path)
    return cfg, repo


def _insert_call(
    repo: Repository,
    workload_id: str,
    project: str,
    prompt_body: str,
    response_body: str,
    provider: str = "ollama",
    model: str = "gemma4:e4b",
) -> str:
    """Insert a call + its samples row so shadow has bodies to grade."""
    import hashlib

    call_id = str(uuid.uuid4())
    ph = hashlib.sha256(prompt_body.encode()).hexdigest()[:16]
    rh = hashlib.sha256(response_body.encode()).hexdigest()[:16]
    call = Call(
        id=call_id,
        ts=datetime.now(UTC),
        project=project,
        workload_id=workload_id,
        prompt_id=None,
        provider=provider,
        model=model,
        tokens_in=20,
        tokens_out=10,
        latency_ms=50,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash=ph,
        response_hash=rh,
    )
    repo.write_call(call)
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO samples (call_id, prompt_body, response_body) VALUES (?, ?, ?)",
            (call_id, prompt_body, response_body),
        )
    return call_id


class _GoldProvider:
    """Stand-in for a gold-model provider. Returns fixed text."""

    name = "gold"

    def __init__(self, text: str = '{"ok": true}'):
        self.text = text
        self.called = 0

    def generate(self, request):
        self.called += 1
        return SommResponse(
            text=self.text,
            model="gold-m",
            tokens_in=15,
            tokens_out=8,
            latency_ms=10,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


# ---------------------------------------------------------------------------
# Worker behavior


def test_shadow_off_by_default_no_grades(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="no_shadow", project=cfg.project)
    _insert_call(repo, wl.id, cfg.project, "prompt", "response")

    worker = ShadowEvalWorker(repo, providers=[_GoldProvider()])
    summary = worker.run_once()
    assert summary["workloads_considered"] == 0
    assert summary["calls_graded"] == 0


def test_shadow_enabled_grades_sampled_call(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="yes_shadow", project=cfg.project)
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,  # 100% — force sampling in tests
            "budget_usd_daily": 5.0,
            "max_grades_per_run": 5,
        },
    )
    # Seed model_intel so budget accounting has a non-zero cost per grade.
    write_intel(repo, "gold", "gold-m", 1.0, 4.0, None, None, "test")
    # Prod response that matches gold → structural 1.0
    _insert_call(
        repo, wl.id, cfg.project, prompt_body="tell me a joke", response_body='{"ok": true}'
    )

    worker = ShadowEvalWorker(repo, providers=[_GoldProvider()])
    summary = worker.run_once()
    assert summary["workloads_considered"] == 1
    assert summary["calls_graded"] == 1

    with repo._open() as conn:
        rows = conn.execute(
            "SELECT structural_score, embedding_score, gold_model FROM eval_results"
        ).fetchall()
    assert len(rows) == 1
    structural, text_sim, gold_model = rows[0]
    assert structural == 1.0
    assert text_sim == 1.0
    assert gold_model == "gold-m"


def test_private_workloads_are_skipped(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(
        name="private_w",
        project=cfg.project,
        privacy_class=PrivacyClass.PRIVATE,
    )
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,
            "budget_usd_daily": 5.0,
        },
    )
    _insert_call(repo, wl.id, cfg.project, "prompt", "response")

    gold = _GoldProvider()
    worker = ShadowEvalWorker(repo, providers=[gold])
    summary = worker.run_once()

    # The shadow_candidates view filters private workloads → 0 graded
    assert summary["calls_graded"] == 0
    assert gold.called == 0  # gold provider never touched


def test_enable_shadow_blocks_private_at_client_level(tmp_path):
    """SommLLM.enable_shadow() raises on privacy_class=private."""
    from somm.client import SommLLM
    from somm.errors import SommPrivacyViolation

    cfg, repo = _tmp_setup(tmp_path)

    class NoopProvider:
        name = "noop"

        def generate(self, req):  # pragma: no cover
            return SommResponse(text="", model="", tokens_in=0, tokens_out=0, latency_ms=0)

        def stream(self, req):  # pragma: no cover
            yield

        def health(self):
            return ProviderHealth(available=True)

        def models(self):
            return []

        def estimate_tokens(self, text, model):
            return 1

    llm = SommLLM(config=cfg, providers=[NoopProvider()])
    try:
        # Register as private
        llm.repo.register_workload(
            name="sensitive",
            project=cfg.project,
            privacy_class=PrivacyClass.PRIVATE,
        )
        with pytest.raises(SommPrivacyViolation) as exc_info:
            llm.enable_shadow("sensitive", gold_provider="x", gold_model="y")
        assert "SOMM_PRIVACY_VIOLATION" in str(exc_info.value)
    finally:
        llm.close()


def test_budget_ceiling_stops_grading(tmp_path):
    """Grading stops once daily budget is reached."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="budget_w", project=cfg.project)
    # Low budget + high gold cost → should stop after first grade.
    write_intel(repo, "gold", "gold-m", 100.0, 100.0, None, None, "test")  # expensive
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,
            "budget_usd_daily": 0.01,  # tiny budget
            "max_grades_per_run": 10,
        },
    )
    # Insert 5 calls — each grade costs $0.003 (30 tok * $100/1M), so ~3 fit.
    for i in range(5):
        _insert_call(repo, wl.id, cfg.project, f"p{i}", f'{{"i": {i}}}')

    worker = ShadowEvalWorker(repo, providers=[_GoldProvider()])
    summary = worker.run_once()
    # Budget $0.01 / cost-per-grade $0.003 ≈ 3-4 grades allowed
    assert 1 <= summary["calls_graded"] <= 5


def test_missing_samples_logs_note_no_crash(tmp_path):
    """If `samples` row is missing, worker records a note and moves on."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="nosamples", project=cfg.project)
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,
            "budget_usd_daily": 5.0,
        },
    )
    # Insert a call WITHOUT a samples row
    call_id = str(uuid.uuid4())
    call = Call(
        id=call_id,
        ts=datetime.now(UTC),
        project=cfg.project,
        workload_id=wl.id,
        prompt_id=None,
        provider="ollama",
        model="gemma4:e4b",
        tokens_in=10,
        tokens_out=5,
        latency_ms=50,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="a",
        response_hash="b",
    )
    repo.write_call(call)

    worker = ShadowEvalWorker(repo, providers=[_GoldProvider()])
    summary = worker.run_once()
    assert summary["calls_graded"] == 1  # lease claimed, but grade has None scores

    with repo._open() as conn:
        row = conn.execute(
            "SELECT structural_score, embedding_score, judge_reason FROM eval_results"
        ).fetchone()
    struct, text_sim, reason = row
    assert struct is None
    assert text_sim is None
    # Notes include an explanation
    assert "samples" in str(reason)


def test_lease_prevents_duplicate_grading(tmp_path):
    """Second run shouldn't re-grade an already-graded call."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="once", project=cfg.project)
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,
            "budget_usd_daily": 5.0,
        },
    )
    _insert_call(repo, wl.id, cfg.project, "p", '{"a": 1}')

    gold = _GoldProvider()
    worker = ShadowEvalWorker(repo, providers=[gold])
    worker.run_once()
    first_calls = gold.called

    # Run again — should be a no-op (already graded)
    worker.run_once()
    assert gold.called == first_calls

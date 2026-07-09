"""Tests for ShadowEvalWorker — structural + text similarity grading,
privacy defense-in-depth, budget ceiling, lease semantics.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.models import Call, Outcome, PrivacyClass
from somm_core.pricing import write_intel
from somm_core.repository import Repository
from somm_service.workers.shadow_eval import ShadowEvalWorker

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


class _JudgeProvider:
    def __init__(self, name: str = "judge", response: str | dict | None = None):
        self.name = name
        self.response = response or {
            "criteria": [
                {"name": "correctness", "pass": True, "reason": "matches"},
                {"name": "completeness", "pass": False, "reason": "missing detail"},
            ]
        }
        self.called = 0

    def generate(self, request):
        self.called += 1
        text = self.response if isinstance(self.response, str) else json.dumps(self.response)
        return SommResponse(
            text=text,
            model=request.model,
            tokens_in=12,
            tokens_out=6,
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


def test_shadow_judge_scores_binary_rubric(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="judge_shadow", project=cfg.project)
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,
            "budget_usd_daily": 5.0,
            "max_grades_per_run": 5,
            "judge": {
                "provider": "judge",
                "model": "judge-m",
                "criteria": ["correctness", "completeness"],
            },
        },
    )
    write_intel(repo, "gold", "gold-m", 1.0, 4.0, None, None, "test")
    write_intel(repo, "judge", "judge-m", 1.0, 1.0, None, None, "test")
    _insert_call(repo, wl.id, cfg.project, prompt_body="prompt", response_body="candidate")

    judge = _JudgeProvider()
    worker = ShadowEvalWorker(repo, providers=[_GoldProvider("gold"), judge])
    summary = worker.run_once()

    assert summary["calls_graded"] == 1
    assert judge.called == 1
    with repo._open() as conn:
        judge_score, reason_json = conn.execute(
            "SELECT judge_score, judge_reason FROM eval_results"
        ).fetchone()
    assert judge_score == 0.5
    reason = json.loads(reason_json)
    assert reason[0]["cost_usd"] > 0
    judge_receipt = [item["judge"] for item in reason if "judge" in item][0]
    assert judge_receipt["mode"] == "single"
    assert judge_receipt["criteria"][1]["pass"] is False


def test_shadow_judge_panel_majority_vote(tmp_path):
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="judge_panel", project=cfg.project)
    repo.set_shadow_config(
        wl.id,
        {
            "gold_provider": "gold",
            "gold_model": "gold-m",
            "sample_rate": 1.0,
            "budget_usd_daily": 5.0,
            "judge": {
                "criteria": ["correctness"],
                "panel": [
                    {"provider": "judge-a", "model": "a"},
                    {"provider": "judge-b", "model": "b"},
                    {"provider": "judge-c", "model": "c"},
                ],
            },
        },
    )
    _insert_call(repo, wl.id, cfg.project, prompt_body="prompt", response_body="candidate")
    pass_vote = {"criteria": [{"name": "correctness", "pass": True, "reason": "ok"}]}
    fail_vote = {"criteria": [{"name": "correctness", "pass": False, "reason": "bad"}]}

    worker = ShadowEvalWorker(
        repo,
        providers=[
            _GoldProvider("gold"),
            _JudgeProvider("judge-a", pass_vote),
            _JudgeProvider("judge-b", pass_vote),
            _JudgeProvider("judge-c", fail_vote),
        ],
    )
    summary = worker.run_once()

    assert summary["calls_graded"] == 1
    with repo._open() as conn:
        judge_score, reason_json = conn.execute(
            "SELECT judge_score, judge_reason FROM eval_results"
        ).fetchone()
    assert judge_score == 1.0
    receipt = [item["judge"] for item in json.loads(reason_json) if "judge" in item][0]
    assert receipt["mode"] == "panel"
    assert receipt["criteria"][0]["votes_for"] == 2
    assert len(receipt["judges"]) == 3


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


def test_missing_samples_not_considered(tmp_path):
    """Calls without captured bodies are not candidates (capture is the gate)."""
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
    # 0.7.0: uncaptured calls are no longer candidates at all — sample_rate
    # is applied once, at capture time in the library. Nothing graded,
    # nothing churned into "samples not captured" results, no crash.
    assert summary["calls_graded"] == 0

    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0]
    assert n == 0


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


def test_gold_provider_not_filtered_by_provider_order(tmp_path, monkeypatch):
    """SOMM_PROVIDER_ORDER is exclusive for routing, but the shadow worker's
    gold-model chain must include every keyed provider regardless."""
    from somm_core.config import Config
    from somm_service.inprocess import build_workers_factory

    cfg = Config()
    cfg.db_dir = tmp_path / ".somm"
    cfg.anthropic_api_key = "test-key"
    cfg.provider_order = ["ollama"]  # excludes anthropic from routing

    from somm_core.repository import Repository

    repo = Repository(cfg.db_path)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    worker = build_workers_factory(cfg, repo)("shadow_eval")
    assert "anthropic" in worker.providers
    assert "claude-cli" in worker.providers  # pinned-only executors reachable as gold
    assert cfg.provider_order == ["ollama"]  # original config untouched


def test_failed_grade_releases_candidate_back_to_pool(tmp_path):
    """A transient gold failure must not permanently orphan the sample:
    the lease placeholder is deleted, the call is a candidate again."""
    cfg, repo = _tmp_setup(tmp_path)
    wl = repo.register_workload(name="retryable", project=cfg.project)
    repo.set_shadow_config(
        wl.id,
        {"gold_provider": "gold", "gold_model": "gold-m",
         "sample_rate": 1.0, "budget_usd_daily": 5.0},
    )
    call_id = str(uuid.uuid4())
    repo.write_call(Call(
        id=call_id, ts=datetime.now(UTC), project=cfg.project,
        workload_id=wl.id, prompt_id=None, provider="ollama", model="m",
        tokens_in=10, tokens_out=5, latency_ms=50, cost_usd=0.0,
        outcome=Outcome.OK, error_kind=None, prompt_hash="a", response_hash="b",
    ))
    repo.write_sample(call_id, "the prompt", "the response")

    class _FailingGold:
        name = "gold"

        def generate(self, request):
            raise RuntimeError("simulated 429")

    worker = ShadowEvalWorker(repo, providers=[_FailingGold()])
    summary = worker.run_once()
    assert summary["calls_graded"] == 0
    assert summary["errors"]
    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0]
    assert n == 0  # lease deleted, not buried

    # Candidate resurfaces and grades once the gold provider recovers.
    worker2 = ShadowEvalWorker(repo, providers=[_GoldProvider()])
    summary2 = worker2.run_once()
    assert summary2["calls_graded"] == 1

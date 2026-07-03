"""Tests for online-eval sample capture on the call path (0.7.0)."""

from __future__ import annotations

from pathlib import Path

from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="the answer", model="fake-m", tokens_in=3, tokens_out=2, latency_ms=1
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _llm(tmp_path: Path) -> SommLLM:
    cfg = Config()
    cfg.project = "capture_test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(providers=[FakeProvider()], config=cfg)


def _samples(llm) -> list[tuple]:
    with llm.repo._open() as conn:
        return conn.execute(
            "SELECT call_id, prompt_body, response_body FROM samples"
        ).fetchall()


def _enable_shadow(llm, workload: str, rate: float = 1.0):
    wl = llm.repo.register_workload(name=workload, project=llm.config.project)
    llm.repo.set_shadow_config(
        wl.id,
        {"gold_provider": "fake", "gold_model": "fake-m", "sample_rate": rate,
         "budget_usd_daily": 1.0},
    )
    llm._shadow_cfg_cache.clear()
    return wl


def test_shadow_workload_captures_bodies(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    result = llm.generate(prompt="what is 2+2?", workload="graded")
    rows = _samples(llm)
    assert len(rows) == 1
    assert rows[0][0] == result.call_id
    assert rows[0][1] == "what is 2+2?"
    assert rows[0][2] == "the answer"


def test_no_shadow_config_means_no_capture(tmp_path):
    llm = _llm(tmp_path)
    llm.generate(prompt="hi", workload="plain")
    assert _samples(llm) == []


def test_rate_zero_never_captures(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=0.0)
    llm.generate(prompt="hi", workload="graded")
    assert _samples(llm) == []


def test_private_workload_never_captured(tmp_path):
    from somm_core import PrivacyClass

    llm = _llm(tmp_path)
    wl = llm.repo.register_workload(
        name="secret", project=llm.config.project, privacy_class=PrivacyClass.PRIVATE
    )
    llm.repo.set_shadow_config(
        wl.id,
        {"gold_provider": "fake", "gold_model": "fake-m", "sample_rate": 1.0,
         "budget_usd_daily": 1.0},
    )
    llm._shadow_cfg_cache.clear()
    llm.generate(prompt="hi", workload="secret")
    assert _samples(llm) == []


def test_oversized_body_skipped(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    llm.generate(prompt="x" * 300_000, workload="graded")
    assert _samples(llm) == []


def test_messages_captured_as_json(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    llm.generate(
        prompt="ignored",
        messages=[{"role": "user", "content": "hello"}],
        workload="graded",
    )
    rows = _samples(llm)
    assert len(rows) == 1
    assert '"role": "user"' in rows[0][1]


def test_rate_partitions_deterministically(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=0.5)
    for _ in range(60):
        llm.generate(prompt="hi", workload="graded")
    n = len(_samples(llm))
    assert 10 < n < 50  # ~50% with slack; 0 or 60 would mean the gate is broken
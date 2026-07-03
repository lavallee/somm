"""Tests for SOMM_INPROCESS_WORKERS wiring + the dormant-loop warning."""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path

from somm import client as client_mod
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(text="ok", model="fake-m", tokens_in=1, tokens_out=1, latency_ms=1)

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_config(tmp_path: Path, **kw) -> Config:
    cfg = Config()
    cfg.project = "workers_test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_inprocess_workers_start_once_per_db(tmp_path, monkeypatch):
    started = []

    class StubScheduler:
        def stop(self):
            pass

    def stub_start(cfg, repo):
        started.append(str(cfg.db_path))
        return StubScheduler()

    import somm_service.inprocess as service_inprocess

    monkeypatch.setattr(service_inprocess, "start_inprocess_scheduler", stub_start)
    monkeypatch.setattr(client_mod, "_inprocess_schedulers", {})

    cfg = _tmp_config(tmp_path, inprocess_workers=True)
    llm1 = SommLLM(providers=[FakeProvider()], config=cfg)
    llm2 = SommLLM(providers=[FakeProvider()], config=cfg)
    assert len(started) == 1  # singleton per DB per process
    assert llm1._scheduler is not None
    assert llm2._scheduler is llm1._scheduler


def test_dormant_loop_warning_fires_once(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "_warned_dormant_loop", set())
    cfg = _tmp_config(tmp_path)

    llm = SommLLM(providers=[FakeProvider()], config=cfg)
    # Configure shadow on a workload but never run a worker.
    wl = llm.repo.register_workload(name="graded", project=cfg.project)
    with llm.repo._open() as conn:
        conn.execute(
            "UPDATE workloads SET shadow_config_json = '{}' WHERE id = ?", (wl.id,)
        )

    monkeypatch.setattr(client_mod, "_warned_dormant_loop", set())
    buf = io.StringIO()
    with redirect_stderr(buf):
        SommLLM(providers=[FakeProvider()], config=cfg)
        SommLLM(providers=[FakeProvider()], config=cfg)
    warnings = [ln for ln in buf.getvalue().splitlines() if "online eval is configured" in ln]
    assert len(warnings) == 1  # once per DB per process


def test_no_warning_without_shadow_config(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "_warned_dormant_loop", set())
    cfg = _tmp_config(tmp_path)
    buf = io.StringIO()
    with redirect_stderr(buf):
        SommLLM(providers=[FakeProvider()], config=cfg)
    assert "online eval is configured" not in buf.getvalue()

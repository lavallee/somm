from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.pricing import write_intel


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="ok",
            model=request.model or "m",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path, project: str) -> Config:
    cfg = Config()
    cfg.project = project
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _clear_registry_guards() -> None:
    import somm.client as client
    import somm_core.registry as registry

    client._registered_project_keys.clear()
    registry._registered_projects.clear()


def test_sommllm_registers_project_once_per_process(tmp_path, monkeypatch):
    _clear_registry_guards()
    monkeypatch.setenv("SOMM_REGISTRY_ALLOW_TMP", "1")
    original_replace = Path.replace
    registry_rewrites = 0

    def counting_replace(self: Path, target: Path) -> Path:
        nonlocal registry_rewrites
        if self.name == "registry.json.tmp":
            registry_rewrites += 1
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", counting_replace)
    cfg = _cfg(tmp_path, "registry-throttle")

    llm1 = SommLLM(config=cfg, providers=[FakeProvider()])
    llm1.close()
    llm1.repo.close()

    llm2 = SommLLM(config=cfg, providers=[FakeProvider()])
    llm2.close()
    llm2.repo.close()

    assert registry_rewrites == 1


def test_generate_on_warmed_llm_opens_no_calling_thread_connections(tmp_path, monkeypatch):
    _clear_registry_guards()
    cfg = _cfg(tmp_path, "connect-budget")
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.repo.register_workload(name="hot", project=cfg.project)
        write_intel(llm.repo, "fake", "m", 1.0, 1.0, None, None, "test")
        llm.repo._open()

        main_thread = threading.get_ident()
        real_connect = sqlite3.connect
        calling_thread_connects = 0

        def counting_connect(*args, **kwargs):
            nonlocal calling_thread_connects
            if threading.get_ident() == main_thread:
                calling_thread_connects += 1
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", counting_connect)

        result = llm.generate("prompt", workload="hot", provider="fake", model="m")
        assert result.text == "ok"
        assert calling_thread_connects == 0
    finally:
        llm.close()
        llm.repo.close()

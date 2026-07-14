"""Tests for somm.hooks — correlation-id provider + call observers."""

from __future__ import annotations

from pathlib import Path

import pytest
from somm import hooks
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="ok",
            model=request.model or "fake-m",
            tokens_in=3,
            tokens_out=2,
            latency_ms=5,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "hooks_test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Hooks are process-global; reset around every test."""
    hooks.set_correlation_provider(None)
    saved = list(hooks._call_observers)
    hooks._call_observers.clear()
    yield
    hooks.set_correlation_provider(None)
    hooks._call_observers[:] = saved


def _make_llm(tmp_path: Path) -> SommLLM:
    return SommLLM(providers=[FakeProvider()], config=_tmp_config(tmp_path))


def test_correlation_id_lands_in_calls_row(tmp_path):
    hooks.set_correlation_provider(lambda: "corr-123")
    llm = _make_llm(tmp_path)
    llm.generate(prompt="hi", workload="w")
    llm._writer.flush()
    with llm.repo._open() as conn:
        row = conn.execute("SELECT correlation_id FROM calls").fetchone()
    assert row[0] == "corr-123"


def test_explicit_correlation_id_overrides_process_hook(tmp_path):
    hooks.set_correlation_provider(lambda: "ambient")
    events: list[dict] = []
    hooks.add_call_observer(events.append)
    llm = _make_llm(tmp_path)
    llm.generate(prompt="hi", workload="w", correlation_id="job-42")
    llm._writer.flush()
    with llm.repo._open() as conn:
        row = conn.execute("SELECT correlation_id FROM calls").fetchone()
    assert row[0] == "job-42"
    assert events[0]["correlation_id"] == "job-42"


def test_no_provider_means_null_correlation(tmp_path):
    llm = _make_llm(tmp_path)
    llm.generate(prompt="hi", workload="w")
    llm._writer.flush()
    with llm.repo._open() as conn:
        row = conn.execute("SELECT correlation_id FROM calls").fetchone()
    assert row[0] is None


def test_broken_correlation_provider_never_raises(tmp_path):
    def boom():
        raise RuntimeError("provider bug")

    hooks.set_correlation_provider(boom)
    llm = _make_llm(tmp_path)
    result = llm.generate(prompt="hi", workload="w")
    assert result.text == "ok"


def test_observer_receives_call_event(tmp_path):
    events: list[dict] = []
    hooks.add_call_observer(events.append)
    hooks.set_correlation_provider(lambda: "corr-9")
    llm = _make_llm(tmp_path)
    llm.generate(prompt="hi", workload="my_wl")
    assert len(events) == 1
    evt = events[0]
    assert evt["workload"] == "my_wl"
    assert evt["project"] == "hooks_test"
    assert evt["provider"] == "fake"
    assert evt["outcome"] == "ok"
    assert evt["correlation_id"] == "corr-9"
    assert evt["tokens_in"] == 3
    assert evt["tokens_out"] == 2
    assert evt["call_id"]


def test_broken_observer_never_breaks_call_or_other_observers(tmp_path):
    seen: list[dict] = []

    def bad(_evt):
        raise RuntimeError("observer bug")

    hooks.add_call_observer(bad)
    hooks.add_call_observer(seen.append)
    llm = _make_llm(tmp_path)
    result = llm.generate(prompt="hi", workload="w")
    assert result.text == "ok"
    assert len(seen) == 1


def test_remove_call_observer(tmp_path):
    events: list[dict] = []
    hooks.add_call_observer(events.append)
    hooks.remove_call_observer(events.append)
    llm = _make_llm(tmp_path)
    llm.generate(prompt="hi", workload="w")
    assert events == []

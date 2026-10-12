"""generate / stream / embed record through one path and agree with each other."""

from __future__ import annotations

from pathlib import Path

import pytest
from somm import hooks
from somm.client import SommLLM
from somm.errors import SommBudgetExceeded, SommUpstream5xx
from somm.providers.base import ProviderHealth, SommChunk, SommEmbedResponse, SommResponse
from somm_core.config import Config


class Fake:
    name = "fake"
    default_embed_model = "fake-embed"

    def __init__(self, *, stream_fail: bool = False) -> None:
        self.requests = []
        self.stream_fail = stream_fail

    def generate(self, request):
        self.requests.append(request)
        return SommResponse(text="ok", model="m", tokens_in=1, tokens_out=1, latency_ms=1, raw={})

    def stream(self, request):
        self.requests.append(request)
        yield SommChunk(text="partial", done=False)
        if self.stream_fail:
            raise SommUpstream5xx("fake 503 on m", cooldown_s=1.0, model="m")
        yield SommChunk(text="", done=True)

    def embed(self, request):
        return SommEmbedResponse(embedding=[0.1, 0.2], model="fake-embed", tokens_in=2, latency_ms=1)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path, **kw) -> Config:
    cfg = Config()
    cfg.project = "paths"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def _hook():
    registered = []

    def register(phase, fn):
        hooks.register_hook(phase, fn)
        registered.append((phase, fn))

    yield register
    for phase, fn in registered:
        hooks.unregister_hook(phase, fn)


def test_pre_call_hook_keeps_reasoning_effort(tmp_path, _hook):
    """The request rebuilt after pre_call used to drop reasoning_effort."""
    fake = Fake()
    _hook(hooks.PRE_CALL, lambda ctx: None)
    llm = SommLLM(config=_cfg(tmp_path), providers=[fake])
    try:
        llm.generate("hi", workload="w", provider="fake", reasoning_effort="high")
    finally:
        llm.close()
    assert fake.requests[-1].reasoning_effort == "high"


def test_stream_failure_reaches_on_error_with_detail(tmp_path):
    seen = []
    llm = SommLLM(config=_cfg(tmp_path), providers=[Fake(stream_fail=True)], on_error=seen.append)
    try:
        with pytest.raises(SommUpstream5xx):
            list(llm.stream("hi", workload="w", provider="fake"))
    finally:
        llm.close()
    assert len(seen) == 1
    assert seen[0]["outcome"] == "upstream_error"
    assert seen[0]["provider"] == "fake"
    assert "503" in (seen[0]["error_detail"] or "")


def test_embed_honors_the_fail_closed_budget_gate(tmp_path):
    cfg = _cfg(tmp_path, budget_fail_closed=True)
    fake = Fake()
    fake.name = "ollama"  # embed() is served by the ollama provider
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        llm.register_workload(name="capped", budget_cap_usd_daily=0.5)
        llm.generate("hi", workload="capped", provider="ollama")
        llm._writer.flush()
        with llm.repo._open() as conn:
            conn.execute("UPDATE calls SET cost_usd = 1.0")
        with pytest.raises(SommBudgetExceeded):
            llm.embed("hi", workload="capped")
    finally:
        llm.close()


def test_observer_event_carries_token_split(tmp_path, _hook):
    events = []
    _hook(hooks.POST_CALL, events.append)
    llm = SommLLM(config=_cfg(tmp_path), providers=[Fake()])
    try:
        llm.generate("hi", workload="w", provider="fake")
    finally:
        llm.close()
    assert {"cache_tokens_in", "cache_tokens_out", "reasoning_tokens", "ttft_ms"} <= set(events[-1])

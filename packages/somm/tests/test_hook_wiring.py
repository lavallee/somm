from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from threading import Event

import pytest
from somm import hooks
from somm.client import SommLLM
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommEmbedResponse,
    SommResponse,
)
from somm_core import Outcome
from somm_core.config import Config
from somm_core.parse import stable_hash


class FakeProvider:
    name = "fake"

    def __init__(self, *, text: str = "ok", fail: bool = False):
        self.text = text
        self.fail = fail
        self.generate_calls = 0
        self.stream_calls = 0
        self.last_request = None

    def generate(self, request):
        self.generate_calls += 1
        self.last_request = request
        if self.fail:
            raise AssertionError("provider should not be called")
        return SommResponse(
            text=self.text,
            model=request.model or "fake-model",
            tokens_in=7,
            tokens_out=5,
            latency_ms=11,
            raw={"prompt": request.prompt},
        )

    def stream(self, request):
        self.stream_calls += 1
        self.last_request = request
        if self.fail:
            raise AssertionError("provider should not be called")
        yield SommChunk(text=self.text, done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


class FakeOllamaProvider(FakeProvider):
    name = "ollama"
    default_embed_model = "nomic-embed-text"

    def __init__(self, *, embedding: list[float] | None = None, fail: bool = False):
        super().__init__(fail=fail)
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.embed_calls = 0

    def embed(self, request):
        self.embed_calls += 1
        self.last_request = request
        if self.fail:
            raise AssertionError("provider should not be called")
        return SommEmbedResponse(
            embedding=list(self.embedding),
            model=request.model or self.default_embed_model,
            tokens_in=max(1, len(request.text) // 4),
            latency_ms=3,
            raw={"embedding": self.embedding},
        )


@pytest.fixture(autouse=True)
def _reset_hooks(monkeypatch):
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    saved_hooks = {
        phase: list(hooks._hooks_by_phase[phase]) for phase in hooks.HOOK_PHASES
    }
    saved_index = hooks._next_insertion_index
    saved_entry_points_loaded = hooks._entry_points_loaded
    saved_correlation_provider = hooks._correlation_provider
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase].clear()
    hooks._next_insertion_index = 0
    hooks._entry_points_loaded = True
    monkeypatch.setattr(hooks, "load_entry_points", lambda: None)
    yield
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase][:] = saved_hooks[phase]
    hooks._next_insertion_index = saved_index
    hooks._entry_points_loaded = saved_entry_points_loaded
    hooks._correlation_provider = saved_correlation_provider


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "hook_wiring"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _make_llm(tmp_path: Path, provider) -> SommLLM:
    return SommLLM(config=_tmp_config(tmp_path), providers=[provider], on_error=lambda _: None)


def _call_row(llm: SommLLM, call_id: str) -> dict:
    llm._writer.flush(timeout=2.0)
    with sqlite3.connect(llm.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, provider, model, tokens_in, tokens_out, latency_ms, "
            "cost_usd, outcome, error_kind, error_detail, prompt_hash, "
            "prompt_id, temperature, max_tokens FROM calls WHERE id = ?",
            (call_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def test_generate_no_hooks_keeps_result_row_and_post_call_shape(tmp_path, monkeypatch):
    post_events: list[dict] = []
    monkeypatch.setattr(hooks, "fire_pre_call", lambda _ctx: pytest.fail("no pre_call expected"))
    monkeypatch.setattr(
        hooks,
        "PreCallContext",
        lambda **_kwargs: pytest.fail("no PreCallContext expected"),
    )
    monkeypatch.setattr(hooks, "fire_post_call", post_events.append)

    fake = FakeProvider(text="plain")
    llm = _make_llm(tmp_path, fake)
    try:
        result = llm.generate(
            "hello",
            workload="w",
            model="requested-model",
            max_tokens=123,
            temperature=0.4,
        )
        row = _call_row(llm, result.call_id)
    finally:
        llm.close()

    assert fake.generate_calls == 1
    assert fake.last_request.prompt == "hello"
    assert fake.last_request.model == "requested-model"
    assert result.text == "plain"
    assert result.provider == "fake"
    assert result.model == "requested-model"
    assert result.tokens_in == 7
    assert result.tokens_out == 5
    assert result.outcome == Outcome.OK
    assert row["provider"] == "fake"
    assert row["model"] == "requested-model"
    assert row["tokens_in"] == 7
    assert row["tokens_out"] == 5
    assert row["outcome"] == "ok"
    assert row["prompt_hash"] == stable_hash("hello")
    assert row["prompt_id"] is None
    assert row["temperature"] == 0.4
    assert row["max_tokens"] == 123
    assert len(post_events) == 1
    assert post_events[0]["schema_version"] == hooks.HOOK_EVENT_SCHEMA_VERSION
    assert post_events[0]["short_circuited"] is None


def test_generate_pre_call_mutation_rewrites_provider_request_and_prompt_hash(tmp_path):
    def redact(ctx: hooks.PreCallContext):
        ctx.prompt = "redacted"

    hooks.register_hook(hooks.PRE_CALL, redact)
    fake = FakeProvider(text="done")
    llm = _make_llm(tmp_path, fake)
    try:
        result = llm.generate("secret", workload="w")
        row = _call_row(llm, result.call_id)
    finally:
        llm.close()

    assert fake.last_request.prompt == "redacted"
    assert result.raw == {"prompt": "redacted"}
    assert row["prompt_hash"] == stable_hash("redacted")


def test_generate_pre_call_short_circuit_records_ok_row_and_event(tmp_path):
    post_events: list[dict] = []

    def cached(_ctx: hooks.PreCallContext):
        return hooks.ShortCircuit(
            text="cached",
            provider="cache",
            model="cache-model",
            tokens_in=1,
            tokens_out=2,
            raw={"hit": True},
            source="cache",
        )

    hooks.register_hook(hooks.PRE_CALL, cached)
    hooks.register_hook(hooks.POST_CALL, post_events.append)
    fake = FakeProvider(fail=True)
    llm = _make_llm(tmp_path, fake)
    try:
        result = llm.generate("hello", workload="w")
        row = _call_row(llm, result.call_id)
    finally:
        llm.close()

    assert fake.generate_calls == 0
    assert result.text == "cached"
    assert result.provider == "cache"
    assert result.model == "cache-model"
    assert result.cost_usd == 0.0
    assert row["provider"] == "cache"
    assert row["model"] == "cache-model"
    assert row["outcome"] == "ok"
    assert row["error_kind"] is None
    assert row["error_detail"] is None
    assert post_events[0]["short_circuited"] == "cache"


def test_generate_post_call_and_post_process_receive_stamped_events_without_blocking(tmp_path):
    post_call_events: list[dict] = []
    post_process_events: list[dict] = []
    started = Event()
    finished = Event()

    def slow_post_process(event: dict):
        post_process_events.append(event)
        started.set()
        time.sleep(0.25)
        finished.set()

    hooks.register_hook(hooks.POST_CALL, post_call_events.append)
    hooks.register_hook(hooks.POST_PROCESS, slow_post_process)
    llm = _make_llm(tmp_path, FakeProvider())
    try:
        before = time.monotonic()
        result = llm.generate("hello", workload="w")
        elapsed = time.monotonic() - before
        _call_row(llm, result.call_id)
    finally:
        llm.close()

    assert elapsed < 0.20
    assert started.wait(1)
    assert finished.wait(1)
    assert post_call_events[0]["schema_version"] == hooks.HOOK_EVENT_SCHEMA_VERSION
    assert post_process_events[0]["schema_version"] == hooks.HOOK_EVENT_SCHEMA_VERSION
    assert post_call_events[0]["call_id"] == result.call_id
    assert post_process_events[0]["call_id"] == result.call_id


def test_stream_short_circuit_yields_cached_text_and_records_row(tmp_path):
    def cached(_ctx: hooks.PreCallContext):
        return hooks.ShortCircuit(
            text="cached-stream",
            provider="cache",
            model="stream-cache",
            tokens_in=2,
            tokens_out=3,
            source="cache",
        )

    hooks.register_hook(hooks.PRE_CALL, cached)
    fake = FakeProvider(fail=True)
    llm = _make_llm(tmp_path, fake)
    try:
        chunks = list(llm.stream("hello", workload="w"))
        llm._writer.flush(timeout=2.0)
        with sqlite3.connect(llm.config.db_path) as conn:
            row = conn.execute(
                "SELECT provider, model, tokens_in, tokens_out, outcome, prompt_hash FROM calls"
            ).fetchone()
    finally:
        llm.close()

    assert fake.stream_calls == 0
    assert chunks == ["cached-stream"]
    assert row == ("cache", "stream-cache", 2, 3, "ok", stable_hash("hello"))


def test_embed_short_circuit_returns_cached_vector_and_records_row(tmp_path):
    def cached(_ctx: hooks.PreCallContext):
        return hooks.ShortCircuit(
            text="",
            provider="cache",
            model="embed-cache",
            tokens_in=4,
            raw={"embedding": [0.4, 0.5]},
            source="cache",
        )

    hooks.register_hook(hooks.PRE_CALL, cached)
    fake = FakeOllamaProvider(fail=True)
    llm = _make_llm(tmp_path, fake)
    try:
        result = llm.embed("hello", workload="w")
        row = _call_row(llm, result.call_id)
    finally:
        llm.close()

    assert fake.embed_calls == 0
    assert result.embedding == [0.4, 0.5]
    assert result.provider == "cache"
    assert result.model == "embed-cache"
    assert row["provider"] == "cache"
    assert row["model"] == "embed-cache"
    assert row["tokens_in"] == 4
    assert row["tokens_out"] == 0
    assert row["outcome"] == "ok"


def test_short_circuit_serves_budget_capped_workload(tmp_path):
    """A free short-circuit (cache hit) must bypass the fail-closed budget
    gate — it spends nothing — while a real call would be refused."""
    from somm.errors import SommBudgetExceeded

    cfg = _tmp_config(tmp_path)
    cfg.budget_fail_closed = True
    fake = FakeProvider(fail=True)  # must never be called
    llm = SommLLM(config=cfg, providers=[fake], on_error=lambda _: None)
    # Register the workload with a $0 daily cap: any real spend is refused.
    llm.register_workload(name="capped", budget_cap_usd_daily=0.0)

    def cached(_ctx):
        return hooks.ShortCircuit(text="from-cache", provider="cache", source="cache")

    try:
        # Without a hook, a real call is refused by the gate.
        with pytest.raises(SommBudgetExceeded):
            llm.generate("hi", workload="capped")

        # With a short-circuit, the capped workload is still served.
        hooks.register_hook(hooks.PRE_CALL, cached)
        result = llm.generate("hi", workload="capped")
        row = _call_row(llm, result.call_id)
    finally:
        llm.close()

    assert result.text == "from-cache"
    assert result.provider == "cache"
    assert row["outcome"] == "ok"

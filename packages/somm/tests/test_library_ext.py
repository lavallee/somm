"""Tests for D2c library additions: extract_structured, prompt versioning,
provenance helper, parallel_slots."""

from __future__ import annotations

from pathlib import Path

import pytest
import somm
from somm.client import SommLLM
from somm.prompts import PromptNotFound
from somm.providers.base import ProviderHealth, SommResponse
from somm_core import Outcome
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "ok", model: str = "fake-m"):
        self._text = text
        self._model = model
        self.received_models: list[str | None] = []

    def generate(self, request):
        self.received_models.append(request.model)
        return SommResponse(
            text=self._text,
            model=request.model or self._model,
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


class _FailingProvider(FakeProvider):
    """Records received model and always raises the given exception."""

    def __init__(self, name: str, exc: Exception):
        super().__init__()
        self.name = name
        self._exc = exc

    def generate(self, request):
        self.received_models.append(request.model)
        raise self._exc


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "d2c"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


# ---------------------------------------------------------------------------
# Pinned-provider fallback — regression for cross-provider model leak


# Use a SommTransientError with a cooldown longer than the router's
# exhausted_sleep_cap_s so exhaustion fires immediately (no real sleep) and
# these tests stay fast. 1-hour cooldown >> default 300s cap.
from somm.errors import SommTransientError as _SommTransient

_LONG_COOL = _SommTransient("upstream failed", cooldown_s=3600)


def test_error_detail_captured_on_exhausted(tmp_path):
    """When every provider fails, the SommResult carries a human-readable
    error_detail — exhaustion message + provider context."""
    p1 = _FailingProvider("p1", _LONG_COOL)

    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[p1], on_error=lambda _e: None)
    try:
        result = llm.generate(prompt="hi", workload="errdetail")
        assert result.outcome != Outcome.OK
        assert result.error_kind is not None
        assert result.error_detail is not None
        # error_detail should be informative — at minimum naming the
        # exception class somm raises.
        assert "Exhausted" in result.error_detail or "SommProviders" in result.error_detail
    finally:
        llm.close()


def test_on_error_callback_fires_on_failure(tmp_path):
    """SommLLM(on_error=cb) calls cb with a dict describing the failure."""
    p1 = _FailingProvider("p1", _LONG_COOL)

    events: list[dict] = []
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(
        config=cfg, providers=[p1], on_error=lambda e: events.append(e)
    )
    try:
        llm.generate(prompt="hi", workload="errcb")
        assert len(events) == 1
        ev = events[0]
        assert ev["workload"] == "errcb"
        assert ev["outcome"] != "ok"
        assert ev["error_kind"] is not None
        assert ev["error_detail"]
        assert "call_id" in ev
    finally:
        llm.close()


def test_on_error_does_not_fire_on_success(tmp_path):
    """Happy path: on_error must not be called when outcome is OK."""
    p1 = FakeProvider(text="ok")

    events: list[dict] = []
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(
        config=cfg, providers=[p1], on_error=lambda e: events.append(e)
    )
    try:
        llm.generate(prompt="hi", workload="okcb")
        assert events == []
    finally:
        llm.close()


def test_on_error_exception_is_swallowed(tmp_path):
    """A broken on_error handler must not break the caller."""
    p1 = _FailingProvider("p1", _LONG_COOL)

    def _broken(_event):
        raise ValueError("alerter itself is broken")

    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[p1], on_error=_broken)
    try:
        # Should NOT raise — alerter exceptions are swallowed.
        result = llm.generate(prompt="hi", workload="brokenalerter")
        assert result.outcome != Outcome.OK
    finally:
        llm.close()


def test_pinned_provider_failure_clears_model_on_fallback(tmp_path):
    """When the pinned (provider, model) fails, the fallback chain must NOT
    propagate the pinned model to other providers — they serve different
    inventories. Regression: pinning ollama/qwen3:14b and having ollama down
    would previously ask Minimax for qwen3:14b and also fail.
    """
    failing = _FailingProvider("ollama", RuntimeError("model 'qwen3:14b' not found"))
    backup = FakeProvider(text="fallback-output", model="minimax-default")
    backup.name = "minimax"

    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[failing, backup])
    try:
        result = llm.generate(
            prompt="hi",
            workload="pinfall",
            provider="ollama",
            model="qwen3:14b",
        )
        # Pinned provider first gets asked for the pinned model. When that
        # raises, somm falls through to the router chain which iterates all
        # providers — including ollama again, this time with model=None.
        assert failing.received_models == ["qwen3:14b", None]
        # Fallback provider got asked with model CLEARED — uses its default.
        assert backup.received_models == [None]
        # Call succeeded via the fallback.
        assert result.text == "fallback-output"
        assert result.provider == "minimax"
        assert result.outcome == Outcome.OK
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# extract_structured


def test_extract_structured_happy_path(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text='{"name": "foo", "count": 3}')
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("parse this", workload="es_ok")
        assert data == {"name": "foo", "count": 3}
    finally:
        llm.close()


def test_extract_structured_handles_markdown_fence(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text='```json\n{"a": 1}\n```')
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("p", workload="es_fence")
        assert data == {"a": 1}
    finally:
        llm.close()


def test_extract_structured_returns_raw_on_parse_fail(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="this is definitely not json")
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("p", workload="es_fail")
        assert data["_somm_parse_err"] is True
        assert data["raw"] == "this is definitely not json"
    finally:
        llm.close()


def test_extract_structured_think_stripped(tmp_path):
    cfg = _tmp_config(tmp_path)
    # Upstream strip is in adapters; but our FakeProvider returns raw —
    # the library-side parse still handles <think> via extract_json.
    fake = FakeProvider(text='<think>plan</think>{"ok": true}')
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("p", workload="es_think")
        assert data == {"ok": True}
    finally:
        llm.close()


class _ScriptedProvider:
    """Provider that returns a scripted list of texts in order, one per call.

    Records the `temperature` of each request for retry/jitter verification.
    """

    name = "scripted"

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls = 0
        self.temperatures: list[float] = []

    def generate(self, request):
        self.temperatures.append(request.temperature)
        idx = min(self.calls, len(self._texts) - 1)
        self.calls += 1
        return SommResponse(
            text=self._texts[idx],
            model=request.model or "scripted-m",
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


def test_extract_structured_retries_on_parse_failure(tmp_path):
    """Bad JSON on attempts 1+2, good JSON on 3 — retries=2 should succeed."""
    cfg = _tmp_config(tmp_path)
    scripted = _ScriptedProvider(
        texts=["garbage 1", "garbage 2", '{"ok": true}']
    )
    llm = SommLLM(config=cfg, providers=[scripted])
    try:
        data = llm.extract_structured("p", workload="es_retry", retries=2)
        assert data == {"ok": True}
        assert scripted.calls == 3
    finally:
        llm.close()


def test_extract_structured_bumps_temperature_on_retry(tmp_path):
    """Each retry increases temperature by temperature_jitter."""
    cfg = _tmp_config(tmp_path)
    scripted = _ScriptedProvider(texts=["bad", "bad", '{"a": 1}'])
    llm = SommLLM(config=cfg, providers=[scripted])
    try:
        llm.extract_structured(
            "p",
            workload="es_jitter",
            temperature=0.1,
            temperature_jitter=0.2,
            retries=2,
        )
        assert scripted.temperatures == pytest.approx([0.1, 0.3, 0.5])
    finally:
        llm.close()


def test_extract_structured_retries_zero_returns_err(tmp_path):
    """retries=0 means one attempt only; parse fail → _somm_parse_err dict."""
    cfg = _tmp_config(tmp_path)
    scripted = _ScriptedProvider(texts=["nope"])
    llm = SommLLM(config=cfg, providers=[scripted])
    try:
        data = llm.extract_structured("p", workload="es_no_retry", retries=0)
        assert data["_somm_parse_err"] is True
        assert data["raw"] == "nope"
        assert scripted.calls == 1
    finally:
        llm.close()


def test_extract_structured_exhaustion_returns_last_text(tmp_path):
    """When retries exhaust, the _somm_parse_err dict carries the LAST text."""
    cfg = _tmp_config(tmp_path)
    scripted = _ScriptedProvider(texts=["first-bad", "second-bad", "third-bad"])
    llm = SommLLM(config=cfg, providers=[scripted])
    try:
        data = llm.extract_structured("p", workload="es_exhaust", retries=2)
        assert data["_somm_parse_err"] is True
        assert data["raw"] == "third-bad"
        assert scripted.calls == 3
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Prompt versioning


def test_register_prompt_first_is_v1(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p = llm.register_prompt(workload="pv", body="Extract contacts from {text}")
        assert p.version == "v1"
    finally:
        llm.close()


def test_register_prompt_is_idempotent(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="pv", body="same body")
        p2 = llm.register_prompt(workload="pv", body="same body")
        assert p1.id == p2.id
        assert p1.version == p2.version == "v1"
    finally:
        llm.close()


def test_register_prompt_minor_bump_on_body_change(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="pv", body="first body")
        assert p1.version == "v1"
        p2 = llm.register_prompt(workload="pv", body="second body")
        assert p2.version == "v1.1"
        p3 = llm.register_prompt(workload="pv", body="third body")
        assert p3.version == "v1.2"
    finally:
        llm.close()


def test_register_prompt_major_bump_explicit(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.register_prompt(workload="pv", body="v1 body")
        p = llm.register_prompt(workload="pv", body="breaking change", bump="major")
        assert p.version == "v2"
    finally:
        llm.close()


def test_register_prompt_explicit_version(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p = llm.register_prompt(workload="pv", body="body", bump="v3")
        assert p.version == "v3"
    finally:
        llm.close()


def test_get_prompt_latest_and_pinned(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.register_prompt(workload="pv", body="body-1")
        llm.register_prompt(workload="pv", body="body-2")
        llm.register_prompt(workload="pv", body="body-3")

        latest = llm.prompt("pv", version="latest")
        assert latest.body == "body-3"

        pinned = llm.prompt("pv", version="v1")
        assert pinned.body == "body-1"
    finally:
        llm.close()


def test_get_prompt_missing_raises(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        with pytest.raises(PromptNotFound):
            llm.prompt("nonexistent_workload", version="v1")
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Provenance


def test_provenance_shape(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="hi")
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        r = llm.generate("hi", workload="prov")
        prov = somm.provenance(r)
        assert prov["schema"] == 1
        assert prov["call_id"] == r.call_id
        assert prov["provider"] == "fake"
        assert prov["model"] == "fake-m"
        assert prov["tokens_in"] == 3
        assert prov["tokens_out"] == 2
        assert prov["outcome"] == Outcome.OK.value
        assert isinstance(prov["stamped_at"], str)
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# parallel_slots


def test_parallel_slots_single_provider_fills(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider()
    fake.name = "ollama"
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        assert llm.parallel_slots(4) == ["ollama"] * 4
    finally:
        llm.close()


def test_parallel_slots_stripes_across_providers(tmp_path):
    cfg = _tmp_config(tmp_path)
    p1 = FakeProvider()
    p1.name = "ollama"
    p2 = FakeProvider()
    p2.name = "openrouter"
    llm = SommLLM(config=cfg, providers=[p1, p2])
    try:
        slots = llm.parallel_slots(6)
        # Ollama hint=2, openrouter=4 → 6 slots proportional: 2 ollama, 4 openrouter
        assert len(slots) == 6
        assert slots.count("ollama") == 2
        assert slots.count("openrouter") == 4
        # First two positions interleave (no stampede)
        assert slots[0] != slots[1] or slots[:2] == ["ollama", "ollama"]
    finally:
        llm.close()


def test_parallel_slots_zero(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        assert llm.parallel_slots(0) == []
    finally:
        llm.close()


def test_parallel_slots_skips_cooled(tmp_path):
    cfg = _tmp_config(tmp_path)
    p1 = FakeProvider()
    p1.name = "ollama"
    p2 = FakeProvider()
    p2.name = "openrouter"
    llm = SommLLM(config=cfg, providers=[p1, p2])
    try:
        llm._tracker.mark_failure("ollama", cooldown_s=300)
        slots = llm.parallel_slots(4)
        assert all(s == "openrouter" for s in slots)
    finally:
        llm.close()

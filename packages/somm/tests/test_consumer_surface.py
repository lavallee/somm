"""What a consuming project needs without writing wrapper code around somm.

One JSON call that says who served it, failures a caller can catch by type,
a per-call timeout that reaches every adapter, and a cheap health read.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
from pathlib import Path

import httpx
import pytest
import somm
from somm.client import SommLLM
from somm.errors import (
    SommEmptyResponse,
    SommProviderNotConfigured,
    SommRateLimited,
    SommRefused,
    SommStructuredError,
)
from somm.providers.base import SommRequest, SommResponse
from somm_core import Outcome, SommResult
from somm_core.config import Config


class Scripted:
    """Answers with the next scripted item: a string, or an exception to raise."""

    default_model = "scripted-1"

    def __init__(self, items=(), name: str = "fake", stop_reason: str = "") -> None:
        self.name = name
        self.items = list(items)
        self.stop_reason = stop_reason
        self.requests: list[SommRequest] = []

    def generate(self, request):
        self.requests.append(request)
        item = self.items.pop(0) if self.items else "{}"
        if isinstance(item, BaseException):
            raise item
        return SommResponse(
            text=item,
            model=request.model or self.default_model,
            tokens_in=3,
            tokens_out=2,
            latency_ms=1,
            stop_reason=self.stop_reason,
        )

    def stream(self, request):
        from somm.providers.base import SommChunk

        self.requests.append(request)
        yield SommChunk(text="streamed", done=True, model=self.default_model)

    def health(self):  # pragma: no cover - health() must not call this
        raise AssertionError("client.health() must not probe providers")

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _client(tmp_path: Path, *providers) -> SommLLM:
    cfg = Config()
    cfg.project = "surface"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(config=cfg, providers=list(providers), on_error=lambda _: None)


def _row(llm: SommLLM, call_id: str):
    llm._writer.flush()
    with llm.repo._open() as conn:
        return conn.execute(
            "SELECT provider, outcome FROM calls WHERE id = ?", (call_id,)
        ).fetchone()


# ---------------------------------------------------------------------------
# Failures a caller can catch


def test_pinned_failure_returns_by_default_and_says_so_plainly(tmp_path):
    llm = _client(tmp_path, Scripted([SommRateLimited("429", retry_after_s=5)]))
    try:
        result = llm.generate("hi", provider="fake")
    finally:
        llm.close()
    assert result.outcome == Outcome.RATE_LIMIT
    assert result.ok is False
    assert "429" in result.error


def test_raise_on_failure_raises_the_typed_error_with_its_route(tmp_path):
    llm = _client(tmp_path, Scripted([SommRateLimited("429", retry_after_s=5)]))
    try:
        with pytest.raises(somm.SommRateLimited) as info:
            llm.generate("hi", provider="fake", raise_on_failure=True)
        err = info.value
        assert err.provider == "fake"
        assert err.model == "scripted-1"
        assert err.call_id
        # The failure is on the ledger before the raise.
        assert _row(llm, err.call_id) == ("fake", "rate_limit")
    finally:
        llm.close()


def test_raise_on_failure_raises_exhausted_for_an_unpinned_chain(tmp_path):
    llm = _client(
        tmp_path,
        Scripted([somm.SommTimeout("slow")], name="a"),
        Scripted([somm.SommTimeout("slow")], name="b"),
    )
    try:
        with pytest.raises(somm.SommError) as info:
            llm.generate("hi", raise_on_failure=True)
    finally:
        llm.close()
    assert info.value.call_id


def test_raise_on_failure_is_quiet_on_success(tmp_path):
    llm = _client(tmp_path, Scripted(["fine"]))
    try:
        result = llm.generate("hi", provider="fake", raise_on_failure=True)
    finally:
        llm.close()
    assert result.ok and result.error is None and result.text == "fine"


def test_raise_on_failure_raises_on_empty_unless_empty_is_allowed(tmp_path):
    llm = _client(tmp_path, Scripted(["", ""]))
    try:
        with pytest.raises(SommEmptyResponse) as info:
            llm.generate("hi", provider="fake", raise_on_failure=True)
        assert info.value.call_id
        result = llm.generate("hi", provider="fake", raise_on_failure=True, allow_empty=True)
        assert result.outcome == Outcome.EMPTY
    finally:
        llm.close()


@pytest.mark.parametrize("stop_reason", ["refusal", "content_filter"])
def test_raise_on_failure_names_an_upstream_refusal(tmp_path, stop_reason):
    llm = _client(tmp_path, Scripted(["I can't help with that."], stop_reason=stop_reason))
    try:
        with pytest.raises(SommRefused) as info:
            llm.generate("hi", provider="fake", raise_on_failure=True)
    finally:
        llm.close()
    assert info.value.stop_reason == stop_reason
    assert info.value.provider == "fake"


def test_unconfigured_pin_is_typed_and_still_a_value_error(tmp_path):
    llm = _client(tmp_path, Scripted())
    try:
        with pytest.raises(SommProviderNotConfigured) as info:
            llm.generate("hi", provider="nope")
        assert isinstance(info.value, ValueError)
        assert info.value.provider == "nope"
    finally:
        llm.close()


def test_common_errors_are_exported_from_the_package():
    for name in (
        "SommAuthError",
        "SommEmptyResponse",
        "SommModelNotFound",
        "SommPrivacyViolation",
        "SommProviderError",
        "SommProviderNotConfigured",
        "SommProvidersExhausted",
        "SommRateLimited",
        "SommRefused",
        "SommStructuredError",
        "SommTimeout",
        "SommTransientError",
    ):
        assert name in somm.__all__, name
        assert issubclass(getattr(somm, name), somm.SommError), name


def test_describe_error_carries_the_route_when_known():
    err = SommRateLimited("429", model="m")
    err.provider, err.call_id = "p", "c1"
    described = somm.describe_error(err)
    assert described["provider"] == "p" and described["call_id"] == "c1"
    assert "provider" not in somm.describe_error(somm.SommError("x"))


# ---------------------------------------------------------------------------
# Per-call timeout


def test_timeout_reaches_the_provider_request_on_every_entry_point(tmp_path):
    provider = Scripted(['{"a": 1}'] * 4)
    llm = _client(tmp_path, provider)
    try:
        llm.generate("hi", provider="fake", timeout=7)
        list(llm.stream("hi", provider="fake", timeout=8))
        llm.generate_structured("hi", schema={"type": "object"}, provider="fake", timeout=9)
        llm.extract_structured("hi", provider="fake", timeout=10)
        llm.generate("hi", provider="fake")
    finally:
        llm.close()
    assert [r.timeout for r in provider.requests] == [7, 8, 9, 10, None]


def _capture_httpx_timeout(monkeypatch, status=200, body=None):
    """The read timeout each request went out with. Clients are reused, so
    the timeout rides on the request, not on the client."""
    seen: list = []
    real = httpx.Client

    def answer(request):
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(status, json=body or {})

    class Capturing(real):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.MockTransport(answer))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Capturing)
    return seen


_OPENAI_BODY = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def test_http_adapters_use_the_per_call_timeout(monkeypatch):
    from somm.providers.anthropic import AnthropicProvider
    from somm.providers.ollama import OllamaProvider
    from somm.providers.openai import OpenAIProvider

    seen = _capture_httpx_timeout(monkeypatch, body=_OPENAI_BODY)
    OpenAIProvider(api_key="k", timeout=50).generate(SommRequest(prompt="x", timeout=3))
    OpenAIProvider(api_key="k", timeout=50).generate(SommRequest(prompt="x"))
    assert seen == [3, 50]

    seen = _capture_httpx_timeout(
        monkeypatch,
        body={"content": [{"type": "text", "text": "ok"}], "usage": {}, "model": "m"},
    )
    AnthropicProvider(api_key="k", timeout=50).generate(SommRequest(prompt="x", timeout=4))
    assert seen == [4]

    seen = _capture_httpx_timeout(
        monkeypatch, body={"message": {"content": "ok"}, "done": True}
    )
    OllamaProvider(timeout=50).generate(SommRequest(prompt="x", timeout=5))
    assert seen == [5]


def test_configured_timeout_reaches_every_http_provider():
    from somm.providers import registry

    cfg = Config()
    cfg.http_timeout = 42.0
    cfg.anthropic_api_key = cfg.gemini_api_key = cfg.openrouter_api_key = "k"
    for factory in (registry._anthropic, registry._gemini, registry._openrouter, registry._ollama):
        assert factory(cfg, None).timeout == 42.0, factory.__name__


# ---------------------------------------------------------------------------
# One JSON call


def test_generate_structured_uses_the_workload_schema_when_the_call_gives_none(tmp_path):
    provider = Scripted(['{"b": 1}', '{"b": 1}', '{"a": 2}'])
    llm = _client(tmp_path, provider)
    try:
        llm.register_workload(
            name="triage",
            output_schema={"type": "object", "required": ["a"]},
        )
        value, result = llm.generate_structured("hi", workload="triage", provider="fake")
    finally:
        llm.close()
    assert value == {"a": 2}
    assert result.parsed == {"a": 2}
    assert result.ok and result.provider == "fake" and result.call_id
    assert '"required":["a"]' in provider.requests[0].system
    assert provider.requests[0].response_format["json_schema"]["schema"]["required"] == ["a"]


def test_generate_structured_without_any_schema_returns_whatever_json_came_back(tmp_path):
    provider = Scripted(['Sure! ```json\n[1, 2]\n```'])
    llm = _client(tmp_path, provider)
    try:
        value, result = llm.generate_structured("hi", provider="fake")
    finally:
        llm.close()
    assert value == [1, 2] and result.parsed == [1, 2]
    assert provider.requests[0].response_format is None


def test_structured_failure_carries_the_route_and_records_bad_json(tmp_path):
    llm = _client(tmp_path, Scripted(["nope", "nope", "nope"]))
    try:
        with pytest.raises(SommStructuredError) as info:
            llm.generate_structured("hi", schema={"type": "object"}, provider="fake")
        err = info.value
        assert (err.provider, err.model) == ("fake", "scripted-1")
        assert err.call_id == err.last_result.call_id
        assert _row(llm, err.call_id) == ("fake", "bad_json")
    finally:
        llm.close()


def test_extract_structured_keeps_its_sentinel_and_gains_a_plain_ok(tmp_path):
    llm = _client(tmp_path, Scripted(["nope", "nope", "nope", '{"x": 1}']))
    try:
        failed, result = llm.extract_structured("hi", provider="fake", with_result=True)
        assert failed == {"raw": "nope", "_somm_parse_err": True}
        assert result.ok is False and result.outcome == Outcome.BAD_JSON
        parsed, result = llm.extract_structured("hi", provider="fake", with_result=True)
        assert parsed == {"x": 1} and result.parsed == {"x": 1} and result.ok
    finally:
        llm.close()


def test_a_structured_result_stays_a_plain_value():
    result = SommResult(
        text='{"a": 1}', provider="p", model="m", tokens_in=1, tokens_out=1,
        latency_ms=1, cost_usd=0.0, call_id="c", parsed={"a": 1},
    )
    assert copy.deepcopy(result).parsed == {"a": 1}
    assert pickle.loads(pickle.dumps(result)).parsed == {"a": 1}
    assert dataclasses.asdict(result)["parsed"] == {"a": 1}
    assert "ok" not in dataclasses.asdict(result)


# ---------------------------------------------------------------------------
# Health


def test_health_reads_cooldowns_and_last_errors_without_probing(tmp_path):
    good, bad = Scripted(["fine"], name="good"), Scripted(
        [SommRateLimited("429", retry_after_s=600)], name="bad"
    )
    llm = _client(tmp_path, bad, good)
    try:
        llm.generate("hi", provider="bad")
        llm.generate("hi", provider="good")
        llm._tracker.mark_failure("bad", "", cooldown_s=600)
        llm._writer.flush()
        health = llm.health()
    finally:
        llm.close()
    assert health["bad"]["configured"] is True
    assert health["bad"]["cooled"] is True
    assert health["bad"]["cooled_until"]
    assert health["bad"]["last_error_kind"] == "SommRateLimited"
    assert health["good"]["cooled"] is False
    assert health["good"]["last_error_kind"] is None
    assert health["anthropic"]["configured"] is False

"""OpenRouter provider tests — httpx MockTransport, no network.

Covers: happy path, 429 with retry-after, 500, body-error, think-strip,
per-model roster cycling, all-models-cooled transient raise.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from somm.errors import (
    SommAuthError,
    SommTransientError,
)
from somm.providers.base import SommRequest
from somm.providers.openrouter import OpenRouterProvider
from somm.routing import ProviderHealthTracker
from somm_core.config import Config
from somm_core.repository import Repository


def _tmp_tracker(tmp_path: Path) -> ProviderHealthTracker:
    cfg = Config()
    cfg.db_dir = tmp_path / ".somm"
    return ProviderHealthTracker(Repository(cfg.db_path))


def _transport(handler):
    """Wrap a handler fn so we can inject it into an httpx.Client."""
    return httpx.MockTransport(handler)


def _openrouter_with_transport(
    handler,
    tracker: ProviderHealthTracker | None = None,
    roster: list[str] | None = None,
) -> OpenRouterProvider:
    """OpenRouterProvider where httpx.Client uses our mocked transport."""
    p = OpenRouterProvider(
        api_key="fake-key",
        roster=roster or ["m-a", "m-b"],
        tracker=tracker,
    )
    mock = _transport(handler)

    # Monkey-patch the client factory for tests.
    def _factory(timeout=60):
        return httpx.Client(transport=mock, timeout=timeout)

    p._client = _factory  # type: ignore[attr-defined]
    # Also patch the inline .post call path
    return p


def test_happy_path_returns_response(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer fake-key"
        assert request.headers["HTTP-Referer"]
        assert request.headers["X-Title"] == "somm"
        body = {
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))

    p = OpenRouterProvider(api_key="fake-key", roster=["m-a"], tracker=tr)
    resp = p.generate(SommRequest(prompt="ping"))
    assert resp.text == "pong"
    assert resp.model == "m-a"
    assert resp.tokens_in == 3
    assert resp.tokens_out == 1
    assert not tr.get("openrouter", "m-a").is_cooling()


def test_null_content_becomes_empty_not_crash(tmp_path, monkeypatch):
    """Some OpenRouter free models (elephant-alpha et al.) return
    {"content": null} on a 200 response — adapter used to crash with
    TypeError when strip_think_block saw None. Regression: treat null like
    empty, preserve call-tree so SommLLM can mark Outcome.EMPTY."""
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        body = {
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    resp = p.generate(SommRequest(prompt="hi"))
    # No crash — empty text instead.
    assert resp.text == ""
    assert resp.tokens_in == 5
    assert resp.tokens_out == 0


def test_think_block_stripped(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        body = {
            "choices": [{"message": {"content": "<think>planning</think>answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    resp = p.generate(SommRequest(prompt="hi"))
    assert resp.text == "answer"


def test_auth_error_is_fatal(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        return httpx.Response(401, text="unauthorized")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="bad", roster=["m-a"], tracker=tr)
    with pytest.raises(SommAuthError):
        p.generate(SommRequest(prompt="hi"))
    # Auth error is fatal — shouldn't cool the model
    assert not tr.get("openrouter", "m-a").is_cooling()


def test_rate_limited_respects_retry_after(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "180"}, text="slow down")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    with pytest.raises(SommTransientError):
        # Single model, rate limited → transient (all roster cooled)
        p.generate(SommRequest(prompt="hi"))
    h = tr.get("openrouter", "m-a")
    assert h.is_cooling()


def test_500_cools_model_transiently(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        return httpx.Response(502, text="bad gateway")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    with pytest.raises(SommTransientError):
        p.generate(SommRequest(prompt="hi"))
    assert tr.get("openrouter", "m-a").is_cooling()


def test_roster_cycles_to_next_on_failure(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)
    call_log = []

    def handler(request):
        body = request.read().decode() if request.read else ""
        # Extract model from request JSON
        import json as _json

        model = _json.loads(body).get("model")
        call_log.append(model)
        if model == "m-a":
            return httpx.Response(500, text="err")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok-from-b"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a", "m-b"], tracker=tr)
    resp = p.generate(SommRequest(prompt="hi"))
    assert call_log == ["m-a", "m-b"]
    assert resp.text == "ok-from-b"
    assert tr.get("openrouter", "m-a").is_cooling()
    assert not tr.get("openrouter", "m-b").is_cooling()


def test_body_error_treated_as_rate_limit(tmp_path, monkeypatch):
    """OpenRouter sometimes returns 200 with {'error': {...}} inside."""
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        return httpx.Response(
            200,
            json={"error": {"message": "rate limited for free tier", "code": 429}},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    with pytest.raises(SommTransientError):
        p.generate(SommRequest(prompt="hi"))


def test_all_roster_cooled_raises_transient_fast(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)
    tr.mark_failure("openrouter", "m-a", cooldown_s=300)
    tr.mark_failure("openrouter", "m-b", cooldown_s=300)

    def handler(request):
        pytest.fail("should not hit network when all roster models cooled")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a", "m-b"], tracker=tr)
    with pytest.raises(SommTransientError):
        p.generate(SommRequest(prompt="hi"))


def test_explicit_model_bypasses_roster(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        import json as _json

        body = _json.loads(request.read())
        assert body["model"] == "explicit/pick"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    resp = p.generate(SommRequest(prompt="hi", model="explicit/pick"))
    assert resp.model == "explicit/pick"


# ---------------------------------------------------------------------------


def _patch_client(handler):
    """Return a factory that yields an httpx.Client bound to MockTransport."""
    transport = httpx.MockTransport(handler)

    class _MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    return _MockedClient


# ---------------------------------------------------------------------------
# OpenRouter rides the OpenAI-compatible base: nothing on the request is
# dropped, and the gateway's own cost figure is recorded as reported.


def test_openrouter_forwards_tools_messages_and_reasoning(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)
    captured = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    p.generate(
        SommRequest(
            prompt="ignored when messages is set",
            system="be brief",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "get_weather", "parameters": {"type": "object"}}],
            tool_choice="any",
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )
    )
    assert captured["model"] == "m-a"
    assert captured["messages"][0] == {"role": "system", "content": "be brief"}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}
    assert captured["tools"][0]["function"]["name"] == "get_weather"
    assert captured["tool_choice"] == "required"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["reasoning_effort"] == "low"
    assert captured["usage"] == {"include": True}


def test_openrouter_reported_cost_and_reasoning_tokens(tmp_path, monkeypatch):
    tr = _tmp_tracker(tmp_path)

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 25},
                    "cost": 0.00042,
                },
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    resp = p.generate(SommRequest(prompt="hi"))
    assert resp.cost_usd == 0.00042
    assert resp.reasoning_tokens == 25
    assert resp.tokens_out == 30


def test_openrouter_exhaustion_has_no_sentinel_model(tmp_path, monkeypatch):
    """Sentinel names in the model field used to leak into per-model rollups."""
    tr = _tmp_tracker(tmp_path)
    tr.mark_failure("openrouter", "m-a", cooldown_s=300)

    def handler(request):
        pytest.fail("all cooled — no network")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenRouterProvider(api_key="k", roster=["m-a"], tracker=tr)
    with pytest.raises(SommTransientError) as exc_info:
        p.generate(SommRequest(prompt="hi"))
    assert exc_info.value.model == ""
    assert "cooldown" in str(exc_info.value)

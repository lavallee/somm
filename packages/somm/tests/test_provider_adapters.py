"""Tests for openai, minimax, anthropic adapters.

Shares the MockTransport pattern from test_openrouter.py. Exercises the
status-code classification, body-error handling, think-strip, auth path.
"""

from __future__ import annotations

import httpx
import pytest
from somm.errors import SommAuthError, SommBadRequest, SommRateLimited, SommUpstream5xx
from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import SommRequest
from somm.providers.minimax import MinimaxProvider
from somm.providers.openai import OpenAIProvider


def _patch_client(handler):
    transport = httpx.MockTransport(handler)

    class _MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    return _MockedClient


# ---------------------------------------------------------------------------
# OpenAI-compatible (covers openai + minimax + any gateway)


def _oai_ok_handler(content: str = "ok", usage=None):
    usage = usage or {"prompt_tokens": 5, "completion_tokens": 2}

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": usage,
            },
        )

    return handler


def test_openai_happy_path(monkeypatch):
    monkeypatch.setattr(httpx, "Client", _patch_client(_oai_ok_handler("pong")))
    p = OpenAIProvider(api_key="sk-fake")
    resp = p.generate(SommRequest(prompt="hi"))
    assert resp.text == "pong"
    assert resp.model == "gpt-4o-mini"
    assert resp.tokens_in == 5
    assert resp.tokens_out == 2


def test_openai_sends_bearer_and_system(monkeypatch):
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    p.generate(SommRequest(prompt="hi", system="be brief"))
    assert captured["auth"] == "Bearer sk-fake"
    import json as _json

    body = _json.loads(captured["body"])
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "be brief"
    assert body["messages"][1]["role"] == "user"


def test_openai_401_auth_error(monkeypatch):
    def handler(request):
        return httpx.Response(401, text="unauthorized")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-bad")
    with pytest.raises(SommAuthError):
        p.generate(SommRequest(prompt="hi"))


def test_openai_429_rate_limited(monkeypatch):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "90"}, text="slow")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    with pytest.raises(SommRateLimited) as exc_info:
        p.generate(SommRequest(prompt="hi"))
    assert exc_info.value.retry_after_s == 90.0


def test_openai_5xx_transient(monkeypatch):
    def handler(request):
        return httpx.Response(503, text="down")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    with pytest.raises(SommUpstream5xx):
        p.generate(SommRequest(prompt="hi"))


def test_openai_think_stripped(monkeypatch):
    monkeypatch.setattr(
        httpx, "Client", _patch_client(_oai_ok_handler("<think>plan</think>answer"))
    )
    p = OpenAIProvider(api_key="sk-fake")
    resp = p.generate(SommRequest(prompt="hi"))
    assert resp.text == "answer"


def test_openai_404_bad_request(monkeypatch):
    def handler(request):
        return httpx.Response(404, text="model not found")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    with pytest.raises(SommBadRequest):
        p.generate(SommRequest(prompt="hi", model="nonexistent"))


# ---------------------------------------------------------------------------
# Minimax (same base class, different endpoint)


def test_minimax_uses_minimaxi_endpoint(monkeypatch):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = MinimaxProvider(api_key="mm-fake")
    p.generate(SommRequest(prompt="hi"))
    assert "minimax.io" in captured["url"]


def test_minimax_default_model(monkeypatch):
    captured = {}

    def handler(request):
        import json as _json

        body = _json.loads(request.read())
        captured["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = MinimaxProvider(api_key="mm-fake")
    p.generate(SommRequest(prompt="hi"))
    assert captured["model"] == "MiniMax-M2.7"


# ---------------------------------------------------------------------------
# Anthropic (distinct Messages API)


def test_anthropic_happy_path(monkeypatch):
    def handler(request):
        assert request.headers["x-api-key"] == "sk-ant-fake"
        assert request.headers["anthropic-version"] == "2023-06-01"
        import json as _json

        body = _json.loads(request.read())
        # Anthropic: system is top-level, NOT in messages
        assert "system" not in [m.get("role") for m in body["messages"]]
        return httpx.Response(
            200,
            json={
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": "pong"}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="sk-ant-fake")
    resp = p.generate(SommRequest(prompt="ping", system="be brief"))
    assert resp.text == "pong"
    assert resp.tokens_in == 3
    assert resp.tokens_out == 1
    assert resp.model.startswith("claude-")


def test_anthropic_system_prompt_top_level(monkeypatch):
    captured = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    p.generate(SommRequest(prompt="hi", system="be brief"))
    assert captured.get("system") == "be brief"
    assert len(captured["messages"]) == 1
    assert captured["messages"][0]["role"] == "user"


def test_anthropic_drops_thinking_blocks(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "claude-opus-4-7",
                "content": [
                    {"type": "thinking", "thinking": "reasoning..."},
                    {"type": "text", "text": "answer"},
                ],
                "usage": {"input_tokens": 2, "output_tokens": 2},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    resp = p.generate(SommRequest(prompt="hi"))
    assert resp.text == "answer"
    assert "reasoning" not in resp.text


def test_anthropic_401_auth(monkeypatch):
    def handler(request):
        return httpx.Response(401, text="bad key")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="bad")
    with pytest.raises(SommAuthError):
        p.generate(SommRequest(prompt="hi"))


def test_anthropic_529_overloaded_is_transient(monkeypatch):
    from somm.errors import SommTransientError

    def handler(request):
        return httpx.Response(529, text="overloaded")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    with pytest.raises(SommTransientError):
        p.generate(SommRequest(prompt="hi"))


def test_anthropic_429_rate_limited(monkeypatch):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "45"}, text="slow")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    with pytest.raises(SommRateLimited) as exc_info:
        p.generate(SommRequest(prompt="hi"))
    assert exc_info.value.retry_after_s == 45.0


# ---------------------------------------------------------------------------
# Default provider chain reflects env keys


def test_default_provider_chain_only_local_when_no_keys(tmp_path, monkeypatch):
    from somm.client import SommLLM
    from somm_core.config import Config

    # Ensure no env keys leak into this test
    for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    cfg = Config()
    cfg.project = "chain-test"
    cfg.db_dir = tmp_path / ".somm"

    llm = SommLLM(config=cfg)
    try:
        names = [p.name for p in llm.providers]
        assert names == ["ollama"]
    finally:
        llm.close()


def test_default_provider_chain_includes_all_when_keys_set(tmp_path, monkeypatch):
    from somm.client import SommLLM
    from somm_core.config import Config

    cfg = Config()
    cfg.project = "chain-test-full"
    cfg.db_dir = tmp_path / ".somm"
    cfg.openrouter_api_key = "or-k"
    cfg.minimax_api_key = "mm-k"
    cfg.anthropic_api_key = "a-k"
    cfg.openai_api_key = "oai-k"

    llm = SommLLM(config=cfg)
    try:
        names = [p.name for p in llm.providers]
        # sovereign-first order: ollama then commercial providers
        assert names == ["ollama", "openrouter", "minimax", "anthropic", "openai"]
    finally:
        llm.close()

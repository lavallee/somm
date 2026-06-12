"""Tests for openai, minimax, anthropic adapters.

Shares the MockTransport pattern from test_openrouter.py. Exercises the
status-code classification, body-error handling, think-strip, auth path.
"""

from __future__ import annotations

import httpx
import pytest
from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommInsufficientCredit,
    SommRateLimited,
    SommTransientError,
    SommUpstream5xx,
)
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


def test_openai_gpt5_uses_max_completion_tokens(monkeypatch):
    captured = {}

    def handler(request):
        import json as _json

        captured["body"] = _json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    p.generate(SommRequest(prompt="hi", model="gpt-5.4", max_tokens=123))

    assert captured["body"]["max_completion_tokens"] == 123
    assert "max_tokens" not in captured["body"]


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


def test_openai_insufficient_quota_is_transient(monkeypatch):
    # Out of money is a billing state on this provider, not a bad key — the
    # router should skip it (transient), not abort the chain (fatal).
    def handler(request):
        return httpx.Response(
            429,
            json={"error": {"type": "insufficient_quota", "message": "quota"}},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    with pytest.raises(SommInsufficientCredit) as exc_info:
        p.generate(SommRequest(prompt="hi"))
    assert isinstance(exc_info.value, SommTransientError)
    assert not isinstance(exc_info.value, SommAuthError)
    assert exc_info.value.cooldown_s >= 600


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


def test_anthropic_credit_balance_is_transient(monkeypatch):
    # Anthropic returns a 400 for an exhausted balance. It's a billing state,
    # not a malformed request, so it must be transient (router falls through),
    # NOT a fatal SommBadRequest that aborts the whole fallback chain.
    def handler(request):
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "Your credit balance is too low to access the "
                        "Anthropic API. Please go to Plans & Billing to "
                        "upgrade or purchase credits."
                    ),
                },
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    with pytest.raises(SommInsufficientCredit) as exc_info:
        p.generate(SommRequest(prompt="hi"))
    assert isinstance(exc_info.value, SommTransientError)
    assert not isinstance(exc_info.value, SommBadRequest)


def test_anthropic_plain_400_still_fatal(monkeypatch):
    # A genuine malformed request (no billing wording) stays fatal — retrying
    # it on another provider would just fail again.
    def handler(request):
        return httpx.Response(400, text="invalid 'max_tokens': must be > 0")

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    with pytest.raises(SommBadRequest):
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
# Anthropic tool-calling (see docs/tool-calling.md)


_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


def _anthropic_text_ok(content_text: str = "ok", stop_reason: str = "end_turn"):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "claude-haiku-4-5-20251001",
                "content": [{"type": "text", "text": content_text}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": stop_reason,
            },
        )

    return handler


def test_anthropic_tools_renamed_to_input_schema(monkeypatch):
    """somm-neutral `parameters` is sent to Anthropic as `input_schema`."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _anthropic_text_ok()(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL]))

    assert "tools" in captured
    assert len(captured["tools"]) == 1
    tool = captured["tools"][0]
    assert tool["name"] == "get_weather"
    assert "input_schema" in tool
    assert "parameters" not in tool
    assert tool["input_schema"]["required"] == ["location"]


def test_anthropic_tool_choice_translations(monkeypatch):
    """tool_choice strings + explicit-tool dict translate to Anthropic shape."""
    captured: list = []

    def handler(request):
        import json as _json

        captured.append(_json.loads(request.read()).get("tool_choice"))
        return _anthropic_text_ok()(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL], tool_choice="auto"))
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL], tool_choice="any"))
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL], tool_choice="none"))
    p.generate(
        SommRequest(
            prompt="hi",
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
    )

    assert captured == [
        {"type": "auto"},
        {"type": "any"},
        {"type": "none"},
        {"type": "tool", "name": "get_weather"},
    ]


def test_anthropic_parses_tool_use_block(monkeypatch):
    """tool_use blocks in the response become ToolCall entries; stop_reason set."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_017abc",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    },
                ],
                "usage": {"input_tokens": 4, "output_tokens": 12},
                "stop_reason": "tool_use",
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    resp = p.generate(SommRequest(prompt="weather in SF?", tools=[_WEATHER_TOOL]))

    assert resp.text == "Let me check."
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "toolu_017abc"
    assert call.name == "get_weather"
    assert call.arguments == {"location": "SF"}


def test_anthropic_messages_overrides_prompt(monkeypatch):
    """SommRequest.messages takes precedence over prompt; multi-turn history is forwarded verbatim."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _anthropic_text_ok()(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    history = [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "tu_01",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_01", "content": "62F sunny"},
            ],
        },
    ]
    p.generate(SommRequest(prompt="IGNORED", messages=history, tools=[_WEATHER_TOOL]))

    assert captured["messages"] == history


def test_anthropic_no_tools_no_field(monkeypatch):
    """Backward compat: empty tools list omits the field entirely."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _anthropic_text_ok()(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = AnthropicProvider(api_key="k")
    p.generate(SommRequest(prompt="hi"))

    assert "tools" not in captured
    assert "tool_choice" not in captured


# ---------------------------------------------------------------------------
# OpenAI-compat tool-calling (covers openai, minimax, openrouter, deepseek)


def _oai_tool_use_handler(
    *,
    text: str = "checking",
    tool_id: str = "call_abc",
    tool_name: str = "get_weather",
    arguments_json: str = '{"location": "SF"}',
    finish_reason: str = "tool_calls",
):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "tool_calls": [
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": arguments_json,
                                    },
                                }
                            ],
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 7},
            },
        )

    return handler


def test_openai_tools_wrapped_as_function(monkeypatch):
    """somm-neutral tool → OpenAI `{type:function, function:{...}}` shape."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _oai_ok_handler("ok")(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL]))

    assert len(captured["tools"]) == 1
    sent = captured["tools"][0]
    assert sent["type"] == "function"
    assert sent["function"]["name"] == "get_weather"
    assert sent["function"]["parameters"]["required"] == ["location"]


def test_openai_tool_choice_translations(monkeypatch):
    """tool_choice strings + explicit-tool dict translate to OpenAI shape."""
    captured: list = []

    def handler(request):
        import json as _json

        captured.append(_json.loads(request.read()).get("tool_choice"))
        return _oai_ok_handler("ok")(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL], tool_choice="auto"))
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL], tool_choice="any"))
    p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL], tool_choice="none"))
    p.generate(
        SommRequest(
            prompt="hi",
            tools=[_WEATHER_TOOL],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
    )

    assert captured == [
        "auto",
        "required",
        "none",
        {"type": "function", "function": {"name": "get_weather"}},
    ]


def test_openai_parses_tool_calls(monkeypatch):
    """OpenAI tool_calls → ToolCall list with parsed JSON arguments; stop_reason normalized."""
    monkeypatch.setattr(httpx, "Client", _patch_client(_oai_tool_use_handler()))
    p = OpenAIProvider(api_key="sk-fake")
    resp = p.generate(SommRequest(prompt="weather?", tools=[_WEATHER_TOOL]))

    assert resp.text == "checking"
    assert resp.stop_reason == "tool_use"  # normalized from "tool_calls"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "get_weather"
    assert call.arguments == {"location": "SF"}
    assert call.arguments_raw == ""


def test_openai_malformed_tool_arguments_preserved(monkeypatch):
    """Bad JSON args don't crash; arguments_raw preserves the original for repair."""
    monkeypatch.setattr(
        httpx, "Client", _patch_client(_oai_tool_use_handler(arguments_json="{broken"))
    )
    p = OpenAIProvider(api_key="sk-fake")
    resp = p.generate(SommRequest(prompt="hi", tools=[_WEATHER_TOOL]))

    assert resp.tool_calls[0].arguments == {}
    assert resp.tool_calls[0].arguments_raw == "{broken"


@pytest.mark.parametrize(
    "finish_reason,expected_stop",
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "content_filter"),
        ("unrecognized", "unrecognized"),  # pass through, don't lie
    ],
)
def test_openai_finish_reason_normalization(monkeypatch, finish_reason, expected_stop):
    """OpenAI finish_reason values translate to somm-neutral stop_reason.

    Parametrized rather than looped — chained monkeypatch on httpx.Client
    creates a subclass wrapper stack where the innermost (oldest) transport
    wins, so a single-test loop reads the wrong handler.
    """
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    resp = p.generate(SommRequest(prompt="hi"))
    assert resp.stop_reason == expected_stop


def test_openai_translates_multi_turn_messages(monkeypatch):
    """Anthropic-shaped messages → OpenAI tool_calls + role:tool messages."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _oai_ok_handler("ok")(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    history = [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "tu_01",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_01", "content": "62F sunny"},
            ],
        },
    ]
    p.generate(SommRequest(prompt="IGNORED", messages=history, tools=[_WEATHER_TOOL]))

    msgs = captured["messages"]
    # Plain-string user message unchanged
    assert msgs[0] == {"role": "user", "content": "weather in SF?"}
    # Assistant message gets text + tool_calls
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "checking"
    assert len(msgs[1]["tool_calls"]) == 1
    tc = msgs[1]["tool_calls"][0]
    assert tc["id"] == "tu_01"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    import json as _json

    assert _json.loads(tc["function"]["arguments"]) == {"location": "SF"}
    # tool_result becomes role:tool with tool_call_id
    assert msgs[2] == {"role": "tool", "tool_call_id": "tu_01", "content": "62F sunny"}


def test_openai_assistant_only_tool_calls_no_content(monkeypatch):
    """When assistant only emits tool_use blocks (no text), content=None."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _oai_ok_handler("ok")(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    history = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_02", "name": "get_weather",
                 "input": {"location": "NYC"}},
            ],
        },
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_02", "content": "55F rain"},
        ]},
    ]
    p.generate(SommRequest(prompt="x", messages=history, tools=[_WEATHER_TOOL]))

    assistant_msg = captured["messages"][1]
    assert assistant_msg["content"] is None
    assert len(assistant_msg["tool_calls"]) == 1


def test_openai_no_tools_backward_compat(monkeypatch):
    """No tools = no tools/tool_choice fields in payload."""
    captured: dict = {}

    def handler(request):
        import json as _json

        captured.update(_json.loads(request.read()))
        return _oai_ok_handler("ok")(request)

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OpenAIProvider(api_key="sk-fake")
    p.generate(SommRequest(prompt="hi"))

    assert "tools" not in captured
    assert "tool_choice" not in captured


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


# ---------------------------------------------------------------------------
# Ollama — native `think: true` passthrough


def _ollama_ok_handler():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": "ok"},
                "prompt_eval_count": 3,
                "eval_count": 1,
            },
        )

    return handler


def test_ollama_omits_think_by_default(monkeypatch):
    """Default OllamaProvider config does not send `think`: true."""
    from somm.providers.ollama import OllamaProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))

    p = OllamaProvider()
    p.generate(SommRequest(prompt="hi", system="", model="qwen2.5:7b",
                           temperature=0.2, max_tokens=10))

    assert "think" not in captured["payload"]


def test_ollama_sends_think_when_enabled(monkeypatch):
    """OllamaProvider(enable_think=True) sends `think`: true on generate."""
    from somm.providers.ollama import OllamaProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))

    p = OllamaProvider(enable_think=True)
    p.generate(SommRequest(prompt="hi", system="", model="qwen3:14b",
                           temperature=0.2, max_tokens=10))

    assert captured["payload"].get("think") is True


def test_ollama_sends_keep_alive_by_default(monkeypatch):
    """Default OllamaProvider sends keep_alive="30m" so weights stay resident."""
    from somm.providers.ollama import OllamaProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))

    p = OllamaProvider()
    p.generate(SommRequest(prompt="hi", system="", model="qwen2.5:7b",
                           temperature=0.2, max_tokens=10))

    assert captured["payload"].get("keep_alive") == "30m"


def test_ollama_keep_alive_opt_out_with_empty_string(monkeypatch):
    """Passing keep_alive='' disables the knob (ollama uses its default)."""
    from somm.providers.ollama import OllamaProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))

    p = OllamaProvider(keep_alive="")
    p.generate(SommRequest(prompt="hi", system="", model="qwen2.5:7b",
                           temperature=0.2, max_tokens=10))

    assert "keep_alive" not in captured["payload"]


def test_ollama_keep_alive_config_threads_through_client(tmp_path, monkeypatch):
    """SOMM_OLLAMA_KEEP_ALIVE=1h → Config → OllamaProvider.keep_alive."""
    from somm.client import SommLLM
    from somm.providers.ollama import OllamaProvider
    from somm_core.config import load as load_config

    monkeypatch.setenv("SOMM_OLLAMA_KEEP_ALIVE", "1h")
    cfg = load_config(cwd=tmp_path)
    cfg.db_dir = tmp_path / ".somm"
    assert cfg.ollama_keep_alive == "1h"

    llm = SommLLM(config=cfg)
    try:
        ollama_provider = next(p for p in llm.providers if isinstance(p, OllamaProvider))
        assert ollama_provider.keep_alive == "1h"
    finally:
        llm.close()


def test_ollama_think_config_threads_through_client(tmp_path, monkeypatch):
    """SOMM_OLLAMA_THINK=1 → Config.ollama_think → OllamaProvider.enable_think."""
    from somm.client import SommLLM
    from somm.providers.ollama import OllamaProvider
    from somm_core.config import load as load_config

    monkeypatch.setenv("SOMM_OLLAMA_THINK", "1")
    cfg = load_config(cwd=tmp_path)
    cfg.db_dir = tmp_path / ".somm"
    assert cfg.ollama_think is True

    llm = SommLLM(config=cfg)
    try:
        ollama_provider = next(p for p in llm.providers if isinstance(p, OllamaProvider))
        assert ollama_provider.enable_think is True
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Ollama tool-calling (item 2 — native /api/chat tools field)


def test_ollama_sends_tools_openai_shape(monkeypatch):
    """somm-neutral tool → Ollama's OpenAI-shaped tools field on /api/chat."""
    from somm.providers.ollama import OllamaProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OllamaProvider()
    p.generate(SommRequest(prompt="weather?", model="llama3.1:8b", tools=[_WEATHER_TOOL]))

    tools = captured["payload"]["tools"]
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["parameters"]["required"] == ["location"]


def test_ollama_parses_tool_calls_args_already_dict(monkeypatch):
    """Ollama returns function.arguments as a dict (not JSON string); stop_reason=tool_use."""
    from somm.providers.ollama import OllamaProvider

    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {"location": "SF"}}}
                    ],
                },
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 7,
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OllamaProvider()
    resp = p.generate(SommRequest(prompt="weather?", model="llama3.1:8b", tools=[_WEATHER_TOOL]))

    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"location": "SF"}  # dict, no json.loads
    assert call.arguments_raw == ""


def test_ollama_tool_choice_raises(monkeypatch):
    """Ollama has no tool_choice — passing one is a loud SommBadRequest."""
    from somm.providers.ollama import OllamaProvider

    # No transport patch needed: the raise happens before any HTTP call.
    p = OllamaProvider()
    with pytest.raises(SommBadRequest):
        p.generate(
            SommRequest(prompt="hi", model="llama3.1:8b",
                        tools=[_WEATHER_TOOL], tool_choice="any")
        )


def test_ollama_translates_multi_turn_messages(monkeypatch):
    """Anthropic-shaped messages → Ollama's OpenAI-style tool_calls + role:tool."""
    from somm.providers.ollama import OllamaProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "62F"}, "prompt_eval_count": 1, "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = OllamaProvider()
    history = [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_01", "name": "get_weather",
                 "input": {"location": "SF"}},
            ],
        },
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_01", "content": "62F sunny"},
        ]},
    ]
    p.generate(SommRequest(prompt="IGNORED", model="llama3.1:8b",
                           messages=history, tools=[_WEATHER_TOOL]))

    msgs = captured["payload"]["messages"]
    assert msgs[0] == {"role": "user", "content": "weather in SF?"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert msgs[2] == {"role": "tool", "tool_call_id": "tu_01", "content": "62F sunny"}


# ---------------------------------------------------------------------------
# Gemini tool-calling (item 1 — inherited from OpenAICompatProvider)


def test_gemini_tools_pass_through_oai_compat(monkeypatch):
    """Gemini inherits OAI-compat tool support: tools wrapped, tool_choice
    translated, tool_calls parsed — no native functionDeclarations adapter."""
    from somm.providers.gemini import GeminiProvider

    captured: dict = {}

    def handler(request):
        import json as _json

        captured["payload"] = _json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_g1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "SF"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 7},
            },
        )

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    p = GeminiProvider(api_key="g-fake")
    resp = p.generate(
        SommRequest(prompt="weather?", tools=[_WEATHER_TOOL], tool_choice="any")
    )

    sent = captured["payload"]
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["tool_choice"] == "required"  # "any" → OpenAI "required"
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"location": "SF"}


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

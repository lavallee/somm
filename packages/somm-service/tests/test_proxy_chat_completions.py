"""Tests for the /v1/chat/completions OpenAI-compatible proxy route."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from somm.adaptive import _QUOTA
from somm.chatgpt_auth import ChatGPTAuth
from somm.providers._openai_compat import _translate_messages_to_openai
from somm.providers.base import ToolCall
from somm.providers.chatgpt import ChatGPTProvider, chat_completion
from somm_core.adaptive import WeeklyQuota
from somm_core.config import Config
from somm_core.pricing import write_intel
from somm_core.repository import Repository
from somm_service.app import create_app
from starlette.testclient import TestClient


def _cfg(tmp_path: Path, *, fail_closed: bool = False) -> Config:
    cfg = Config()
    cfg.project = "proxy-chat-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.budget_fail_closed = fail_closed
    return cfg


def _auth_headers(app, **extra: str) -> dict[str, str]:
    return {"authorization": f"Bearer {app.state.service_token.value}", **extra}


def _chatgpt(tmp_path: Path) -> ChatGPTProvider:
    # complete_chat is patched in every test; the auth file is never read.
    return ChatGPTProvider(auth=ChatGPTAuth(tmp_path / "chatgpt.json"))


def _answer(text: str = "Done", model: str = "served-model") -> dict:
    return chat_completion({"id": "r1", "model": model, "output": [
        {"type": "message", "content": [{"type": "output_text", "text": text}]}]})


_ADAPTIVE = {"adaptive": {
    "fast": {"provider": "chatgpt", "model": "small"},
    "strong": {"provider": "chatgpt", "model": "large"},
    "fallback": {"provider": "minimax", "model": "MiniMax-M3"},
    "fast_patterns": [r"add .+ to groceries"],
}}


def _quota(monkeypatch, used: int) -> None:
    monkeypatch.setattr(_QUOTA, "read", lambda: WeeklyQuota(used, int(time.time()) + 1000, time.time()))


@pytest.mark.parametrize("reasoning_effort", ["low", None])
def test_chat_completions_route_dispatches_through_the_client(
    tmp_path, fake_provider, ledger, reasoning_effort
):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="chat_wl", project=cfg.project)
    write_intel(repo, provider="openai", model="gpt-4o-mini", price_in_per_1m=1.0,
                price_out_per_1m=4.0, context_window=None, capabilities=None, source="test_seed")
    fake = fake_provider("openai", text="howdy", tokens_in=1_000_000, tokens_out=1_000_000)
    app = create_app(cfg, providers=[fake])

    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "reasoning_effort": reasoning_effort,
            "thinking": {"type": "adaptive"},
        },
        headers=_auth_headers(app, **{"x-somm-workload": "chat_wl"}),
    )

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o-mini"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "howdy"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                             "total_tokens": 2_000_000}

    sent = fake.requests[0]
    assert sent.model == "gpt-4o-mini"
    assert sent.system == "be brief"
    assert sent.messages == [{"role": "user", "content": "hello"}]
    assert sent.max_tokens == 32
    assert sent.reasoning_effort == reasoning_effort

    rows = ledger(app, "SELECT provider, model, tokens_in, tokens_out, cost_usd, outcome FROM calls")
    assert rows == [("openai", "gpt-4o-mini", 1_000_000, 1_000_000, 5.0, "ok")]


def test_chat_completions_route_accepts_explicit_provider_prefix(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="chat_wl", project=cfg.project)
    app = create_app(cfg, providers=[fake_provider("openrouter")])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "openrouter/openai/gpt-4o-mini",
              "messages": [{"role": "user", "content": "hello"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "chat_wl"}),
    )
    assert r.status_code == 200
    assert ledger(app, "SELECT provider, model FROM calls") == [("openrouter", "openai/gpt-4o-mini")]


def test_chat_completions_translates_tool_turns_and_returns_tool_calls(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    fake = fake_provider("openai", text="", tool_calls=[ToolCall("call_2", "lookup", {"id": 42})])
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "find it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": '{"q":"x"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "required",
    }, headers=_auth_headers(app))
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [{"id": "call_2", "type": "function",
                                                "function": {"name": "lookup", "arguments": '{"id": 42}'}}]
    sent = fake.requests[0]
    assert sent.tools == [{"name": "lookup", "description": "", "parameters": {"type": "object"}}]
    assert sent.tool_choice == "any"
    assert sent.messages == [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "search",
                                           "input": {"q": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                      "content": "result"}]},
    ]


def test_chat_completions_route_budget_gate_blocks_before_dispatch(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path, fail_closed=True)
    Repository(cfg.db_path).register_workload(name="capped", project=cfg.project,
                                              budget_cap_usd_daily=0.0)
    fake = fake_provider("openai")
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "capped"}),
    )
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert r.json()["error"]["retryable"] is False
    assert r.headers["x-should-retry"] == "false"
    assert fake.requests == []
    assert ledger(app, "SELECT COUNT(*) FROM calls")[0][0] == 0


def test_chat_completions_route_upstream_error_returns_502_and_records_call(
    tmp_path, fake_provider, ledger
):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="error_wl", project=cfg.project)
    app = create_app(cfg, providers=[fake_provider("openai", error=RuntimeError("provider down"))])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ping"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "error_wl"}),
    )
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "api_error"
    assert ledger(app, "SELECT outcome, error_kind FROM calls") == [("upstream_error", "RuntimeError")]


def test_chat_completions_route_respects_x_somm_project_header(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="wl-oai", project="custom-proj")
    app = create_app(cfg, providers=[fake_provider("openai")])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "wl-oai", "x-somm-project": "custom-proj"}),
    )
    assert r.status_code == 200
    assert ledger(app, "SELECT project FROM calls") == [("custom-proj",)]


def test_chat_completions_accepts_routine_fields_and_names_them(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    fake = fake_provider("openai")
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post("/v1/chat/completions", json={
        "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}],
        "stop": ["END"], "top_p": 1, "parallel_tool_calls": True, "n": 1, "logprobs": False,
        "thinking": {"type": "disabled"},
    }, headers=_auth_headers(app))
    assert r.status_code == 200
    assert r.headers["x-somm-ignored"] == "logprobs,n,parallel_tool_calls,stop,top_p,thinking"
    assert len(fake.requests) == 1


def test_chat_completions_route_rejects_missing_model_and_unsupported_fields(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    missing_model = client.post("/v1/chat/completions", json={"messages": []},
                                headers=_auth_headers(app))
    assert missing_model.status_code == 400
    assert missing_model.json()["error"]["type"] == "invalid_request_error"

    for extra in ({"n": 2}, {"logprobs": True}, {"top_logprobs": 2}, {"presence_penalty": 1},
                  {"tools": [{"type": "custom", "custom": {"name": "x"}}]}):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}], **extra},
            headers=_auth_headers(app),
        )
        assert r.status_code == 400, extra
        assert r.json()["error"]["type"] == "invalid_request_error"


def _chunks(text: str) -> list:
    frames = [f.removeprefix("data: ") for f in text.strip().split("\n\n")]
    assert frames[-1] == "[DONE]"
    return [json.loads(f) for f in frames[:-1]]


def test_chat_completions_streams_chunks_with_tool_calls_usage_and_done(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    fake = fake_provider("openai", text="hi", tokens_in=5, tokens_out=2,
                         tool_calls=[ToolCall("call_1", "search", {"q": "x"})])
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post("/v1/chat/completions", json={
        "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}],
        "stream": True, "stream_options": {"include_usage": True},
    }, headers=_auth_headers(app))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = _chunks(r.text)
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}
    assert len({c["id"] for c in chunks}) == 1
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert {"content": "hi"} in deltas
    assert {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'}}]} in deltas
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert [c["choices"][0]["finish_reason"] for c in finals] == ["tool_calls"]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert ledger(app, "SELECT outcome, tokens_in FROM calls") == [("ok", 5)]


def test_chat_completions_route_requires_auth_and_json_content_type(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    unauth = client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []})
    assert unauth.status_code == 403
    assert unauth.json()["error"]["type"] == "authentication_error"

    wrong_content_type = client.post(
        "/v1/chat/completions",
        content=b'{"model":"gpt-4o-mini","messages":[]}',
        headers=_auth_headers(app, **{"content-type": "text/plain"}),
    )
    assert wrong_content_type.status_code == 415
    assert wrong_content_type.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_route_rejects_unknown_explicit_workload_before_dispatch(
    tmp_path, fake_provider
):
    cfg = _cfg(tmp_path)
    fake = fake_provider("openai")
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "new-uncapped"}),
    )
    assert r.status_code == 403
    assert "pre-register" in r.json()["error"]["message"]
    assert fake.requests == []


def test_chat_completions_route_rejects_oversized_body_before_dispatch(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    cfg.service_proxy_max_body_bytes = 8
    fake = fake_provider("openai")
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/chat/completions", content=b'{"model":"x","messages":[]}',
        headers=_auth_headers(app, **{"content-type": "application/json", "content-length": "999"}),
    )
    assert r.status_code == 413
    assert fake.requests == []


def test_reasoning_tool_context_survives_a_round_trip(tmp_path, fake_provider):
    """A proxy must retain the provider context the next tool turn needs."""
    cfg = _cfg(tmp_path)
    tool_call = {"id": "synthetic-call", "type": "function", "function": {
        "name": "household_read", "arguments": '{"domain":"groceries"}'}}
    assistant = {"role": "assistant", "content": "", "tool_calls": [tool_call],
                 "reasoning_content": "opaque-provider-context",
                 "reasoning_details": [{"type": "reasoning.encrypted", "data": "synthetic-token"}]}
    fake = fake_provider("minimax", text="", tool_calls=[
        ToolCall("synthetic-call", "household_read", {"domain": "groceries"})],
        raw={"choices": [{"message": assistant, "finish_reason": "tool_calls"}]})
    app = create_app(cfg, providers=[fake])
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={"model": "minimax/MiniMax-M3",
        "messages": [{"role": "user", "content": "Read groceries"}]}, headers=_auth_headers(app))
    assert response.status_code == 200
    replay = response.json()["choices"][0]["message"]
    assert replay == assistant  # argument strings and reasoning fields byte-for-byte

    fake.text, fake.tool_calls, fake.raw = "Milk noted", [], None
    messages = [{"role": "user", "content": "Read groceries"}, replay,
                {"role": "tool", "tool_call_id": "synthetic-call", "content": "Milk"}]
    second = client.post("/v1/chat/completions", json={
        "model": "minimax/MiniMax-M3", "messages": messages}, headers=_auth_headers(app))
    assert second.status_code == 200
    # What the provider puts on the wire for the next turn keeps the context.
    wire = _translate_messages_to_openai(fake.requests[1].messages)
    assert wire[1]["reasoning_content"] == "opaque-provider-context"
    assert wire[1]["reasoning_details"] == assistant["reasoning_details"]
    assert wire[1]["tool_calls"][0]["id"] == "synthetic-call"
    assert wire[2] == {"role": "tool", "tool_call_id": "synthetic-call", "content": "Milk"}


def test_reasoning_context_on_provider_message_is_preserved(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    fake = fake_provider("minimax", text="Done", raw={"choices": [{"message": {
        "role": "assistant", "content": "Done", "reasoning_content": "opaque-provider-context",
        "reasoning": {"signature": "synthetic"}}}]})
    app = create_app(cfg, providers=[fake])
    result = TestClient(app).post("/v1/chat/completions", json={
        "model": "minimax/MiniMax-M3", "messages": [{"role": "user", "content": "Hello"}]},
        headers=_auth_headers(app))
    message = result.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "opaque-provider-context"
    assert message["reasoning"] == {"signature": "synthetic"}


def test_adaptive_conservation_defers_research_without_any_provider_call(
    tmp_path, monkeypatch, fake_provider, ledger
):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    workload = repo.register_workload(name="adaptive", project=cfg.project)
    repo.set_workload_policy(workload.id, _ADAPTIVE)
    _quota(monkeypatch, 90)
    fake = fake_provider("minimax")
    app = create_app(cfg, providers=[fake])
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "somm/auto",
            "messages": [{"role": "user", "content": "Write a biography of Alex"}]},
            headers=_auth_headers(app, **{"x-somm-workload": "adaptive"}))
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "SOMM_WORK_DEFERRED"
        assert response.json()["error"]["retryable"] is False
        assert response.headers["x-should-retry"] == "false"
        assert response.json()["error"]["routing"]["target"] is None
        assert fake.requests == []
        assert ledger(app, "SELECT COUNT(*) FROM calls")[0][0] == 0


def test_adaptive_simple_fallback_records_served_model_and_reason(
    tmp_path, monkeypatch, fake_provider, ledger
):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    workload = repo.register_workload(name="adaptive", project=cfg.project)
    repo.set_workload_policy(workload.id, _ADAPTIVE)
    _quota(monkeypatch, 90)
    fake = fake_provider("minimax", model="served-m3")
    app = create_app(cfg, providers=[fake])
    response = TestClient(app).post("/v1/chat/completions", json={"model": "somm/auto",
        "messages": [{"role": "user", "content": "Add milk to groceries"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "adaptive"}))
    assert response.status_code == 200
    assert response.json()["model"] == "served-m3"
    assert fake.requests[0].model == "MiniMax-M3"
    (provider, model, routing), = ledger(app, "SELECT provider, model, routing_json FROM calls")
    assert (provider, model) == ("minimax", "served-m3")
    assert json.loads(routing)["reason"] == "weekly_quota_conservation"


def test_subscription_proxy_bypasses_api_dollar_cap_without_losing_provenance(tmp_path, ledger):
    cfg = _cfg(tmp_path, fail_closed=True)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="seat", project=cfg.project, budget_cap_usd_daily=0)
    app = create_app(cfg, providers=[_chatgpt(tmp_path)])
    with patch.object(ChatGPTProvider, "complete_chat", return_value=_answer()):
        response = TestClient(app).post("/v1/chat/completions", json={
            "model": "chatgpt/requested-model", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(app, **{"x-somm-workload": "seat"}))
    assert response.status_code == 200
    (call_id,), = ledger(app, "SELECT id FROM calls")
    call = repo.get_call(call_id)
    assert call.model == "served-model" and call.cost_kind == "included"
    assert call.cost_usd == 0 and call.cost_source == "chatgpt-subscription"


def test_routing_preview_requires_auth_and_does_not_run_inference(tmp_path, monkeypatch, fake_provider):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="preview", project=cfg.project)
    repo.set_workload_policy(wl.id, _ADAPTIVE)
    _quota(monkeypatch, 90)
    fake = fake_provider("minimax")
    app = create_app(cfg, providers=[fake])
    with TestClient(app) as client:
        body = {"messages": [{"role": "user", "content": "Write a biography"}]}
        assert client.post("/api/routing/decide", json=body).status_code == 403
        response = client.post("/api/routing/decide", json=body,
            headers=_auth_headers(app, **{"x-somm-workload": "preview"}))
        assert response.status_code == 200 and response.json()["deferred"] is True
        assert fake.requests == []


@pytest.mark.parametrize('kind,status,retryable', [
    ('bad_request', 400, False), ('auth', 401, False), ('policy', 403, False),
    ('rate_limit', 429, True), ('timeout', 504, True), ('upstream', 502, True),
])
def test_typed_provider_failures_preserve_recovery_contract_without_fallback(
    tmp_path, fake_provider, ledger, kind, status, retryable
):
    from somm.errors import (
        SommAuthError,
        SommBadRequest,
        SommRateLimited,
        SommTimeout,
        SommUpstream5xx,
    )
    error = {
        'bad_request': SommBadRequest('Tool result has no matching call'),
        'auth': SommAuthError('Authorization expired'),
        'policy': SommAuthError('Plan permission denied'),
        'rate_limit': SommRateLimited('Try later', retry_after_s=3.2),
        'timeout': SommTimeout('Request timed out'),
        'upstream': SommUpstream5xx('Provider unavailable'),
    }[kind]
    error.provider_code = 'synthetic_provider_code'
    error.provider_type = 'invalid_request_error' if not retryable else 'server_error'
    error.provider_param = 'input' if kind == 'bad_request' else None
    error.provider_request_id = 'req_synthetic'
    if kind == 'policy':
        error.status_code = 403
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name='recoverable', project=cfg.project)
    other = fake_provider('openai')  # a healthy chain member that must not be used
    app = create_app(cfg, providers=[other, _chatgpt(tmp_path)])
    with patch.object(ChatGPTProvider, 'complete_chat', side_effect=error) as infer:
        response = TestClient(app).post('/v1/chat/completions', json={
            'model': 'chatgpt/pinned-strong', 'messages': [{'role': 'user', 'content': 'A synthetic task'}]},
            headers=_auth_headers(app, **{'x-somm-workload': 'recoverable'}))
    assert response.status_code == status
    body = response.json()['error']
    assert body['code'] == error.code and body['retryable'] is retryable
    assert body['provider'] == 'chatgpt' and body['model'] == 'pinned-strong'
    assert body['call_id'] == response.headers['x-somm-call-id']
    assert response.headers['x-should-retry'] == str(retryable).lower()
    assert body['provider_code'] == 'synthetic_provider_code'
    assert body['provider_request_id'] == 'req_synthetic'
    if kind == 'rate_limit':
        assert response.headers['retry-after'] == '4'
        assert body['retry_after_s'] == 3.2
    if kind == 'bad_request':
        assert body['param'] == 'input'
    assert infer.call_count == 1
    assert other.requests == []
    rows = ledger(app, 'SELECT id, provider, model, outcome FROM calls')
    assert rows[0][:3] == (body['call_id'], 'chatgpt', 'pinned-strong')
    assert rows[0][3] == {'timeout': 'timeout', 'rate_limit': 'rate_limit'}.get(kind, 'upstream_error')


@pytest.mark.parametrize('adaptive', [False, True])
def test_provider_recovery_metadata_is_recorded_for_pins_and_adaptive_calls(
    tmp_path, monkeypatch, ledger, adaptive
):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name='replayed', project=cfg.project)
    if adaptive:
        repo.set_workload_policy(wl.id, _ADAPTIVE)
        _quota(monkeypatch, 20)
    marker = {'kind': 'normalized_tool_replay', 'attempts': 0, 'rewritten_tool_calls': 5}
    answer = _answer()
    answer['somm_recovery'] = marker
    app = create_app(cfg, providers=[_chatgpt(tmp_path)])
    with patch.object(ChatGPTProvider, 'complete_chat', return_value=answer) as infer:
        response = TestClient(app).post('/v1/chat/completions', json={
            'model': 'somm/auto' if adaptive else 'chatgpt/pinned-model',
            'messages': [{'role': 'user', 'content': 'Hello'}]},
            headers=_auth_headers(app, **{'x-somm-workload': 'replayed'}))
    assert response.status_code == 200 and infer.call_count == 1
    assert response.json()['somm_recovery'] == marker
    (provider, model, routing), = ledger(app, 'SELECT provider, model, routing_json FROM calls')
    assert (provider, model) == ('chatgpt', 'served-model')
    route = json.loads(routing)
    assert route['recovery'] == marker
    if adaptive:
        assert route['tier'] == 'fast' and route['target']['model'] == 'small'
    else:
        assert route['route_source'] == 'pin'


def test_budget_refusal_does_not_trigger_openai_sdk_transport_retries():
    import httpx
    import openai
    from somm.errors import SommBudgetExceeded
    from somm_service.proxy import _failure

    status, payload, headers = _failure(SommBudgetExceeded('Daily cap reached'))
    attempts = []

    def refuse(request):
        attempts.append(request)
        return httpx.Response(status, json={'error': payload}, headers=headers)

    with openai.OpenAI(api_key='synthetic', base_url='http://somm.test/v1', max_retries=2,
                       http_client=httpx.Client(transport=httpx.MockTransport(refuse))) as client:
        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model='pinned-strong',
                                           messages=[{'role': 'user', 'content': 'Synthetic request'}])
    assert len(attempts) == 1

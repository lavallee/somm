"""Tests for the /v1/chat/completions OpenAI-compatible proxy gateway."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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


def _fake_chat_response(
    *,
    text: str = "hello back",
    tokens_in: int = 7,
    tokens_out: int = 3,
    finish: str = "stop",
    model: str = "gpt-4o-mini",
) -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-xyz",
        created=1_700_000_000,
        model=model,
        choices=[
            SimpleNamespace(
                message={"role": "assistant", "content": text},
                finish_reason=finish,
            )
        ],
        usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    )


def _auth_headers(app, **extra: str) -> dict[str, str]:
    return {"authorization": f"Bearer {app.state.service_token.value}", **extra}


@pytest.mark.parametrize("reasoning_effort", ["low", None])
def test_chat_completions_route_dispatches_litellm_and_returns_openai_shape(tmp_path, reasoning_effort):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="chat_wl", project=cfg.project)
    write_intel(
        repo,
        provider="openai",
        model="gpt-4o-mini",
        price_in_per_1m=1.0,
        price_out_per_1m=4.0,
        context_window=None,
        capabilities=None,
        source="test_seed",
    )
    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_chat_response(text="howdy", tokens_in=1_000_000, tokens_out=1_000_000)
    with patch("somm_service.proxy.litellm.completion", return_value=fake) as mock_comp:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
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
    assert body["usage"] == {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "total_tokens": 2_000_000,
    }

    kwargs = mock_comp.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["max_tokens"] == 32
    if reasoning_effort is None:
        assert "reasoning_effort" not in kwargs
    else:
        assert kwargs["reasoning_effort"] == reasoning_effort
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["timeout"] == cfg.http_timeout

    with repo._open() as conn:
        row = conn.execute(
            "SELECT provider, model, tokens_in, tokens_out, cost_usd, outcome FROM calls"
        ).fetchone()
    assert row == ("openai", "gpt-4o-mini", 1_000_000, 1_000_000, 5.0, "ok")


def test_chat_completions_route_accepts_explicit_provider_prefix(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="chat_wl", project=cfg.project)
    app = create_app(cfg)
    client = TestClient(app)

    with patch(
        "somm_service.proxy.litellm.completion",
        return_value=_fake_chat_response(),
    ):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "openrouter/openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=_auth_headers(app, **{"x-somm-workload": "chat_wl"}),
        )

    assert r.status_code == 200
    with repo._open() as conn:
        row = conn.execute("SELECT provider, model FROM calls").fetchone()
    assert row == ("openrouter", "openai/gpt-4o-mini")


def test_chat_completions_route_budget_gate_blocks_before_dispatch(tmp_path):
    cfg = _cfg(tmp_path, fail_closed=True)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="capped", project=cfg.project, budget_cap_usd_daily=0.0)
    app = create_app(cfg)
    client = TestClient(app)

    with patch("somm_service.proxy.litellm.completion") as mock_comp:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "x"}],
            },
            headers=_auth_headers(app, **{"x-somm-workload": "capped"}),
        )

    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert r.json()["error"]["retryable"] is False
    assert r.headers["x-should-retry"] == "false"
    mock_comp.assert_not_called()

    with repo._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


def test_chat_completions_route_upstream_error_returns_502_and_records_call(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="error_wl", project=cfg.project)
    app = create_app(cfg)
    client = TestClient(app)

    with patch(
        "somm_service.proxy.litellm.completion",
        side_effect=RuntimeError("provider down"),
    ):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers=_auth_headers(app, **{"x-somm-workload": "error_wl"}),
        )

    assert r.status_code == 502
    assert r.json()["error"]["type"] == "api_error"
    with repo._open() as conn:
        row = conn.execute("SELECT outcome, error_kind FROM calls").fetchone()
    assert row == ("upstream_error", "RuntimeError")


def test_chat_completions_route_respects_x_somm_project_header(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="wl-oai", project="custom-proj")
    app = create_app(cfg)
    client = TestClient(app)

    with patch(
        "somm_service.proxy.litellm.completion",
        return_value=_fake_chat_response(),
    ):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=_auth_headers(
                app,
                **{"x-somm-workload": "wl-oai", "x-somm-project": "custom-proj"},
            ),
        )

    assert r.status_code == 200
    with repo._open() as conn:
        row = conn.execute("SELECT project FROM calls").fetchone()
    assert row == ("custom-proj",)


def test_chat_completions_route_rejects_missing_model_and_stream_true(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    missing_model = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers=_auth_headers(app),
    )
    assert missing_model.status_code == 400
    assert missing_model.json()["error"]["type"] == "invalid_request_error"

    stream_true = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
        headers=_auth_headers(app),
    )
    assert stream_true.status_code == 400
    assert "streaming is not supported" in stream_true.json()["error"]["message"]


def test_chat_completions_route_requires_auth_and_json_content_type(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    unauth = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": []},
    )
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
    tmp_path,
):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    with patch("somm_service.proxy.litellm.completion") as mock_comp:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=_auth_headers(app, **{"x-somm-workload": "new-uncapped"}),
        )

    assert r.status_code == 403
    assert "pre-register" in r.json()["error"]["message"]
    mock_comp.assert_not_called()


def test_chat_completions_route_rejects_oversized_body_before_dispatch(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.service_proxy_max_body_bytes = 8
    app = create_app(cfg)
    client = TestClient(app)

    with patch("somm_service.proxy.litellm.completion") as mock_comp:
        r = client.post(
            "/v1/chat/completions",
            content=b'{"model":"x","messages":[]}',
            headers=_auth_headers(
                app,
                **{"content-type": "application/json", "content-length": "999"},
            ),
        )

    assert r.status_code == 413
    mock_comp.assert_not_called()


def test_reasoning_tool_context_survives_a_round_trip(tmp_path):
    """A proxy must retain the provider context the next tool turn needs."""
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)
    tool_call = {"id": "synthetic-call", "type": "function", "function": {
        "name": "household_read", "arguments": '{"domain":"groceries"}'}}
    assistant = {"role": "assistant", "content": "", "tool_calls": [tool_call],
                 "reasoning_content": "opaque-provider-context",
                 "reasoning_details": [{"type": "reasoning.encrypted", "data": "synthetic-token"}]}
    first = _fake_chat_response(finish="tool_calls")
    first.choices[0].message = assistant
    with patch("somm_service.proxy.litellm.completion", return_value=first):
        response = client.post("/v1/chat/completions", json={"model": "minimax/MiniMax-M3",
            "messages": [{"role": "user", "content": "Read groceries"}]},
            headers=_auth_headers(app))
    assert response.status_code == 200
    replay = response.json()["choices"][0]["message"]
    assert replay == assistant
    messages = [{"role": "user", "content": "Read groceries"}, replay,
                {"role": "tool", "tool_call_id": "synthetic-call", "content": "Milk"}]
    with patch("somm_service.proxy.litellm.completion", return_value=_fake_chat_response()) as call:
        second = client.post("/v1/chat/completions", json={
            "model": "minimax/MiniMax-M3", "messages": messages}, headers=_auth_headers(app))
    assert second.status_code == 200
    assert call.call_args.kwargs["messages"] == messages


def test_reasoning_context_on_sdk_message_is_preserved(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)
    response = _fake_chat_response()
    response.choices[0].message = SimpleNamespace(role="assistant", content="Done",
        reasoning_content="opaque-provider-context", reasoning={"signature": "synthetic"},
        tool_calls=None)
    with patch("somm_service.proxy.litellm.completion", return_value=response):
        result = client.post("/v1/chat/completions", json={
            "model": "minimax/MiniMax-M3", "messages": [{"role": "user", "content": "Hello"}]},
            headers=_auth_headers(app))
    message = result.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "opaque-provider-context"
    assert message["reasoning"] == {"signature": "synthetic"}


def test_adaptive_conservation_defers_research_without_any_provider_call(tmp_path, monkeypatch):
    import time

    from somm.adaptive import _QUOTA
    from somm_core.adaptive import WeeklyQuota

    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    workload = repo.register_workload(name="adaptive", project=cfg.project)
    repo.set_workload_policy(workload.id, {"adaptive": {
        "fast": {"provider": "chatgpt", "model": "small"},
        "strong": {"provider": "chatgpt", "model": "large"},
        "fallback": {"provider": "minimax", "model": "MiniMax-M3"},
        "fast_patterns": [r"add .+ to groceries"],
    }})
    monkeypatch.setattr(_QUOTA, "read", lambda: WeeklyQuota(90, int(time.time()) + 1000, time.time()))
    app = create_app(cfg)
    with TestClient(app) as client, patch("somm_service.proxy.litellm.completion") as infer:
        response = client.post("/v1/chat/completions", json={"model": "somm/auto",
            "messages": [{"role": "user", "content": "Write a biography of Alex"}]},
            headers=_auth_headers(app, **{"x-somm-workload": "adaptive"}))
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "SOMM_WORK_DEFERRED"
    assert response.json()["error"]["retryable"] is False
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"]["routing"]["target"] is None
    infer.assert_not_called()
    with repo._open() as connection:
        assert connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


def test_adaptive_simple_fallback_records_served_model_and_reason(tmp_path, monkeypatch):
    import json
    import time

    from somm.adaptive import _QUOTA
    from somm_core.adaptive import WeeklyQuota
    from somm_core.read import search_calls

    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    workload = repo.register_workload(name="adaptive", project=cfg.project)
    repo.set_workload_policy(workload.id, {"adaptive": {
        "fast": {"provider": "chatgpt", "model": "small"},
        "strong": {"provider": "chatgpt", "model": "large"},
        "fallback": {"provider": "minimax", "model": "MiniMax-M3"},
        "fast_patterns": [r"add .+ to groceries"],
    }})
    monkeypatch.setattr(_QUOTA, "read", lambda: WeeklyQuota(90, int(time.time()) + 1000, time.time()))
    app = create_app(cfg)
    with TestClient(app) as client, patch("somm_service.proxy.litellm.completion",
                                         return_value=_fake_chat_response(model="served-m3")) as infer:
        response = client.post("/v1/chat/completions", json={"model": "somm/auto",
            "messages": [{"role": "user", "content": "Add milk to groceries"}]},
            headers=_auth_headers(app, **{"x-somm-workload": "adaptive"}))
    assert response.status_code == 200
    assert response.json()["model"] == "served-m3"
    assert infer.call_args.kwargs["model"] == "minimax/MiniMax-M3"
    calls = search_calls(cfg.db_path, cfg.project)
    assert calls[0]["model"] == "served-m3" and calls[0]["provider"] == "minimax"
    assert json.loads(calls[0]["routing_json"])["reason"] == "weekly_quota_conservation"


def test_subscription_proxy_bypasses_api_dollar_cap_without_losing_provenance(tmp_path):
    from somm.providers.chatgpt import chat_completion
    from somm_core.read import search_calls
    cfg = _cfg(tmp_path, fail_closed=True)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="seat", project=cfg.project, budget_cap_usd_daily=0)
    app = create_app(cfg)
    answer = chat_completion({"id": "r1", "model": "served-model", "output": []})
    with TestClient(app) as client, patch("somm.providers.chatgpt.ChatGPTProvider.complete_chat", return_value=answer):
        response = client.post("/v1/chat/completions", json={"model": "chatgpt/requested-model",
            "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers(app, **{"x-somm-workload": "seat"}))
    assert response.status_code == 200
    call = repo.get_call(search_calls(cfg.db_path, cfg.project)[0]["id"])
    assert call.model == "served-model" and call.cost_kind == "included"
    assert call.cost_usd == 0 and call.cost_source == "chatgpt-subscription"


def test_routing_preview_requires_auth_and_does_not_run_inference(tmp_path, monkeypatch):
    import time

    from somm.adaptive import _QUOTA
    from somm_core.adaptive import WeeklyQuota
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="preview", project=cfg.project)
    repo.set_workload_policy(wl.id, {"adaptive": {
        "fast": {"provider": "chatgpt", "model": "small"},
        "strong": {"provider": "chatgpt", "model": "large"},
        "fallback": {"provider": "minimax", "model": "MiniMax-M3"}}})
    monkeypatch.setattr(_QUOTA, "read", lambda: WeeklyQuota(90, int(time.time())+1000, time.time()))
    app = create_app(cfg)
    with TestClient(app) as client, patch("somm_service.proxy.litellm.completion") as infer:
        body = {"messages": [{"role": "user", "content": "Write a biography"}]}
        assert client.post("/api/routing/decide", json=body).status_code == 403
        response = client.post("/api/routing/decide", json=body,
            headers=_auth_headers(app, **{"x-somm-workload": "preview"}))
        assert response.status_code == 200 and response.json()["deferred"] is True
        infer.assert_not_called()


@pytest.mark.parametrize('kind,status,retryable', [
    ('bad_request', 400, False), ('auth', 401, False), ('policy', 403, False),
    ('rate_limit', 429, True), ('timeout', 504, True), ('upstream', 502, True),
])
def test_typed_provider_failures_preserve_recovery_contract_without_fallback(tmp_path, kind, status, retryable):
    from somm.errors import (
        SommAuthError,
        SommBadRequest,
        SommRateLimited,
        SommTimeout,
        SommUpstream5xx,
    )
    errors = {
        'bad_request': SommBadRequest('Tool result has no matching call'),
        'auth': SommAuthError('Authorization expired'),
        'policy': SommAuthError('Plan permission denied'),
        'rate_limit': SommRateLimited('Try later', retry_after_s=3.2),
        'timeout': SommTimeout('Request timed out'),
        'upstream': SommUpstream5xx('Provider unavailable'),
    }
    error = errors[kind]
    error.provider_code = 'synthetic_provider_code'
    error.provider_type = 'invalid_request_error' if not retryable else 'server_error'
    error.provider_param = 'input' if kind == 'bad_request' else None
    error.provider_request_id = 'req_synthetic'
    if kind == 'policy':
        error.status_code = 403
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name='recoverable', project=cfg.project)
    app = create_app(cfg)
    with TestClient(app) as client, patch('somm.providers.chatgpt.ChatGPTProvider.complete_chat', side_effect=error) as infer, patch('somm_service.proxy.litellm.completion') as fallback:
        response = client.post('/v1/chat/completions', json={
            'model': 'chatgpt/pinned-strong', 'messages': [{'role':'user','content':'A synthetic task'}]},
            headers=_auth_headers(app, **{'x-somm-workload':'recoverable'}))
    assert response.status_code == status
    body = response.json()['error']
    assert body['code'] == error.code and body['retryable'] is retryable
    assert body['provider_code'] == 'synthetic_provider_code'
    assert body['provider_request_id'] == 'req_synthetic'
    assert body['provider'] == 'chatgpt' and body['model'] == 'pinned-strong'
    assert body['call_id'] == response.headers['x-somm-call-id']
    assert response.headers['x-should-retry'] == str(retryable).lower()
    if kind == 'rate_limit':
        assert response.headers['retry-after'] == '4'
        assert body['retry_after_s'] == 3.2
    if kind == 'bad_request':
        assert body['param'] == 'input'
    assert infer.call_count == 1
    fallback.assert_not_called()
    with repo._open() as conn:
        row = conn.execute('SELECT id, provider, model, outcome FROM calls').fetchone()
    assert row[:3] == (body['call_id'], 'chatgpt', 'pinned-strong')
    assert row[3] == ('timeout' if kind == 'timeout' else 'upstream_error')


@pytest.mark.parametrize('adaptive', [False, True])
def test_provider_recovery_metadata_is_recorded_for_pins_and_adaptive_calls(tmp_path, monkeypatch, adaptive):
    import json
    import time

    from somm.adaptive import _QUOTA
    from somm.providers.chatgpt import chat_completion
    from somm_core.adaptive import WeeklyQuota
    from somm_core.read import search_calls

    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name='replayed', project=cfg.project)
    if adaptive:
        repo.set_workload_policy(wl.id, {'adaptive': {
            'fast': {'provider': 'chatgpt', 'model': 'small'},
            'strong': {'provider': 'chatgpt', 'model': 'large'},
            'fallback': {'provider': 'minimax', 'model': 'MiniMax-M3'}}})
        monkeypatch.setattr(_QUOTA, 'read', lambda: WeeklyQuota(20, int(time.time())+1000, time.time()))
    marker = {'kind': 'normalized_tool_replay', 'attempts': 0, 'rewritten_tool_calls': 5}
    answer = chat_completion({'id':'r1', 'model':'served-model', 'output':[]})
    answer['somm_recovery'] = marker
    app = create_app(cfg)
    with TestClient(app) as client, patch('somm.providers.chatgpt.ChatGPTProvider.complete_chat', return_value=answer) as infer:
        response = client.post('/v1/chat/completions', json={
            'model': 'somm/auto' if adaptive else 'chatgpt/pinned-model',
            'messages':[{'role':'user','content':'Hello'}]},
            headers=_auth_headers(app, **{'x-somm-workload':'replayed'}))
    assert response.status_code == 200 and infer.call_count == 1
    assert response.json()['somm_recovery'] == marker
    call = search_calls(cfg.db_path, cfg.project)[0]
    assert call['provider'] == 'chatgpt' and call['model'] == 'served-model'
    route = json.loads(call['routing_json'])
    assert route['recovery'] == marker
    if adaptive:
        assert route['tier'] == 'fast' and route['target']['model'] == 'small'
    else:
        assert set(route) == {'recovery'}


def test_budget_refusal_does_not_trigger_openai_sdk_transport_retries():
    import httpx
    import openai
    from somm.errors import SommBudgetExceeded
    from somm_service.proxy import _openai_provider_error

    response = _openai_provider_error(SommBudgetExceeded('Daily cap reached'))
    attempts = []
    def refuse(request):
        attempts.append(request)
        return httpx.Response(response.status_code, content=response.body, headers=response.headers)
    with openai.OpenAI(api_key='synthetic', base_url='http://somm.test/v1', max_retries=2,
                       http_client=httpx.Client(transport=httpx.MockTransport(refuse))) as client:
        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model='pinned-strong', messages=[{'role':'user','content':'Synthetic request'}])
    assert len(attempts) == 1

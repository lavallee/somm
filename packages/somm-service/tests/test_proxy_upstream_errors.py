"""Upstream detail and refusals through the proxy's error mapping."""

from __future__ import annotations

from pathlib import Path

from somm.errors import SommRefused, http_error
from somm_core.config import Config
from somm_core.repository import Repository
from somm_service.app import create_app
from somm_service.proxy import _CHAT, _chat_body, _refusal
from starlette.testclient import TestClient

MODEL = "claude-haiku-4-5-20251001"


def _app(tmp_path: Path, provider):
    cfg = Config()
    cfg.project = "proxy-upstream-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    Repository(cfg.db_path).register_workload(name="wl", project=cfg.project)
    return create_app(cfg, providers=[provider])


def _post(app, route: str, body: dict):
    headers = {"authorization": f"Bearer {app.state.service_token.value}", "x-somm-workload": "wl"}
    return TestClient(app).post(route, json=body, headers=headers)


def test_an_upstream_403_stays_403_with_the_providers_fields(tmp_path, fake_provider):
    error = http_error(
        403,
        '{"type":"error","error":{"type":"permission_error","message":"no"},"request_id":"req_1"}',
        provider="anthropic", model=MODEL, headers={"request-id": "req_1"},
    )
    app = _app(tmp_path, fake_provider(error=error))
    r = _post(app, "/v1/messages",
              {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["type"] == "authentication_error"
    assert err["provider_type"] == "permission_error"
    assert err["provider_request_id"] == "req_1"
    assert err["status_code"] == 403
    assert r.headers["x-should-retry"] == "false"


def test_exhausted_credit_is_not_retryable_through_the_proxy(tmp_path, fake_provider):
    error = http_error(402, '{"error":{"code":402,"message":"add credits"}}',
                       provider="openrouter", model="m")
    app = _app(tmp_path, fake_provider("openai", error=error))
    r = _post(app, "/v1/chat/completions",
              {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 402
    assert r.json()["error"]["code"] == "SOMM_PROVIDER_INSUFFICIENT_CREDIT"
    assert r.json()["error"]["retryable"] is False
    assert r.headers["x-should-retry"] == "false"


def test_a_refusal_is_answered_in_the_anthropic_refusal_shape(tmp_path, fake_provider):
    app = _app(tmp_path, fake_provider(stop_reason="refusal", text=""))
    r = _post(app, "/v1/messages",
              {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message" and body["stop_reason"] == "refusal"
    assert body["id"] == "msg_" + r.headers["x-somm-call-id"].replace("-", "")


def test_a_refusal_is_a_content_filter_finish_on_chat_completions(tmp_path, fake_provider):
    app = _app(tmp_path, fake_provider("openai", stop_reason="content_filter", text=""))
    r = _post(app, "/v1/chat/completions",
              {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "content_filter"


def test_refusal_carries_text_and_usage_when_the_client_attaches_them():
    exc = SommRefused("declined", stop_reason="refusal", text="I can't help with that.",
                      tokens_in=11, tokens_out=6, model="served", call_id="c-1")
    body = _chat_body(_refusal(exc, _CHAT), "gpt-x")
    assert body["choices"][0]["message"]["content"] == "I can't help with that."
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17}

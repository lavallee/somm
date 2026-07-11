"""Tests for the /v1/chat/completions OpenAI-compatible proxy gateway."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


def test_chat_completions_route_dispatches_litellm_and_returns_openai_shape(tmp_path):
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

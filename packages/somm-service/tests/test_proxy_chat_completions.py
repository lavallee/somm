"""Tests for the /v1/chat/completions OpenAI-compatible proxy gateway.

Coverage:
  - request mapping: OpenAI body → litellm params
  - response mapping: litellm response → OpenAI chat.completion shape
  - budget gate fires BEFORE litellm is called (returns 429)
  - telemetry row is written to calls.sqlite (one ledger, same as /v1/messages)
  - X-Somm-Project header overrides the project used for workload routing
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from somm_core.config import Config
from somm_core.pricing import write_intel
from somm_core.repository import Repository
from somm_service.app import create_app
from somm_service.proxy import _litellm_to_openai_response, _openai_to_litellm_params
from starlette.testclient import TestClient


def _cfg(tmp_path: Path, *, fail_closed: bool = False) -> Config:
    cfg = Config()
    cfg.project = "oai-proxy-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.budget_fail_closed = fail_closed
    return cfg


def _fake_litellm_response(
    *,
    text: str = "hello back",
    tokens_in: int = 7,
    tokens_out: int = 3,
    finish: str = "stop",
    model: str = "claude-haiku-4-5-20251001",
    tool_calls: list | None = None,
) -> SimpleNamespace:
    """Minimal OpenAI/litellm-shaped ModelResponse stand-in."""
    message = {"role": "assistant", "content": text, "tool_calls": tool_calls or []}
    return SimpleNamespace(
        id="chatcmpl-test123",
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    )


# -- Pure mapping unit tests -------------------------------------------------


def test_openai_to_litellm_params_basic():
    body = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.5,
    }
    params = _openai_to_litellm_params(body)
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert params["max_tokens"] == 64
    assert params["temperature"] == 0.5


def test_openai_to_litellm_params_requires_model():
    with pytest.raises(ValueError, match="model"):
        _openai_to_litellm_params({"messages": [{"role": "user", "content": "x"}]})


def test_openai_to_litellm_params_stop_passthrough():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x"}],
        "stop": ["\n\n"],
    }
    params = _openai_to_litellm_params(body)
    assert params["stop"] == ["\n\n"]


def test_openai_to_litellm_params_drops_stream():
    """stream is silently ignored (non-streaming only)."""
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    params = _openai_to_litellm_params(body)
    assert "stream" not in params


def test_litellm_to_openai_response_basic():
    resp = _fake_litellm_response(text="hello world", tokens_in=4, tokens_out=2)
    body = _litellm_to_openai_response(resp, requested_model="claude-haiku-4-5-20251001")
    assert body["object"] == "chat.completion"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 4
    assert body["usage"]["completion_tokens"] == 2
    assert body["usage"]["total_tokens"] == 6


def test_litellm_to_openai_response_tool_calls():
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"x"}'}}
    ]
    resp = _fake_litellm_response(text="", tool_calls=tool_calls, finish="tool_calls")
    body = _litellm_to_openai_response(resp, requested_model="m")
    ch = body["choices"][0]
    assert ch["finish_reason"] == "tool_calls"
    assert ch["message"]["content"] is None
    assert ch["message"]["tool_calls"][0]["function"]["name"] == "search"


# -- End-to-end route tests --------------------------------------------------


def test_chat_completions_returns_openai_shape(tmp_path):
    """Happy path: 200 with object=chat.completion and choices array."""
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_litellm_response(text="world", tokens_in=5, tokens_out=2)
    with patch("somm_service.proxy.litellm.completion", return_value=fake) as mock_comp:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
            headers={"x-somm-workload": "oai_wl"},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert isinstance(body["choices"], list) and len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 5
    assert body["usage"]["completion_tokens"] == 2
    assert body["usage"]["total_tokens"] == 7

    mock_comp.assert_called_once()
    kwargs = mock_comp.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["max_tokens"] == 32


def test_chat_completions_writes_telemetry_row(tmp_path):
    """Proxy call shows up in calls.sqlite (one ledger)."""
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    write_intel(
        repo,
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        price_in_per_1m=1.0,
        price_out_per_1m=4.0,
        context_window=None,
        capabilities=None,
        source="test_seed",
    )

    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_litellm_response(tokens_in=1_000_000, tokens_out=1_000_000)
    with patch("somm_service.proxy.litellm.completion", return_value=fake):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "p"}],
                "max_tokens": 8,
            },
            headers={"x-somm-workload": "oai_wl"},
        )
    assert r.status_code == 200

    with repo._open() as conn:
        rows = conn.execute(
            "SELECT c.provider, c.model, c.tokens_in, c.tokens_out, c.cost_usd, "
            "       c.outcome, w.name "
            "FROM calls c LEFT JOIN workloads w ON w.id = c.workload_id"
        ).fetchall()
    assert len(rows) == 1
    provider, model, tin, tout, cost, outcome, wlname = rows[0]
    assert provider == "anthropic"
    assert model == "claude-haiku-4-5-20251001"
    assert tin == 1_000_000
    assert tout == 1_000_000
    assert cost == pytest.approx(5.0)
    assert outcome == "ok"
    assert wlname == "oai_wl"


def test_chat_completions_budget_gate_blocks(tmp_path):
    """Over-cap workload returns 429 with OpenAI error shape; litellm not called."""
    cfg = _cfg(tmp_path, fail_closed=True)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="capped", project=cfg.project, budget_cap_usd_daily=0.0)

    app = create_app(cfg)
    client = TestClient(app)

    with patch("somm_service.proxy.litellm.completion") as mock_comp:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "x"}],
            },
            headers={"x-somm-workload": "capped"},
        )

    assert r.status_code == 429
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "rate_limit_error"
    assert "SOMM_BUDGET_EXCEEDED" in body["error"]["message"]
    mock_comp.assert_not_called()

    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    assert n == 0


def test_chat_completions_x_somm_project_header_routes_workload(tmp_path):
    """X-Somm-Project header overrides cfg.project for workload registration."""
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_litellm_response()
    with patch("somm_service.proxy.litellm.completion", return_value=fake):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-somm-workload": "wl-oai", "x-somm-project": "custom-proj"},
        )
    assert r.status_code == 200

    with repo._open() as conn:
        row = conn.execute(
            "SELECT project FROM workloads WHERE name = ?", ("wl-oai",)
        ).fetchone()
    assert row is not None
    assert row[0] == "custom-proj"


def test_chat_completions_missing_model_returns_400(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request_error"


def test_chat_completions_upstream_error_returns_502(tmp_path):
    """litellm.completion raises → 502, telemetry row with outcome=upstream_error."""
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    app = create_app(cfg)
    client = TestClient(app)

    with patch(
        "somm_service.proxy.litellm.completion",
        side_effect=RuntimeError("provider down"),
    ):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers={"x-somm-workload": "error_wl"},
        )

    assert r.status_code == 502
    body = r.json()
    assert body["error"]["type"] == "api_error"

    with repo._open() as conn:
        rows = conn.execute("SELECT outcome FROM calls").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "upstream_error"

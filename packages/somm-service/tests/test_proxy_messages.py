"""Tests for the /v1/messages Anthropic-compatible proxy gateway.

Coverage:
  - request → litellm.completion params mapping
  - litellm response → Anthropic Messages response mapping
  - budget gate fires BEFORE litellm is called (returns 429)
  - successful pass-through writes a telemetry row with the right workload,
    provider, model, tokens, and cost (one ledger)

litellm.completion is mocked throughout; no network and no real provider keys.
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
from somm_service.proxy import (
    _anthropic_to_litellm_params,
    _litellm_to_anthropic_response,
    _provider_from_model,
)
from starlette.testclient import TestClient


def _cfg(tmp_path: Path, *, fail_closed: bool = False) -> Config:
    cfg = Config()
    cfg.project = "proxy-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.budget_fail_closed = fail_closed
    return cfg


def _fake_completion_response(
    *,
    text: str = "hello back",
    tokens_in: int = 7,
    tokens_out: int = 3,
    finish: str = "stop",
    model: str = "claude-haiku-4-5-20251001",
    tool_calls: list | None = None,
) -> SimpleNamespace:
    """Build an OpenAI/litellm-ish ModelResponse stand-in."""
    message = {"role": "assistant", "content": text, "tool_calls": tool_calls or []}
    return SimpleNamespace(
        id="chatcmpl-xyz",
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage={"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    )


# -- Pure mapping unit tests -------------------------------------------------


def test_provider_split_default_to_anthropic():
    assert _provider_from_model("claude-haiku-4-5-20251001") == (
        "anthropic",
        "claude-haiku-4-5-20251001",
    )


def test_provider_split_explicit_prefix():
    assert _provider_from_model("openrouter/anthropic/claude-3.5-sonnet") == (
        "openrouter",
        "anthropic/claude-3.5-sonnet",
    )


def test_anthropic_to_litellm_params_basic():
    body = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.7,
    }
    params = _anthropic_to_litellm_params(body)
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert params["max_tokens"] == 64
    assert params["temperature"] == 0.7


def test_anthropic_to_litellm_params_promotes_system_field():
    """Anthropic's top-level `system` field → leading role=system message."""
    body = {
        "model": "claude-haiku-4-5-20251001",
        "system": "you are concise",
        "messages": [{"role": "user", "content": "ping"}],
    }
    params = _anthropic_to_litellm_params(body)
    assert params["messages"][0] == {"role": "system", "content": "you are concise"}
    assert params["messages"][1]["role"] == "user"


def test_anthropic_to_litellm_params_flattens_system_blocks():
    """`system` may be a list of text blocks — flatten to plain text."""
    body = {
        "model": "claude-haiku-4-5-20251001",
        "system": [
            {"type": "text", "text": "rule 1"},
            {"type": "text", "text": "rule 2"},
        ],
        "messages": [{"role": "user", "content": "x"}],
    }
    params = _anthropic_to_litellm_params(body)
    assert params["messages"][0]["role"] == "system"
    assert params["messages"][0]["content"] == "rule 1\nrule 2"


def test_anthropic_to_litellm_params_requires_model():
    with pytest.raises(ValueError):
        _anthropic_to_litellm_params({"messages": []})


def test_anthropic_to_litellm_params_passes_tools_through():
    body = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "search", "description": "d", "input_schema": {}}],
        "tool_choice": {"type": "auto"},
    }
    params = _anthropic_to_litellm_params(body)
    assert params["tools"][0]["name"] == "search"
    assert params["tool_choice"] == {"type": "auto"}


def test_litellm_to_anthropic_response_text_only():
    resp = _fake_completion_response(text="hello world", tokens_in=4, tokens_out=2)
    body = _litellm_to_anthropic_response(resp, requested_model="claude-haiku-4-5-20251001")
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["content"] == [{"type": "text", "text": "hello world"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 4, "output_tokens": 2}


def test_litellm_to_anthropic_response_max_tokens_finish():
    resp = _fake_completion_response(finish="length")
    body = _litellm_to_anthropic_response(resp, requested_model="m")
    assert body["stop_reason"] == "max_tokens"


def test_litellm_to_anthropic_response_tool_use():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "x"}'},
        }
    ]
    resp = _fake_completion_response(text="", tool_calls=tool_calls, finish="tool_calls")
    body = _litellm_to_anthropic_response(resp, requested_model="m")
    assert body["stop_reason"] == "tool_use"
    # First (and only) content block should be the tool_use, with parsed args.
    assert body["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}}
    ]


def test_litellm_to_anthropic_response_text_and_tool_use():
    """Response with both text AND a tool call → two content blocks in order."""
    tool_calls = [
        {"id": "call_1", "type": "function",
         "function": {"name": "search", "arguments": '{"q": "x"}'}}
    ]
    resp = _fake_completion_response(
        text="Let me search for that.", tool_calls=tool_calls, finish="tool_calls"
    )
    body = _litellm_to_anthropic_response(resp, requested_model="m")
    assert len(body["content"]) == 2
    assert body["content"][0] == {"type": "text", "text": "Let me search for that."}
    assert body["content"][1] == {
        "type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}
    }
    assert body["stop_reason"] == "tool_use"


def test_litellm_to_anthropic_response_multiple_tool_calls():
    """Two parallel tool calls produce two separate tool_use content blocks."""
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}},
        {"id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": '{"id": 42}'}},
    ]
    resp = _fake_completion_response(text="", finish="tool_calls", tool_calls=tool_calls)
    body = _litellm_to_anthropic_response(resp, requested_model="m")
    assert body["stop_reason"] == "tool_use"
    assert len(body["content"]) == 2
    assert body["content"][0] == {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}}
    assert body["content"][1] == {"type": "tool_use", "id": "call_2", "name": "lookup", "input": {"id": 42}}


# -- End-to-end route tests --------------------------------------------------


def test_messages_route_dispatches_litellm_and_returns_anthropic_shape(tmp_path):
    """Happy path: route calls litellm.completion, maps response, returns 200."""
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_completion_response(text="howdy", tokens_in=5, tokens_out=2)
    with patch("somm_service.proxy.litellm.completion", return_value=fake) as mock_comp:
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
            headers={"x-somm-workload": "proxy_wl"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["content"] == [{"type": "text", "text": "howdy"}]
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 2}

    mock_comp.assert_called_once()
    kwargs = mock_comp.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["max_tokens"] == 32


def test_messages_route_writes_telemetry_row(tmp_path):
    """One ledger: a proxy call shows up in calls.sqlite with the right
    workload, provider, model, tokens, and cost — identical to a direct
    somm.llm() call."""
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    # Seed pricing so cost > 0 is observable.
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

    fake = _fake_completion_response(tokens_in=1_000_000, tokens_out=1_000_000)
    with patch("somm_service.proxy.litellm.completion", return_value=fake):
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "p"}],
                "max_tokens": 8,
            },
            headers={"x-somm-workload": "proxy_wl"},
        )
    assert r.status_code == 200

    # Read back the call row directly.
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
    # 1M in @ $1 + 1M out @ $4 = $5.0 exact.
    assert cost == pytest.approx(5.0)
    assert outcome == "ok"
    assert wlname == "proxy_wl"


def test_messages_route_budget_gate_blocks_before_dispatch(tmp_path):
    """Over-cap workload returns 429 + Anthropic error AND litellm is NOT called."""
    cfg = _cfg(tmp_path, fail_closed=True)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="capped", project=cfg.project, budget_cap_usd_daily=0.0)

    app = create_app(cfg)
    client = TestClient(app)

    with patch("somm_service.proxy.litellm.completion") as mock_comp:
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 8,
            },
            headers={"x-somm-workload": "capped"},
        )

    assert r.status_code == 429
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    assert "SOMM_BUDGET_EXCEEDED" in body["error"]["message"]
    # Critical: budget gate must short-circuit BEFORE the upstream call.
    mock_comp.assert_not_called()

    # And no telemetry row was written (a blocked call is not a spend event).
    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    assert n == 0


def test_messages_route_falls_back_to_default_workload(tmp_path):
    """No X-Somm-Workload header → calls land in the catch-all `proxy_default`."""
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_completion_response()
    with patch("somm_service.proxy.litellm.completion", return_value=fake):
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    assert r.status_code == 200

    repo = Repository(cfg.db_path)
    with repo._open() as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM workloads").fetchall()]
    assert "proxy_default" in names


def test_messages_route_rejects_missing_model(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)
    r = client.post("/v1/messages", json={"messages": []})
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_messages_route_respects_x_somm_project_header(tmp_path):
    """X-Somm-Project header overrides cfg.project for workload registration."""
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    app = create_app(cfg)
    client = TestClient(app)

    fake = _fake_completion_response()
    with patch("somm_service.proxy.litellm.completion", return_value=fake):
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-somm-workload": "wl-custom", "x-somm-project": "custom-proj"},
        )
    assert r.status_code == 200

    with repo._open() as conn:
        row = conn.execute(
            "SELECT project FROM workloads WHERE name = ?", ("wl-custom",)
        ).fetchone()
    assert row is not None
    assert row[0] == "custom-proj"

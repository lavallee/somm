"""Tests for the /v1/messages Anthropic-compatible proxy route.

Coverage:
  - Anthropic request → SommLLM call translation (system, tools, tool_choice,
    effort, refused fields)
  - SommResult → Anthropic Messages response translation
  - budget gate fires before any provider call (429)
  - a served call writes one ledger row through the client (one ledger)
  - streaming replays the answer as Anthropic SSE events
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from somm.providers.base import ToolCall
from somm_core.config import Config
from somm_core.models import SommResult
from somm_core.models import ToolCall as CoreToolCall
from somm_core.pricing import write_intel
from somm_core.repository import Repository
from somm_service.app import create_app
from somm_service.proxy import (
    WireError,
    _anthropic_body,
    _anthropic_request,
    split_model,
)
from starlette.testclient import TestClient

MODEL = "claude-haiku-4-5-20251001"


def _cfg(tmp_path: Path, *, fail_closed: bool = False) -> Config:
    cfg = Config()
    cfg.project = "proxy-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.budget_fail_closed = fail_closed
    return cfg


def _auth_headers(app, **extra: str) -> dict[str, str]:
    return {"authorization": f"Bearer {app.state.service_token.value}", **extra}


def _result(text="", tool_calls=(), stop_reason="", raw=None, tokens=(4, 2)) -> SommResult:
    return SommResult(
        text=text, provider="anthropic", model=MODEL, tokens_in=tokens[0], tokens_out=tokens[1],
        latency_ms=1, cost_usd=0.0, call_id="00000000-0000-0000-0000-000000000001",
        tool_calls=[CoreToolCall(*tc) for tc in tool_calls], stop_reason=stop_reason, raw=raw,
    )


# -- Pure translation unit tests ---------------------------------------------


def test_split_model_defaults_explicit_prefix_and_auto():
    assert split_model(MODEL, "anthropic") == ("anthropic", MODEL)
    assert split_model("openrouter/anthropic/claude-3.5-sonnet", "anthropic") == (
        "openrouter",
        "anthropic/claude-3.5-sonnet",
    )
    assert split_model("somm/auto", "anthropic") == (None, "somm/auto")


def test_anthropic_request_basic():
    call, _ = _anthropic_request({
        "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64, "temperature": 0.7,
    })
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.7


def test_anthropic_request_promotes_and_flattens_system():
    assert _anthropic_request({"model": MODEL, "system": "you are concise", "messages": []})[0][
        "system"] == "you are concise"
    call, _ = _anthropic_request({
        "model": MODEL,
        "system": [{"type": "text", "text": "rule 1"}, {"type": "text", "text": "rule 2"}],
        "messages": [{"role": "user", "content": "x"}],
    })
    assert call["system"] == "rule 1\nrule 2"
    assert call["messages"] == [{"role": "user", "content": "x"}]


def test_anthropic_request_maps_tools_choice_and_effort_to_the_client():
    call, ignored = _anthropic_request({
        "model": MODEL, "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "search", "description": "d", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "search"},
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
        "metadata": {"user_id": "synthetic"},
    })
    assert call["tools"] == [{"name": "search", "description": "d",
                              "parameters": {"type": "object"}}]
    assert call["tool_choice"] == {"type": "tool", "name": "search"}
    assert call["reasoning_effort"] == "low"
    assert ignored == ["metadata"]


def test_anthropic_request_accepts_and_names_routine_fields_it_cannot_apply():
    call, ignored = _anthropic_request({
        "model": MODEL, "messages": [], "stop_sequences": ["</result>"], "top_p": 0.9,
        "top_k": 5, "thinking": {"type": "disabled"},
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    })
    assert call["tool_choice"] == "auto"
    assert ignored == ["stop_sequences", "top_k", "top_p", "thinking",
                       "tool_choice.disable_parallel_tool_use"]


@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
    {"mcp_servers": []},
])
def test_anthropic_request_refuses_what_would_change_the_answer(extra):
    with pytest.raises(WireError):
        _anthropic_request({"model": MODEL, "messages": [], **extra})


def test_anthropic_body_text_only():
    body = _anthropic_body(_result(text="hello world", stop_reason="end_turn"), MODEL)
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["model"] == MODEL
    assert body["content"] == [{"type": "text", "text": "hello world"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 4, "output_tokens": 2}


def test_anthropic_body_max_tokens_stop():
    assert _anthropic_body(_result(text="x", stop_reason="max_tokens"), "m")["stop_reason"] == "max_tokens"


def test_anthropic_body_text_and_multiple_tool_uses_in_order():
    body = _anthropic_body(_result(
        text="Let me search for that.",
        tool_calls=[("call_1", "search", {"q": "x"}), ("call_2", "lookup", {"id": 42})],
        stop_reason="end_turn",  # some backends report a stop alongside tool calls
    ), "m")
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "text", "text": "Let me search for that."},
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}},
        {"type": "tool_use", "id": "call_2", "name": "lookup", "input": {"id": 42}},
    ]


def test_anthropic_body_keeps_signed_thinking_for_the_next_tool_turn():
    thinking = {"type": "thinking", "thinking": "plan", "signature": "synthetic-signature"}
    body = _anthropic_body(_result(
        tool_calls=[("call_1", "search", {})],
        raw={"content": [thinking, {"type": "tool_use", "id": "call_1", "name": "search", "input": {}}]},
    ), "m")
    assert body["content"][0] == thinking
    assert body["content"][1]["type"] == "tool_use"


# -- End-to-end route tests --------------------------------------------------


def test_messages_route_dispatches_through_the_client_and_returns_anthropic_shape(
    tmp_path, fake_provider
):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="proxy_wl", project=cfg.project)
    fake = fake_provider("anthropic", text="howdy", tokens_in=5, tokens_out=2)
    app = create_app(cfg, providers=[fake])
    client = TestClient(app)

    r = client.post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32},
        headers=_auth_headers(app, **{"x-somm-workload": "proxy_wl"}),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == MODEL
    assert body["content"] == [{"type": "text", "text": "howdy"}]
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert body["id"] == "msg_" + r.headers["x-somm-call-id"].replace("-", "")

    assert len(fake.requests) == 1
    sent = fake.requests[0]
    assert sent.model == MODEL
    assert sent.messages == [{"role": "user", "content": "hello"}]
    assert sent.max_tokens == 32


def test_messages_route_writes_telemetry_row(tmp_path, fake_provider, ledger):
    """One ledger: the row is written by the client, with workload, provider,
    model, tokens, cost and cost provenance like any library call."""
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="proxy_wl", project=cfg.project)
    write_intel(repo, provider="anthropic", model=MODEL, price_in_per_1m=1.0, price_out_per_1m=4.0,
                context_window=None, capabilities=None, source="test_seed")
    app = create_app(cfg, providers=[fake_provider(tokens_in=1_000_000, tokens_out=1_000_000)])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "p"}], "max_tokens": 8},
        headers=_auth_headers(app, **{"x-somm-workload": "proxy_wl"}),
    )
    assert r.status_code == 200

    rows = ledger(app,
        "SELECT c.id, c.provider, c.model, c.tokens_in, c.tokens_out, c.cost_usd, c.outcome, "
        "w.name, c.cost_basis, c.routing_json FROM calls c "
        "LEFT JOIN workloads w ON w.id = c.workload_id")
    assert len(rows) == 1
    call_id, provider, model, tin, tout, cost, outcome, wlname, basis, routing = rows[0]
    assert call_id == r.headers["x-somm-call-id"]
    assert (provider, model, tin, tout, outcome, wlname) == (
        "anthropic", MODEL, 1_000_000, 1_000_000, "ok", "proxy_wl")
    assert cost == pytest.approx(5.0)  # 1M in @ $1 + 1M out @ $4
    assert basis == "computed"
    assert json.loads(routing)["route_source"] == "pin"


def test_messages_route_budget_gate_blocks_before_dispatch(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path, fail_closed=True)
    Repository(cfg.db_path).register_workload(name="capped", project=cfg.project,
                                              budget_cap_usd_daily=0.0)
    fake = fake_provider()
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}], "max_tokens": 8},
        headers=_auth_headers(app, **{"x-somm-workload": "capped"}),
    )
    assert r.status_code == 429
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    assert "SOMM_BUDGET_EXCEEDED" in body["error"]["message"]
    assert r.headers["x-should-retry"] == "false"
    assert fake.requests == []
    assert ledger(app, "SELECT COUNT(*) FROM calls")[0][0] == 0


def test_messages_route_falls_back_to_default_workload(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    app = create_app(cfg, providers=[fake_provider()])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
        headers=_auth_headers(app),
    )
    assert r.status_code == 200
    assert app.state.repo.workload_by_name("proxy_default", cfg.project) is not None


def test_messages_route_rejects_missing_model(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    r = TestClient(app).post("/v1/messages", json={"messages": []}, headers=_auth_headers(app))
    assert r.status_code == 400
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_messages_route_names_ignored_fields_in_a_header(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    fake = fake_provider()
    app = create_app(cfg, providers=[fake])
    client = TestClient(app)
    ok = client.post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}],
              "stop_sequences": ["</r>"], "top_p": 0.9},
        headers=_auth_headers(app),
    )
    assert ok.status_code == 200
    assert ok.headers["x-somm-ignored"] == "stop_sequences,top_p"
    plain = client.post("/v1/messages", json={"model": MODEL, "messages": []},
                        headers=_auth_headers(app))
    assert "x-somm-ignored" not in plain.headers
    refused = client.post(
        "/v1/messages",
        json={"model": MODEL, "messages": [], "container": "c1"},
        headers=_auth_headers(app),
    )
    assert refused.status_code == 400
    assert "container" in refused.json()["error"]["message"]
    assert len(fake.requests) == 2


def test_messages_route_rejects_non_dict_body(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    r = TestClient(app).post(
        "/v1/messages", content=b"[1, 2, 3]",
        headers=_auth_headers(app, **{"content-type": "application/json"}),
    )
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "JSON object" in body["error"]["message"]


def test_messages_route_upstream_provider_error_returns_502(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="error_wl", project=cfg.project)
    app = create_app(cfg, providers=[fake_provider(error=RuntimeError("provider down"))])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        headers=_auth_headers(app, **{"x-somm-workload": "error_wl"}),
    )
    assert r.status_code == 502
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert r.headers["x-should-retry"] == "true"
    rows = ledger(app, "SELECT id, outcome, error_kind FROM calls")
    assert rows == [(r.headers["x-somm-call-id"], "upstream_error", "RuntimeError")]


def test_messages_route_unconfigured_provider_pin_is_a_400(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    app = create_app(cfg, providers=[fake_provider("anthropic")])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": "nosuch/model", "messages": [{"role": "user", "content": "x"}]},
        headers=_auth_headers(app),
    )
    assert r.status_code == 400
    assert "not configured" in r.json()["error"]["message"]


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for frame in text.strip().split("\n\n"):
        lines = frame.split("\n")
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


def test_messages_route_streams_sse_with_tool_use_and_usage(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="stream_wl", project=cfg.project)
    fake = fake_provider(text="hello", tokens_in=5, tokens_out=2,
                         tool_calls=[ToolCall("toolu_1", "search", {"q": "x"})])
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}],
              "max_tokens": 8, "stream": True},
        headers=_auth_headers(app, **{"x-somm-workload": "stream_wl"}),
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    names = [name for name, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert names.count("content_block_start") == 2 == names.count("content_block_stop")
    deltas = [data["delta"] for name, data in events if name == "content_block_delta"]
    assert deltas == [{"type": "text_delta", "text": "hello"},
                      {"type": "input_json_delta", "partial_json": '{"q": "x"}'}]
    tool_start = [d for n, d in events if n == "content_block_start"][1]["content_block"]
    assert tool_start == {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}
    message_delta = next(d for n, d in events if n == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"] == {"output_tokens": 2}
    assert events[0][1]["message"]["usage"]["input_tokens"] == 5

    assert ledger(app, "SELECT outcome, tokens_in, tokens_out FROM calls") == [("ok", 5, 2)]


def test_messages_route_streaming_upstream_error_is_an_error_response(tmp_path, fake_provider, ledger):
    """The answer is complete before the stream opens, so a failure arrives as
    a normal error response the SDK raises on, not a 200 with an error event."""
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="stream_wl", project=cfg.project)
    app = create_app(cfg, providers=[fake_provider(error=RuntimeError("provider down"))])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "ping"}], "stream": True},
        headers=_auth_headers(app, **{"x-somm-workload": "stream_wl"}),
    )
    assert r.status_code == 502
    assert "provider down" in r.json()["error"]["message"]
    assert ledger(app, "SELECT outcome, error_kind FROM calls") == [("upstream_error", "RuntimeError")]


def test_messages_route_respects_x_somm_project_header(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="wl-custom", project="custom-proj")
    app = create_app(cfg, providers=[fake_provider()])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "wl-custom", "x-somm-project": "custom-proj"}),
    )
    assert r.status_code == 200
    assert ledger(app, "SELECT project FROM calls") == [("custom-proj",)]


def test_messages_route_rejects_unknown_explicit_workload_before_dispatch(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    fake = fake_provider()
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        headers=_auth_headers(app, **{"x-somm-workload": "new-uncapped"}),
    )
    assert r.status_code == 403
    assert "pre-register" in r.json()["error"]["message"]
    assert fake.requests == []


def test_messages_route_default_workload_gets_default_cap(tmp_path, fake_provider):
    cfg = _cfg(tmp_path, fail_closed=True)
    cfg.budget_default_cap_usd_daily = 1.25
    app = create_app(cfg, providers=[fake_provider()])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
        headers=_auth_headers(app),
    )
    assert r.status_code == 200
    wl = app.state.repo.workload_by_name("proxy_default", cfg.project)
    assert wl is not None
    assert wl.budget_cap_usd_daily == pytest.approx(1.25)


def test_messages_route_rejects_default_workload_when_fail_closed_without_default_cap(
    tmp_path, fake_provider
):
    cfg = _cfg(tmp_path, fail_closed=True)
    fake = fake_provider()
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
        headers=_auth_headers(app),
    )
    assert r.status_code == 403
    assert fake.requests == []


def test_messages_route_rejects_oversized_body_before_dispatch(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    cfg.service_proxy_max_body_bytes = 8
    fake = fake_provider()
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/messages", content=b'{"model":"x","messages":[]}',
        headers=_auth_headers(app, **{"content-type": "application/json", "content-length": "999"}),
    )
    assert r.status_code == 413
    assert fake.requests == []


def test_messages_route_slow_provider_does_not_block_status_api(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="proxy_wl", project=cfg.project)
    entered = threading.Event()

    class Slow(fake_provider):
        def generate(self, request):
            entered.set()
            time.sleep(0.25)
            return super().generate(request)

    app = create_app(cfg, providers=[Slow()])
    client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))

    def post_message():
        return client.post(
            "/v1/messages",
            json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
            headers=_auth_headers(app, **{"x-somm-workload": "proxy_wl"}),
        )

    thread = threading.Thread(target=post_message)
    thread.start()
    assert entered.wait(timeout=5.0)
    started = time.perf_counter()
    status = client.get("/api/status", headers={"x-somm-local": "1", "sec-fetch-site": "same-origin"})
    elapsed = time.perf_counter() - started
    thread.join(timeout=5.0)

    assert status.status_code == 200
    assert elapsed < 0.20


def test_messages_route_blocks_cross_site_without_token(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    r = TestClient(app).post(
        "/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
        headers={"origin": "https://evil.example"},
    )
    assert r.status_code == 403
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"
    assert "Authorization: Bearer <token>" in body["error"]["message"]


def test_messages_route_accepts_bearer_token_before_downstream_validation(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    r = TestClient(app).post("/v1/messages", json={"messages": []}, headers=_auth_headers(app))
    assert r.status_code == 400


def test_messages_route_requires_json_content_type_with_token(tmp_path):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)
    r = TestClient(app).post(
        "/v1/messages", content=b'{"messages": []}',
        headers=_auth_headers(app, **{"content-type": "text/plain"}),
    )
    assert r.status_code == 415
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"

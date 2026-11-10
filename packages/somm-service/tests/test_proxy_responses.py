"""/v1/responses (OpenAI Responses, common subset) and SDK wire conformance.

The OpenAI and Anthropic-shaped SDK checks drive the proxy through the real
OpenAI Python SDK over the in-process TestClient, so the SDK's own parsers
judge the wire format, streamed or not.
"""

from __future__ import annotations

from pathlib import Path

import openai
import pytest
from somm.providers.base import ToolCall
from somm_core.config import Config
from somm_service.app import create_app
from starlette.testclient import TestClient


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "proxy-responses-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _sdk(app) -> openai.OpenAI:
    return openai.OpenAI(api_key=app.state.service_token.value, base_url="http://testserver/v1",
                         http_client=TestClient(app), max_retries=0)


def test_responses_string_input_with_instructions(tmp_path, fake_provider, ledger):
    fake = fake_provider("openai", text="hi there", tokens_in=4, tokens_out=2)
    app = create_app(_cfg(tmp_path), providers=[fake])
    response = _sdk(app).responses.create(model="gpt-4o-mini", input="hello",
                                          instructions="be brief", max_output_tokens=64)
    assert response.status == "completed"
    assert response.output_text == "hi there"
    assert response.model == "gpt-4o-mini"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (4, 2)
    sent = fake.requests[0]
    assert sent.system == "be brief"
    assert sent.messages == [{"role": "user", "content": "hello"}]
    assert sent.max_tokens == 64
    assert ledger(app, "SELECT provider, outcome FROM calls") == [("openai", "ok")]


def test_responses_tool_round_trip(tmp_path, fake_provider):
    fake = fake_provider("openai", text="", tool_calls=[ToolCall("call_1", "lookup", {"id": 42})])
    app = create_app(_cfg(tmp_path), providers=[fake])
    client = _sdk(app)
    tools = [{"type": "function", "name": "lookup", "description": "find",
              "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}]
    first = client.responses.create(model="gpt-4o-mini", input="find 42", tools=tools,
                                    tool_choice="auto", reasoning={"effort": "low"})
    (call,) = first.output
    assert call.type == "function_call"
    assert (call.call_id, call.name, call.arguments) == ("call_1", "lookup", '{"id": 42}')
    sent = fake.requests[0]
    assert sent.tools == [{"name": "lookup", "description": "find", "parameters": tools[0]["parameters"]}]
    assert sent.tool_choice == "auto" and sent.reasoning_effort == "low"

    fake.text, fake.tool_calls = "found it", []
    second = client.responses.create(model="gpt-4o-mini", tools=tools, input=[
        {"role": "user", "content": "find 42"},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"id": 42}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "the answer"},
    ])
    assert second.output_text == "found it"
    assert fake.requests[1].messages == [
        {"role": "user", "content": "find 42"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup",
                                           "input": {"id": 42}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                      "content": "the answer"}]},
    ]


def test_responses_stream_events_parse_in_the_sdk(tmp_path, fake_provider):
    fake = fake_provider("openai", text="streamed", tool_calls=[ToolCall("call_9", "f", {})])
    app = create_app(_cfg(tmp_path), providers=[fake])
    events = list(_sdk(app).responses.create(model="gpt-4o-mini", input="x", stream=True))
    kinds = [e.type for e in events]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert [e.sequence_number for e in events] == list(range(len(events)))
    assert [e.delta for e in events if e.type == "response.output_text.delta"] == ["streamed"]
    done = next(e for e in events if e.type == "response.function_call_arguments.done")
    assert (done.name, done.arguments) == ("f", "{}")
    assert events[-1].response.output_text == "streamed"


@pytest.mark.parametrize("extra", [
    {"previous_response_id": "resp_1"},
    {"tools": [{"type": "web_search"}]},
    {"input": [{"type": "reasoning", "summary": []}]},
])
def test_responses_rejects_what_it_does_not_support(tmp_path, fake_provider, extra):
    fake = fake_provider("openai")
    app = create_app(_cfg(tmp_path), providers=[fake])
    with pytest.raises(openai.BadRequestError):
        _sdk(app).responses.create(**{"model": "gpt-4o-mini", "input": "x", **extra})
    assert fake.requests == []


def test_responses_accepts_routine_fields_and_names_them(tmp_path, fake_provider):
    app = create_app(_cfg(tmp_path), providers=[fake_provider("openai")])
    raw = _sdk(app).responses.with_raw_response.create(
        model="gpt-4o-mini", input="x", truncation="auto", parallel_tool_calls=True, top_p=1,
        store=False, reasoning={"effort": "low", "summary": "auto"})
    assert raw.headers["x-somm-ignored"] == (
        "parallel_tool_calls,store,top_p,truncation,reasoning.summary")
    assert raw.parse().output_text == "hello back"


def test_responses_needs_the_token(tmp_path):
    app = create_app(_cfg(tmp_path))
    r = TestClient(app).post("/v1/responses", json={"model": "gpt-4o-mini", "input": "x"})
    assert r.status_code == 403
    assert r.json()["error"]["type"] == "authentication_error"


def test_chat_stream_parses_in_the_sdk(tmp_path, fake_provider):
    fake = fake_provider("openai", text="hello", tool_calls=[ToolCall("call_1", "f", {"a": 1})])
    app = create_app(_cfg(tmp_path), providers=[fake])
    chunks = list(_sdk(app).chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], stream=True,
        stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    calls = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
    assert text == "hello"
    assert [(tc.id, tc.function.name, tc.function.arguments) for tc in calls] == [
        ("call_1", "f", '{"a": 1}')]
    assert chunks[-1].usage.completion_tokens == 3

"""Adapter streaming against recorded-shape wire bodies (no network).

Each adapter sends what its generate() sends, yields deltas, and puts the
whole answer (usage, tool calls, stop reason) on its final chunk; a stream
cut off before the wire's completion marker raises.
"""

from __future__ import annotations

import json

import httpx
import pytest
from somm.errors import SommBadRequest, SommTransientError
from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import SommRequest, ToolCallDelta
from somm.providers.ollama import OllamaProvider
from somm.providers.openai import OpenAIProvider
from somm.providers.openrouter import OpenRouterProvider


def _serve(monkeypatch, handler):
    sent: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), "body": json.loads(request.content or b"{}")})
        return handler(request, sent[-1]["body"])

    class _Client(httpx.Client):
        def __init__(self, *a, **kw):
            kw.pop("transport", None)
            super().__init__(*a, transport=httpx.MockTransport(wrapped), **kw)

    monkeypatch.setattr(httpx, "Client", _Client)
    return sent


def _sse(*events, done=True) -> str:
    out = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return out + ("data: [DONE]\n\n" if done else "")


_OPENAI_EVENTS = (
    {"id": "c1", "model": "gpt-test", "choices": [{"delta": {"reasoning_content": "think"}}]},
    {"choices": [{"delta": {"content": "Hel"}}]},
    {"choices": [{"delta": {"content": "lo"}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "t1", "function": {"name": "lookup", "arguments": '{"q"'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": ': 1}'}}]}}]},
    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5, "cost": 0.002}},
)


def test_openai_compat_stream_deltas_and_final_answer(monkeypatch):
    sent = _serve(monkeypatch, lambda req, body: httpx.Response(200, text=_sse(*_OPENAI_EVENTS)))
    p = OpenAIProvider(api_key="k", default_model="gpt-test")
    msgs = [{"role": "user", "content": "m"}]
    chunks = list(p.stream(SommRequest(
        prompt="", messages=msgs, system="sys", tools=[{"name": "lookup", "parameters": {}}],
        tool_choice="auto", reasoning_effort="low",
    )))
    body = sent[0]["body"]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1:] == msgs
    assert body["tools"][0]["function"]["name"] == "lookup" and body["tool_choice"] == "auto"
    assert body["reasoning_effort"] == "low"
    assert [c.reasoning for c in chunks if c.reasoning] == ["think"]
    assert "".join(c.text for c in chunks) == "Hello"
    assert [d for c in chunks for d in c.tool_calls] == [
        ToolCallDelta(0, id="t1", name="lookup", arguments='{"q"'),
        ToolCallDelta(0, arguments=": 1}"),
    ]
    final = chunks[-1]
    assert final.done and final.response is not None
    r = final.response
    assert (r.text, r.tokens_in, r.tokens_out, r.stop_reason) == ("Hello", 12, 5, "tool_use")
    assert [(t.id, t.name, t.arguments) for t in r.tool_calls] == [("t1", "lookup", {"q": 1})]
    assert r.reasoning_content == "think"
    assert r.cost_usd == pytest.approx(0.002)
    assert r.raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"q": 1}'


def test_openai_compat_stream_cut_off_raises(monkeypatch):
    body = _sse({"choices": [{"delta": {"content": "par"}}]}, done=False)
    _serve(monkeypatch, lambda req, b: httpx.Response(200, text=body))
    p = OpenAIProvider(api_key="k", default_model="gpt-test")
    got = []
    with pytest.raises(SommTransientError, match="ended before"):
        for c in p.stream(SommRequest(prompt="hi")):
            got.append(c.text)
    assert got == ["par"]


def test_openai_compat_finish_reason_without_done_is_complete(monkeypatch):
    body = _sse({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}, done=False)
    _serve(monkeypatch, lambda req, b: httpx.Response(200, text=body))
    chunks = list(OpenAIProvider(api_key="k", default_model="m").stream(SommRequest(prompt="hi")))
    assert chunks[-1].done and chunks[-1].response.stop_reason == "end_turn"


def test_openai_compat_drops_stream_options_an_endpoint_rejects(monkeypatch):
    def handler(req, body):
        if "stream_options" in body:
            return httpx.Response(400, text="Unrecognized request argument: stream_options")
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "ok"}}]}))

    sent = _serve(monkeypatch, handler)
    p = OpenAIProvider(api_key="k", default_model="m")
    assert "".join(c.text for c in p.stream(SommRequest(prompt="hi"))) == "ok"
    assert len(sent) == 2 and "stream_options" not in sent[1]["body"]
    assert p.stream_include_usage is False


def test_openrouter_unpinned_stream_walks_the_roster(monkeypatch):
    def handler(req, body):
        if body["model"] == "first:free":
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "hi"}}]}))

    sent = _serve(monkeypatch, handler)
    p = OpenRouterProvider(api_key="k", roster=["first:free", "second:free"])
    chunks = list(p.stream(SommRequest(prompt="hi")))
    assert [s["body"]["model"] for s in sent] == ["first:free", "second:free"]
    assert chunks[-1].done and chunks[-1].model == "second:free"
    assert sent[1]["body"]["usage"] == {"include": True}


def test_ollama_stream_sends_messages_and_reports_usage(monkeypatch):
    lines = [
        {"message": {"content": "Hi", "thinking": "hmm"}, "done": False},
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "lookup", "arguments": {"q": 1}}}]}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 9, "eval_count": 4},
    ]
    body = "".join(json.dumps(x) + "\n" for x in lines)
    sent = _serve(monkeypatch, lambda req, b: httpx.Response(200, text=body))
    p = OllamaProvider(base_url="http://127.0.0.1:1", default_model="qwen")
    msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"}]
    chunks = list(p.stream(SommRequest(prompt="", messages=msgs, system="sys",
                                       tools=[{"name": "lookup", "parameters": {}}],
                                       response_format={"type": "json_object"})))
    payload = sent[0]["body"]
    assert [m["content"] for m in payload["messages"]] == ["sys", "first", "a", "second"]
    assert payload["tools"][0]["function"]["name"] == "lookup" and payload["format"] == "json"
    assert [c.reasoning for c in chunks if c.reasoning] == ["hmm"]
    assert [d.name for c in chunks for d in c.tool_calls] == ["lookup"]
    r = chunks[-1].response
    assert (r.tokens_in, r.tokens_out, r.stop_reason, r.text) == (9, 4, "tool_use", "Hi")
    assert [(t.name, t.arguments) for t in r.tool_calls] == [("lookup", {"q": 1})]


def test_ollama_stream_cut_off_raises(monkeypatch):
    body = json.dumps({"message": {"content": "par"}, "done": False}) + "\n"
    _serve(monkeypatch, lambda req, b: httpx.Response(200, text=body))
    with pytest.raises(SommTransientError, match="ended before done"):
        list(OllamaProvider(base_url="http://127.0.0.1:1").stream(SommRequest(prompt="hi")))


def test_ollama_stream_refuses_tool_choice_like_generate():
    with pytest.raises(SommBadRequest):
        list(OllamaProvider().stream(SommRequest(prompt="hi", tool_choice="any")))


def _anthropic_sse(*events) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_anthropic_stream_is_native_and_assembles_the_message(monkeypatch):
    events = (
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-haiku-5-5",
                                              "usage": {"input_tokens": 20, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "plan"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "On it"}},
        {"type": "content_block_start", "index": 2,
         "content_block": {"type": "tool_use", "id": "tu1", "name": "lookup", "input": {}}},
        {"type": "content_block_delta", "index": 2,
         "delta": {"type": "input_json_delta", "partial_json": '{"q": 2}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 30}},
        {"type": "message_stop"},
    )
    sent = _serve(monkeypatch, lambda req, b: httpx.Response(200, text=_anthropic_sse(*events)))
    p = AnthropicProvider(api_key="k")
    chunks = list(p.stream(SommRequest(prompt="hi", tools=[{"name": "lookup", "parameters": {}}])))
    assert sent[0]["body"]["stream"] is True
    assert "".join(c.text for c in chunks) == "On it"
    assert [c.reasoning for c in chunks if c.reasoning] == ["plan"]
    assert [d for c in chunks for d in c.tool_calls] == [
        ToolCallDelta(0, id="tu1", name="lookup"), ToolCallDelta(0, arguments='{"q": 2}')]
    r = chunks[-1].response
    assert (r.text, r.tokens_in, r.tokens_out, r.stop_reason) == ("On it", 20, 30, "tool_use")
    assert [(t.id, t.arguments) for t in r.tool_calls] == [("tu1", {"q": 2})]
    assert r.raw["content"][0] == {"type": "thinking", "thinking": "plan", "signature": "sig"}
    assert r.raw["content"][2]["input"] == {"q": 2}


def test_anthropic_stream_cut_off_raises(monkeypatch):
    events = ({"type": "message_start", "message": {"model": "m", "usage": {}}},
              {"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "par"}})
    _serve(monkeypatch, lambda req, b: httpx.Response(200, text=_anthropic_sse(*events)))
    with pytest.raises(SommTransientError, match="message_stop"):
        list(AnthropicProvider(api_key="k").stream(SommRequest(prompt="hi")))


def test_a_cut_off_stream_is_not_recorded_ok(monkeypatch, tmp_path):
    import sqlite3

    from somm import SommLLM
    from somm_core.config import Config

    body = _sse({"choices": [{"delta": {"content": "par"}}]}, done=False)
    _serve(monkeypatch, lambda req, b: httpx.Response(200, text=body))
    cfg = Config()
    cfg.project = "cut"
    cfg.db_dir = tmp_path / ".somm"
    llm = SommLLM(config=cfg, providers=[OpenAIProvider(api_key="k", default_model="m")],
                  on_error=lambda _: None)
    with pytest.raises(SommTransientError):
        list(llm.stream("hi", workload="w"))
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT outcome, provider FROM calls").fetchone())
    assert row == {"outcome": "upstream_error", "provider": "openai"}


def test_stream_records_the_providers_usage(monkeypatch, tmp_path):
    import sqlite3

    from somm import SommLLM
    from somm_core.config import Config

    _serve(monkeypatch, lambda req, b: httpx.Response(200, text=_sse(*_OPENAI_EVENTS)))
    cfg = Config()
    cfg.project = "usage"
    cfg.db_dir = tmp_path / ".somm"
    llm = SommLLM(config=cfg, providers=[OpenAIProvider(api_key="k", default_model="gpt-test")])
    assert "".join(llm.stream("hi", workload="w")) == "Hello"
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT tokens_in, tokens_out, routing_json FROM calls").fetchone())
    assert (row["tokens_in"], row["tokens_out"]) == (12, 5)
    assert json.loads(row["routing_json"])["stream"] == {"usage": "provider"}

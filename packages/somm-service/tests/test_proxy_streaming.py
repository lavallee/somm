"""The proxy's streaming routes forward the client's events as they arrive.

A provider that streams in pieces reaches the caller in pieces (the first
bytes leave before the provider finishes); a failure after the first event
is the wire's in-stream error, recorded on the ledger.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import openai
from somm.errors import SommTransientError
from somm.providers.base import ProviderHealth, SommChunk, SommResponse, ToolCallDelta
from somm_core.config import Config
from somm_service.app import create_app
from starlette.testclient import TestClient


class PieceProvider:
    """Streams text pieces, waiting on ``gate`` after the first one."""

    def __init__(self, name, pieces, *, fail=None, gate=None, tools=()):
        self.name = name
        self.default_model = "m"
        self.pieces, self.fail, self.gate, self.tools = list(pieces), fail, gate, list(tools)

    def generate(self, request):  # pragma: no cover - streams only
        raise AssertionError("streaming route must stream")

    def stream(self, request):
        for i, piece in enumerate(self.pieces):
            yield SommChunk(text=piece)
            if i == 0 and self.gate is not None:
                assert self.gate.wait(5)
        if self.tools:
            yield SommChunk(text="", tool_calls=self.tools)
        if self.fail is not None:
            raise self.fail
        yield SommChunk(text="", done=True, model="m", response=SommResponse(
            text="".join(self.pieces), model="m", tokens_in=11, tokens_out=4, latency_ms=1))

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "proxy-streaming-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _headers(app) -> dict:
    return {"authorization": f"Bearer {app.state.service_token.value}"}


def _ledger(app, sql):
    for client in app.state.llm_clients.values():
        client._writer.flush()
    with app.state.repo._open() as conn:
        return conn.execute(sql).fetchall()


def test_first_piece_leaves_before_the_provider_finishes(tmp_path):
    """Driven over raw ASGI: the test client buffers whole responses. The
    provider blocks after its first piece until that piece has been sent."""
    import asyncio

    gate = threading.Event()
    app = create_app(_cfg(tmp_path), providers=[PieceProvider("openai", ["Hel", "lo"], gate=gate)])
    body = json.dumps({"model": "gpt-4o-mini", "stream": True,
                       "messages": [{"role": "user", "content": "x"}]}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
             "raw_path": b"/v1/chat/completions", "query_string": b"",
             "root_path": "", "client": ("127.0.0.1", 1), "server": ("testserver", 80),
             "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode()),
                         *[(k.encode(), v.encode()) for k, v in _headers(app).items()]]}
    sent: list[bytes] = []
    first_piece_before_gate = []

    async def run():
        requested = [False]
        finished = asyncio.Event()

        async def receive():
            if not requested[0]:
                requested[0] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                sent.append(message.get("body", b""))
                if b'"Hel"' in sent[-1]:
                    first_piece_before_gate.append(not gate.is_set())
                    gate.set()
                if not message.get("more_body", False):
                    finished.set()

        await asyncio.wait_for(app(scope, receive, send), 20)

    with TestClient(app):  # startup and shutdown
        asyncio.run(run())
    assert first_piece_before_gate == [True]
    assert b"data: [DONE]" in b"".join(sent) and b'"lo"' in b"".join(sent)
    assert _ledger(app, "SELECT outcome, tokens_in, tokens_out FROM calls") == [("ok", 11, 4)]


def test_chat_stream_tool_call_fragments_parse_in_the_sdk(tmp_path):
    tools = [ToolCallDelta(0, id="call_1", name="f", arguments='{"a"'),
             ToolCallDelta(0, arguments=": 1}")]
    app = create_app(_cfg(tmp_path), providers=[PieceProvider("openai", ["ok"], tools=tools)])
    sdk = openai.OpenAI(api_key=app.state.service_token.value, base_url="http://testserver/v1",
                        http_client=TestClient(app), max_retries=0)
    chunks = list(sdk.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], stream=True,
        stream_options={"include_usage": True}))
    calls = [tc for c in chunks if c.choices for tc in (c.choices[0].delta.tool_calls or [])]
    assert calls[0].id == "call_1" and calls[0].function.name == "f"
    assert "".join(tc.function.arguments for tc in calls) == '{"a": 1}'
    assert chunks[-1].usage.prompt_tokens == 11


def test_failure_after_the_first_event_is_an_in_stream_error(tmp_path):
    app = create_app(_cfg(tmp_path), providers=[
        PieceProvider("anthropic", ["par"], fail=SommTransientError("connection reset"))])
    r = TestClient(app).post("/v1/messages", headers=_headers(app), json={
        "model": "claude-x", "max_tokens": 8, "stream": True,
        "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    frames = [f.split("\n") for f in r.text.strip().split("\n\n")]
    names = [f[0].removeprefix("event: ") for f in frames]
    assert names[0] == "message_start" and names[-1] == "error"
    error = json.loads(frames[-1][1].removeprefix("data: "))["error"]
    assert "connection reset" in error["message"]
    assert _ledger(app, "SELECT outcome, error_kind FROM calls") == [
        ("upstream_error", "SommTransientError")]


def test_responses_stream_failure_after_text_is_an_error_event(tmp_path):
    app = create_app(_cfg(tmp_path), providers=[
        PieceProvider("openai", ["par"], fail=SommTransientError("cut"))])
    r = TestClient(app).post("/v1/responses", headers=_headers(app), json={
        "model": "gpt-4o-mini", "input": "x", "stream": True})
    kinds = [json.loads(f.split("\n")[1].removeprefix("data: "))["type"]
             for f in r.text.strip().split("\n\n")]
    assert kinds[0] == "response.created" and kinds[-1] == "error"
    assert "response.output_text.delta" in kinds

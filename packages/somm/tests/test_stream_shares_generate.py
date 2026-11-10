"""stream() is generate()'s call path with a different attempt.

Workload policy, pins, somm/auto, learned overrides, arguments, fallback
before the first chunk (and not after it), the route record and usage all
behave as they do for generate; ``events=True`` exposes tool calls, reasoning
and the recorded result.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from somm import SommLLM
from somm.errors import SommTransientError
from somm.events import ReasoningDelta, StreamDone, StreamStart, TextDelta, ToolCallDelta
from somm.providers.base import ProviderHealth, SommChunk, SommResponse, ToolCall
from somm_core.config import Config


class Streamer:
    """Streams scripted chunks; ``final`` rides on the done chunk."""

    def __init__(self, name, chunks=("ok",), *, fail=None, fail_after=None, final=None,
                 reasoning=(), tool_deltas=(), done=True):
        self.name = name
        self.default_model = f"{name}-m"
        self.chunks = list(chunks)
        self.fail = fail
        self.fail_after = fail_after
        self.final = final
        self.reasoning = list(reasoning)
        self.tool_deltas = list(tool_deltas)
        self.done = done
        self.requests = []

    def generate(self, request):  # pragma: no cover - streams never call generate
        raise AssertionError("stream must not call generate")

    def stream(self, request):
        self.requests.append(request)
        if self.fail is not None and self.fail_after is None:
            raise self.fail
        for r in self.reasoning:
            yield SommChunk(text="", reasoning=r)
        for i, c in enumerate(self.chunks):
            if self.fail is not None and i == self.fail_after:
                raise self.fail
            yield SommChunk(text=c)
        if self.tool_deltas:
            yield SommChunk(text="", tool_calls=list(self.tool_deltas))
        if self.done:
            yield SommChunk(text="", done=True, model=request.model or self.default_model,
                            response=self.final)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _llm(tmp_path: Path, *providers) -> SommLLM:
    cfg = Config()
    cfg.project = "stream-shares"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(providers=list(providers), config=cfg, on_error=lambda _: None)


def _rows(llm: SommLLM) -> list[dict]:
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY ts")]
    for r in rows:
        r["routing"] = json.loads(r["routing_json"]) if r["routing_json"] else {}
    return rows


def test_stream_sends_every_argument(tmp_path):
    a = Streamer("a")
    llm = _llm(tmp_path, a)
    msgs = [{"role": "user", "content": "m"}]
    out = "".join(llm.stream(
        "", system="sys", workload="w", max_tokens=99, temperature=0.7, messages=msgs,
        tools=[{"name": "t", "parameters": {}}], tool_choice="auto", reasoning_effort="low",
        response_format={"type": "json_object"}, timeout=12.0,
    ))
    assert out == "ok"
    req = a.requests[0]
    assert (req.system, req.max_tokens, req.temperature, req.messages) == ("sys", 99, 0.7, msgs)
    assert (req.tool_choice, req.reasoning_effort, req.timeout) == ("auto", "low", 12.0)
    assert req.tools == [{"name": "t", "parameters": {}}]
    assert req.response_format == {"type": "json_object"}


def test_stream_follows_the_workload_policy(tmp_path):
    a = Streamer("a", fail=SommTransientError("502"))
    b = Streamer("b", ["fine"])
    llm = _llm(tmp_path, a, b)
    llm.register_workload(name="w")
    llm.set_workload_policy("w", {"fallback": [{"provider": "a", "model": "a-1"},
                                               {"provider": "b", "model": "b-1"}]})
    assert "".join(llm.stream("hi", workload="w")) == "fine"
    assert a.requests[0].model == "a-1" and b.requests[0].model == "b-1"
    row = _rows(llm)[-1]
    assert (row["provider"], row["model"], row["outcome"]) == ("b", "b-1", "ok")
    assert row["routing"]["route_source"] == "policy"
    assert row["routing"]["route"]["skipped"] == [["a", "a-1", "SommTransientError"]]


def test_stream_applies_a_learned_override(tmp_path):
    a = Streamer("a")
    llm = _llm(tmp_path, a)
    wl = llm.register_workload(name="w")
    llm.repo.upsert_learned_override(
        workload_id=wl.id, provider="a", model="m1", max_tokens_floor=4096,
        failure_signature="capability_empty:stripped_empty",
    )
    list(llm.stream("hi", workload="w", provider="a", model="m1", max_tokens=100))
    assert a.requests[0].max_tokens == 4096
    assert _rows(llm)[-1]["routing"]["route_source"] == "learned_override"


def test_pinned_stream_rescued_before_its_first_chunk_when_allowed(tmp_path):
    a = Streamer("a", fail=SommTransientError("502"))
    b = Streamer("b", ["saved"])
    events = []
    llm = _llm(tmp_path, a, b)
    llm._on_fallback = events.append
    assert "".join(llm.stream("hi", workload="w", provider="a", allow_fallback=True)) == "saved"
    row = _rows(llm)[-1]
    assert (row["provider"], row["outcome"]) == ("b", "ok")
    assert row["routing"]["route"]["mode"] == "pinned"
    assert [e["actual_provider"] for e in events] == ["b"]


def test_no_fallback_once_a_chunk_reached_the_caller(tmp_path):
    flaky = Streamer("flaky", ["one ", "two"], fail=SommTransientError("reset"), fail_after=1)
    spare = Streamer("spare", ["never"])
    llm = _llm(tmp_path, flaky, spare)
    seen = []
    with pytest.raises(SommTransientError) as info:
        for ev in llm.stream("hi", workload="w", events=True):
            seen.append(ev)
    assert isinstance(seen[0], StreamStart) and seen[0].provider == "flaky"
    assert seen[1:] == [TextDelta("one ")]
    assert seen[0].call_id == _rows(llm)[-1]["id"]
    assert not spare.requests
    row = _rows(llm)[-1]
    assert (row["provider"], row["outcome"]) == ("flaky", "upstream_error")
    assert row["response_hash"] != ""
    assert info.value.call_id == row["id"]


def test_reasoning_alone_is_not_a_first_chunk_for_a_text_stream(tmp_path):
    thinker = Streamer("thinker", [], reasoning=["hmm"], fail=SommTransientError("cut"),
                       fail_after=0)
    thinker.chunks = ["never"]
    spare = Streamer("spare", ["answer"])
    llm = _llm(tmp_path, thinker, spare)
    assert "".join(llm.stream("hi", workload="w")) == "answer"


def test_reasoning_is_a_first_chunk_for_an_events_stream(tmp_path):
    thinker = Streamer("thinker", ["never"], reasoning=["hmm"], fail=SommTransientError("cut"),
                       fail_after=0)
    spare = Streamer("spare", ["answer"])
    llm = _llm(tmp_path, thinker, spare)
    seen = []
    with pytest.raises(SommTransientError):
        for ev in llm.stream("hi", workload="w", events=True):
            seen.append(ev)
    assert seen[1:] == [ReasoningDelta("hmm")]
    assert not spare.requests


def test_events_carry_tool_calls_and_the_recorded_result(tmp_path):
    final = SommResponse(
        text="", model="a-m", tokens_in=40, tokens_out=9, latency_ms=1,
        tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": 1})],
        stop_reason="tool_use",
    )
    deltas = [ToolCallDelta(0, id="c1", name="lookup"), ToolCallDelta(0, arguments='{"q": 1}')]
    a = Streamer("a", [], tool_deltas=deltas, final=final)
    llm = _llm(tmp_path, a)
    got = list(llm.stream("hi", workload="w", events=True,
                          tools=[{"name": "lookup", "parameters": {}}]))
    assert got[0] == StreamStart(call_id=got[-1].result.call_id, provider="a", model="a-m")
    assert got[1:3] == deltas
    done = got[-1]
    assert isinstance(done, StreamDone) and done.usage == "provider"
    r = done.result
    assert (r.outcome.value, r.tokens_in, r.tokens_out, r.stop_reason) == ("ok", 40, 9, "tool_use")
    assert [(t.id, t.name, t.arguments) for t in r.tool_calls] == [("c1", "lookup", {"q": 1})]
    row = _rows(llm)[-1]
    assert row["id"] == r.call_id and row["tokens_in"] == 40
    assert row["routing"]["stream"] == {"usage": "provider"}
    assert row["ttft_ms"] is not None


def test_usage_is_an_estimate_when_the_provider_reports_none(tmp_path):
    llm = _llm(tmp_path, Streamer("a", ["hello"]))
    *_, done = llm.stream("hi", workload="w", events=True)
    assert done.usage == "estimate"
    row = _rows(llm)[-1]
    assert row["routing"]["stream"] == {"usage": "estimate"}
    assert row["model"] == "a-m"  # the provider's model, not its name


def test_plain_stream_still_yields_only_strings(tmp_path):
    final = SommResponse(text="ok", model="a-m", tokens_in=1, tokens_out=1, latency_ms=1)
    llm = _llm(tmp_path, Streamer("a", ["ok"], reasoning=["x"],
                                  tool_deltas=[ToolCallDelta(0, name="t")], final=final))
    assert list(llm.stream("hi", workload="w")) == ["ok"]


def test_somm_auto_streams_through_the_adapter(tmp_path, monkeypatch):
    from dataclasses import asdict

    from somm.adaptive import _QUOTA, AdaptivePolicy, WeeklyQuota

    chatgpt = Streamer("chatgpt", ["streamed"])
    llm = _llm(tmp_path, chatgpt, Streamer("minimax"))
    llm.register_workload(name="w")
    policy = AdaptivePolicy.parse({"fast": {"provider": "chatgpt", "model": "small"},
                                   "strong": {"provider": "chatgpt", "model": "large"},
                                   "fallback": {"provider": "minimax", "model": "m"}})
    llm.set_workload_policy("w", {"adaptive": {**asdict(policy),
                                               "fast_patterns": list(policy.fast_patterns)}})
    monkeypatch.setattr(_QUOTA, "read",
                        lambda: WeeklyQuota(10, int(time.time()) + 1000, time.time()))
    assert "".join(llm.stream("Write a bio", workload="w", model="somm/auto")) == "streamed"
    assert chatgpt.requests[0].model == "large"
    row = _rows(llm)[-1]
    assert row["routing"]["route"]["mode"] == "adaptive"


def test_async_stream_takes_the_new_arguments(tmp_path):
    import asyncio

    a = Streamer("a", ["x"])
    llm = _llm(tmp_path, a)

    async def run():
        return [ev async for ev in llm.astream(
            "", workload="w", messages=[{"role": "user", "content": "m"}], events=True)]

    got = asyncio.run(run())
    assert got[1] == TextDelta("x") and isinstance(got[-1], StreamDone)
    assert a.requests[0].messages == [{"role": "user", "content": "m"}]


def test_one_piece_answers_still_stream_their_tool_calls(tmp_path):
    final = SommResponse(text="", model="a-m", tokens_in=3, tokens_out=2, latency_ms=1,
                         tool_calls=[ToolCall(id="c9", name="f", arguments={"a": 1})])
    llm = _llm(tmp_path, Streamer("a", [], final=final))
    got = [ev for ev in llm.stream("hi", workload="w", events=True)
           if isinstance(ev, ToolCallDelta)]
    assert got == [ToolCallDelta(0, id="c9", name="f", arguments='{"a": 1}')]

"""What ``generate()`` records, pinned before stream and generate shared a path.

Each scenario runs one call against fake providers and compares the stored
``calls`` row (every column that does not vary run to run) and the returned
result with a snapshot taken from 0.24.0. The streaming merge moved
generate's admission, dispatch and recording into helpers both paths call;
these snapshots are how that move is shown to have changed nothing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from somm import SommLLM, hooks
from somm.errors import SommRateLimited, SommTimeout, SommTransientError
from somm.providers.base import ProviderHealth, SommResponse, ToolCall
from somm_core.config import Config
from somm_core.parse import stable_hash

_VOLATILE = {"id", "ts", "call_site", "client_version", "correlation_id"}


class Fake:
    def __init__(self, name, *, text="ok", fail=None, tool_calls=None, cost=None, raw=None):
        self.name = name
        self.default_model = f"{name}-m"
        self.text = text
        self.fail = fail
        self.tool_calls = tool_calls or []
        self.cost = cost
        self.raw = raw
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        return SommResponse(
            text=self.text, model=request.model or self.default_model, tokens_in=7,
            tokens_out=3, latency_ms=11, raw=self.raw, tool_calls=list(self.tool_calls),
            stop_reason="tool_use" if self.tool_calls else "end_turn", cost_usd=self.cost,
        )

    def stream(self, request):  # pragma: no cover - generate scenarios only
        raise AssertionError("generate never streams")

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _llm(tmp_path: Path, *providers) -> SommLLM:
    cfg = Config()
    cfg.project = "characterize"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(providers=list(providers), config=cfg, on_error=lambda _: None)


def _row(llm: SommLLM) -> dict:
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY ts")]
    assert len(rows) == 1
    row = {k: v for k, v in rows[0].items() if k not in _VOLATILE}
    row["workload_id"] = bool(row["workload_id"])
    if row.get("routing_json"):
        row["routing_json"] = json.loads(row["routing_json"])
    if row.get("error_detail"):
        # "next available in 15s" is a clock reading.
        row["error_detail"] = re.sub(r"in \d+s", "in Ns", row["error_detail"])
    return row


def _result(r) -> dict:
    return {
        "text": r.text, "provider": r.provider, "model": r.model, "outcome": r.outcome.value,
        "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "cost_usd": r.cost_usd,
        "error_kind": r.error_kind, "stop_reason": r.stop_reason,
        "tool_calls": [(t.id, t.name, t.arguments) for t in r.tool_calls],
    }


def _base(**over) -> dict:
    """The columns most scenarios share; each scenario names what differs."""
    row = {
        "project": "characterize", "workload_id": True, "prompt_id": None,
        "provider": "a", "model": "a-m", "tokens_in": 7, "tokens_out": 3, "latency_ms": 11,
        "cost_usd": 0.0, "outcome": "ok", "error_kind": None, "error_detail": None,
        "prompt_hash": stable_hash("hi"), "response_hash": stable_hash("ok"),
        "temperature": 0.2, "max_tokens": 256, "ttft_ms": None, "session_id": None,
        "parent_call_id": None, "cache_tokens_in": None, "cache_tokens_out": None,
        "citations_json": None, "cost_basis": "unknown", "cost_kind": "unknown",
        "cost_accuracy": "unknown", "cost_source": "unknown:no-price", "pricing_version": None,
        "provider_request_id": None, "billing_id": None, "reasoning_tokens": None,
        "observation_role": "production", "source_call_id": None, "eval_result_id": None,
        "origin": "native", "budget_eligible": 1, "data_tier": "unvetted",
        "top_p": None, "stop_sequences_json": None,
    }
    row.update(over)
    return row


def _route(mode, first, served, skipped=(), source="default", max_tokens=256):
    depth = sum(1 for s in skipped if s[2] not in ("boundary", "capability", "plan_blocked"))
    return {
        "route_source": source,
        "route": {"mode": mode, "first": first, "served": served, "depth": depth,
                  "skipped": [list(s) for s in skipped], "max_tokens": max_tokens},
    }


def _check(llm, result, expected_row, expected_result):
    row = _row(llm)
    missing = {k: row[k] for k in row if k not in expected_row}
    for key, want in expected_row.items():
        assert row.get(key) == want, f"{key}: {row.get(key)!r} != {want!r}"
    assert not {k: v for k, v in missing.items() if v is not None}, missing
    assert _result(result) == expected_result


def _ok_result(**over):
    out = {"text": "ok", "provider": "a", "model": "a-m", "outcome": "ok", "tokens_in": 7,
           "tokens_out": 3, "cost_usd": 0.0, "error_kind": None, "stop_reason": "end_turn",
           "tool_calls": []}
    out.update(over)
    return out


def test_chain_first_choice(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b"))
    r = llm.generate("hi", workload="w")
    _check(llm, r, _base(routing_json=_route("chain", ["a", "a-m"], ["a", "a-m"])), _ok_result())


def test_chain_falls_past_a_transient_failure(tmp_path):
    llm = _llm(tmp_path, Fake("a", fail=SommTransientError("502")), Fake("b"))
    r = llm.generate("hi", workload="w")
    _check(
        llm, r,
        _base(provider="b", model="b-m", routing_json=_route(
            "chain", ["a", "a-m"], ["b", "b-m"], [["a", "a-m", "SommTransientError"]])),
        _ok_result(provider="b", model="b-m"),
    )


def test_chain_exhausted(tmp_path):
    llm = _llm(tmp_path, Fake("a", fail=SommRateLimited("429")))
    r = llm.generate("hi", workload="w")
    detail = ("SommProvidersExhausted: all providers cooled; next available in Ns | "
              "route: a=SommRateLimited")
    _check(
        llm, r,
        _base(provider="", model="", tokens_in=0, tokens_out=0, latency_ms=0,
              outcome="exhausted", error_kind="SommProvidersExhausted", error_detail=detail, data_tier=None,
              response_hash=stable_hash(""),
              routing_json=_route("chain", ["a", "a-m"], None, [["a", "a-m", "SommRateLimited"]])),
        _ok_result(text="", provider="", model="", outcome="exhausted", tokens_in=0,
                   tokens_out=0, error_kind="SommProvidersExhausted", stop_reason=""),
    )


def test_chain_of_one_empty_answer_is_exhausted(tmp_path):
    llm = _llm(tmp_path, Fake("a", text=""))
    r = llm.generate("hi", workload="w")
    assert r.outcome.value == "exhausted"
    assert _row(llm)["routing_json"]["route"]["skipped"] == [["a", "a-m", "empty"]]


def test_tool_only_answer_is_ok(tmp_path):
    tc = ToolCall(id="t1", name="lookup", arguments={"q": 1})
    llm = _llm(tmp_path, Fake("a", text="", tool_calls=[tc]))
    r = llm.generate("hi", workload="w", tools=[{"name": "lookup", "parameters": {}}])
    _check(
        llm, r,
        _base(response_hash=stable_hash(""),
              routing_json=_route("chain", ["a", "a-m"], ["a", "a-m"])),
        _ok_result(text="", stop_reason="tool_use", tool_calls=[("t1", "lookup", {"q": 1})]),
    )


def test_pinned(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b"))
    r = llm.generate("hi", workload="w", provider="b", model="b-x")
    _check(
        llm, r,
        _base(provider="b", model="b-x",
              routing_json=_route("pinned", ["b", "b-x"], ["b", "b-x"], source="pin")),
        _ok_result(provider="b", model="b-x"),
    )


def test_pinned_failure_is_sticky(tmp_path):
    b = Fake("b")
    llm = _llm(tmp_path, Fake("a", fail=SommTimeout("slow")), b)
    r = llm.generate("hi", workload="w", provider="a")
    assert not b.requests
    row = _row(llm)
    assert row["provider"] == "a" and row["model"] == "a-m"
    assert row["outcome"] == "timeout" and row["error_kind"] == "SommTimeout"
    assert row["error_detail"].startswith("SommTimeout: slow | provider=a")
    assert "allow_fallback=True" in row["error_detail"]
    assert row["routing_json"] == _route(
        "pinned", ["a", "a-m"], None, [["a", "a-m", "SommTimeout"]], source="pin")
    assert r.outcome.value == "timeout"


def test_pinned_rescue_when_allowed(tmp_path):
    llm = _llm(tmp_path, Fake("a", fail=SommTimeout("slow")), Fake("b"))
    events = []
    llm._on_fallback = events.append
    r = llm.generate("hi", workload="w", provider="a", model="a-x", allow_fallback=True)
    row = _row(llm)
    assert (row["provider"], row["model"], row["outcome"]) == ("b", "b-m", "ok")
    assert row["routing_json"]["route"]["skipped"][0] == ["a", "a-x", "SommTimeout"]
    assert row["routing_json"]["route"]["served"] == ["b", "b-m"]
    assert [e["actual_provider"] for e in events] == ["b"]
    assert r.provider == "b"


def test_reported_cost_wins(tmp_path):
    llm = _llm(tmp_path, Fake("a", cost=0.0123))
    r = llm.generate("hi", workload="w")
    row = _row(llm)
    assert row["cost_usd"] == pytest.approx(0.0123)
    assert row["cost_basis"] == "reported"
    assert row["cost_source"] == "provider:a"
    assert r.cost_usd == pytest.approx(0.0123)


def test_short_circuit_hook(tmp_path):
    llm = _llm(tmp_path, Fake("a"))

    def cached(ctx):
        return hooks.ShortCircuit(text="from cache", provider="cache", model="cache-m",
                                  tokens_in=2, tokens_out=2, source="semantic-cache")

    hooks.register_hook(hooks.PRE_CALL, cached)
    try:
        r = llm.generate("hi", workload="w")
    finally:
        hooks.unregister_hook(hooks.PRE_CALL, cached)
    row = _row(llm)
    assert (row["provider"], row["model"], row["outcome"]) == ("cache", "cache-m", "ok")
    assert row["latency_ms"] == 0 and row["cost_usd"] == 0.0
    assert row["routing_json"] == _route("hook", ["cache", "cache-m"], ["cache", "cache-m"])
    assert r.text == "from cache"


def test_workload_policy_fallback(tmp_path):
    a, b = Fake("a", fail=SommTransientError("502")), Fake("b")
    llm = _llm(tmp_path, a, b)
    llm.register_workload(name="w")
    llm.set_workload_policy("w", {"fallback": [{"provider": "a", "model": "a-1"},
                                               {"provider": "b", "model": "b-1"}]})
    r = llm.generate("hi", workload="w")
    assert a.requests[0].model == "a-1" and b.requests[0].model == "b-1"
    row = _row(llm)
    assert (row["provider"], row["model"]) == ("b", "b-1")
    assert row["routing_json"] == _route(
        "chain", ["a", "a-1"], ["b", "b-1"], [["a", "a-1", "SommTransientError"]],
        source="policy")
    assert r.model == "b-1"


def test_request_carries_every_argument(tmp_path):
    a = Fake("a")
    llm = _llm(tmp_path, a)
    llm.generate(
        "hi", system="sys", workload="w", max_tokens=99, temperature=0.7,
        tools=[{"name": "t", "parameters": {}}], tool_choice="auto",
        messages=[{"role": "user", "content": "m"}], reasoning_effort="low",
        response_format={"type": "json_object"}, timeout=12.0,
    )
    req = a.requests[0]
    assert (req.system, req.max_tokens, req.temperature, req.tool_choice) == ("sys", 99, 0.7, "auto")
    assert req.messages == [{"role": "user", "content": "m"}]
    assert (req.reasoning_effort, req.response_format, req.timeout) == (
        "low", {"type": "json_object"}, 12.0)
    assert req.tools == [{"name": "t", "parameters": {}}]

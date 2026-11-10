"""The ledger records how each call ended and which route it took.

Every test here reads the stored ``calls`` row, not the in-memory result:
the bug class is "the result said one thing and the ledger another".
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from somm import SommLLM
from somm.errors import (
    SommEmptyResponse,
    SommProvidersExhausted,
    SommRateLimited,
    SommStructuredError,
    SommTimeout,
    SommTransientError,
)
from somm.providers.base import ProviderHealth, SommChunk, SommRequest, SommResponse
from somm_core.config import Config
from somm_core.models import Outcome, PrivacyClass


class Fake:
    def __init__(self, name="fake", *, text="ok", fail=None, tokens_out=1, raw=None,
                 chunks=None, stream_fail=None, max_tokens_sent=None, base_url=None):
        self.name = name
        self.default_model = f"{name}-m"
        if base_url is not None:
            self.base_url = base_url
        self.text = text
        self.fail = fail
        self.tokens_out = tokens_out
        self.raw = raw
        self.chunks = chunks if chunks is not None else [text]
        self.stream_fail = stream_fail
        self.max_tokens_sent = max_tokens_sent
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        resp = SommResponse(
            text=self.text, model=request.model or self.default_model,
            tokens_in=1, tokens_out=self.tokens_out, latency_ms=1, raw=self.raw,
        )
        if self.max_tokens_sent is not None:
            resp.max_tokens_sent = self.max_tokens_sent
        return resp

    def stream(self, request):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        for c in self.chunks:
            yield SommChunk(text=c)
        if self.stream_fail is not None:
            raise self.stream_fail
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _llm(tmp_path: Path, *providers) -> SommLLM:
    cfg = Config()
    cfg.project = "ledger-truth"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(providers=list(providers), config=cfg, on_error=lambda _: None)


def _rows(llm: SommLLM) -> list[dict]:
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY ts")]


def _route(row: dict) -> dict:
    return json.loads(row["routing_json"])["route"]


# ---------------------------------------------------------------------------
# 1. Outcomes reach the ledger


def test_generate_structured_failure_is_bad_json_in_the_ledger(tmp_path):
    llm = _llm(tmp_path, Fake(text="no json here"))
    with pytest.raises(SommStructuredError):
        llm.generate_structured("hi", schema={"type": "object"}, workload="w", retries=1)
    rows = _rows(llm)
    assert [r["outcome"] for r in rows] == ["bad_json", "bad_json"]
    assert {r["error_kind"] for r in rows} == {"SommStructuredError"}
    with llm.repo._open() as conn:
        updates = conn.execute("SELECT field, value FROM call_updates").fetchall()
    assert [tuple(u) for u in updates] == [("outcome", "bad_json"), ("outcome", "bad_json")]
    llm.close()


def test_generate_structured_schema_failure_is_bad_json(tmp_path):
    llm = _llm(tmp_path, Fake(text='{"a": 1}'))
    schema = {"type": "object", "required": ["b"]}
    with pytest.raises(SommStructuredError):
        llm.generate_structured("hi", schema=schema, workload="w", retries=0)
    row = _rows(llm)[-1]
    assert row["outcome"] == "bad_json"
    assert "b" in row["error_detail"]
    llm.close()


def test_extract_structured_parse_failure_is_bad_json(tmp_path):
    llm = _llm(tmp_path, Fake(text="still not json"))
    out = llm.extract_structured("hi", workload="w", retries=0)
    assert out["_somm_parse_err"] is True
    assert _rows(llm)[-1]["outcome"] == "bad_json"
    llm.close()


def test_a_later_mark_updates_the_stored_row(tmp_path):
    llm = _llm(tmp_path, Fake(text="an answer"))
    result = llm.generate("hi", workload="w")
    assert _rows(llm)[-1]["outcome"] == "ok"
    result.mark(Outcome.OFF_TASK)
    assert result.outcome == Outcome.OFF_TASK
    assert _rows(llm)[-1]["outcome"] == "off_task"
    llm.close()


def test_mark_before_the_writer_flushes_still_lands(tmp_path):
    llm = _llm(tmp_path, Fake(text="an answer"))
    result = llm.generate("hi", workload="w")
    result.mark(Outcome.OFF_TASK)  # no flush in between
    assert _rows(llm)[-1]["outcome"] == "off_task"
    llm.close()


@pytest.mark.parametrize(
    ("exc", "outcome"),
    [
        (SommTimeout("read timed out"), "timeout"),
        (SommRateLimited("429 slow down"), "rate_limit"),
        (SommTransientError("502"), "upstream_error"),
    ],
)
def test_pinned_failure_records_its_real_outcome(tmp_path, exc, outcome):
    llm = _llm(tmp_path, Fake("p", fail=exc))
    result = llm.generate("hi", workload="w", provider="p")
    assert result.outcome == Outcome(outcome)
    row = _rows(llm)[-1]
    assert row["outcome"] == outcome
    assert row["error_kind"] == type(exc).__name__
    llm.close()


def test_stream_timeout_midway_is_timeout(tmp_path):
    llm = _llm(tmp_path, Fake("p", chunks=["part"], stream_fail=SommTimeout("read timeout")))
    with pytest.raises(SommTimeout):
        list(llm.stream("hi", workload="w"))
    assert _rows(llm)[-1]["outcome"] == "timeout"
    llm.close()


# ---------------------------------------------------------------------------
# 2. Reasoning ate the budget


def _reasoning_raw(reasoning: int, out: int) -> dict:
    return {"usage": {"completion_tokens": out,
                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}


def test_empty_because_reasoning_ate_the_budget_says_so(tmp_path):
    p = Fake("p", text="", tokens_out=3072, raw=_reasoning_raw(3071, 3072), max_tokens_sent=3072)
    llm = _llm(tmp_path, p)
    result = llm.generate("hi", workload="w", provider="p", max_tokens=1024)
    assert result.outcome == Outcome.EMPTY
    row = _rows(llm)[-1]
    assert row["outcome"] == "empty"
    assert row["error_kind"] == "EmptyResponse"
    # still matched by the self-heal detector, and now told apart
    assert "stripped_empty" in row["error_detail"]
    assert "reasoning_exhausted" in row["error_detail"]
    assert "3071" in row["error_detail"] and "max_tokens" in row["error_detail"]
    assert _route(row)["max_tokens"] == 3072
    with pytest.raises(SommEmptyResponse, match="reasoning"):
        llm.generate("hi", workload="w", provider="p", max_tokens=1024, raise_on_empty=True)
    llm.close()


def test_plain_blank_reply_is_not_called_reasoning(tmp_path):
    llm = _llm(tmp_path, Fake("p", text="", tokens_out=5, raw=_reasoning_raw(0, 5)))
    llm.generate("hi", workload="w", provider="p")
    detail = _rows(llm)[-1]["error_detail"]
    assert "stripped_empty" in detail and "reasoning_exhausted" not in detail
    llm.close()


# ---------------------------------------------------------------------------
# 3. Abandoned streams


def test_abandoned_stream_is_not_ok(tmp_path):
    llm = _llm(tmp_path, Fake("p", chunks=["one ", "two ", "three"]))
    gen = llm.stream("hi", workload="w")
    assert next(gen) == "one "
    gen.close()
    row = _rows(llm)[-1]
    assert row["outcome"] == "unknown"
    assert row["error_kind"] == "StreamAbandoned"
    llm.close()


def test_abandoned_stream_does_not_alert(tmp_path):
    alerts: list[dict] = []
    llm = _llm(tmp_path, Fake("p", chunks=["one ", "two"]))
    llm._on_error = alerts.append
    for _piece in llm.stream("hi", workload="w"):
        break
    assert alerts == []
    llm.close()


def test_completed_stream_is_still_ok(tmp_path):
    llm = _llm(tmp_path, Fake("p", chunks=["one ", "two"]))
    assert "".join(llm.stream("hi", workload="w")) == "one two"
    row = _rows(llm)[-1]
    assert row["outcome"] == "ok"
    assert _route(row) == {
        "mode": "chain", "first": ["p", "p-m"], "served": ["p", "p-m"],
        "depth": 0, "skipped": [], "max_tokens": 256,
    }
    llm.close()


# ---------------------------------------------------------------------------
# 4. The route on every row


def test_first_choice_served_has_depth_zero(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b"))
    llm.generate("hi", workload="w", max_tokens=300)
    assert _route(_rows(llm)[-1]) == {
        "mode": "chain", "first": ["a", "a-m"], "served": ["a", "a-m"],
        "depth": 0, "skipped": [], "max_tokens": 300,
    }
    llm.close()


def test_fallback_records_depth_and_why_each_candidate_was_skipped(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b", fail=SommRateLimited("429")), Fake("c"))
    llm._tracker.mark_failure("a", cooldown_s=300)
    llm.generate("hi", workload="w")
    route = _route(_rows(llm)[-1])
    assert route["first"] == ["a", "a-m"]
    assert route["served"] == ["c", "c-m"]
    assert route["depth"] == 2
    assert route["skipped"] == [["a", "a-m", "cooled"], ["b", "b-m", "SommRateLimited"]]
    llm.close()


def test_pinned_route_and_effective_max_tokens(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b", max_tokens_sent=999))
    llm.generate("hi", workload="w", provider="b", max_tokens=333)
    row = _rows(llm)[-1]
    assert row["max_tokens"] == 333
    assert _route(row) == {
        "mode": "pinned", "first": ["b", "b-m"], "served": ["b", "b-m"],
        "depth": 0, "skipped": [], "max_tokens": 999,
    }
    llm.close()


def test_pinned_rescue_counts_the_pin_as_skipped(tmp_path):
    llm = _llm(tmp_path, Fake("a", fail=SommTimeout("slow")), Fake("b"))
    llm.generate("hi", workload="w", provider="a", allow_fallback=True)
    route = _route(_rows(llm)[-1])
    assert route["first"] == ["a", "a-m"] and route["served"][0] == "b"
    assert route["skipped"][0] == ["a", "a-m", "SommTimeout"]
    assert route["depth"] >= 1
    llm.close()


def test_routing_json_holds_no_prompt_text(tmp_path):
    llm = _llm(tmp_path, Fake("a"))
    llm.generate("a very secret prompt", system="secret system", workload="w")
    assert "secret" not in _rows(llm)[-1]["routing_json"]
    llm.close()


def test_read_api_exposes_routing_json(tmp_path):
    from somm_core import read

    llm = _llm(tmp_path, Fake("a"))
    llm.generate("hi", workload="w")
    llm._writer.flush()
    rows = read.search_calls(llm.config.db_path, "ledger-truth")
    assert json.loads(rows[0]["routing_json"])["route"]["served"] == ["a", "a-m"]
    llm.close()


def _source(llm: SommLLM) -> str:
    return json.loads(_rows(llm)[-1]["routing_json"])["route_source"]


def test_route_source_default(tmp_path):
    llm = _llm(tmp_path, Fake("a"))
    llm.generate("hi", workload="w")
    assert _source(llm) == "default"
    "".join(llm.stream("hi", workload="w"))
    assert _source(llm) == "default"
    llm.close()


def test_route_source_pin(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b"))
    llm.generate("hi", workload="w", provider="b")
    assert _source(llm) == "pin"
    llm.generate("hi", workload="w", model="some-model")
    assert _source(llm) == "pin"
    "".join(llm.stream("hi", workload="w", provider="b"))
    assert _source(llm) == "pin"
    llm.close()


def test_route_source_learned_override_beats_pin(tmp_path):
    llm = _llm(tmp_path, Fake("a"))
    wl = llm.register_workload(name="w")
    llm.repo.upsert_learned_override(
        workload_id=wl.id, provider="a", model="m1", max_tokens_floor=4096,
        failure_signature="capability_empty:stripped_empty",
    )
    llm.generate("hi", workload="w", provider="a", model="m1", max_tokens=100)
    assert _source(llm) == "learned_override"
    assert _rows(llm)[-1]["max_tokens"] == 4096
    # an override that changes nothing is not the reason
    llm.generate("hi", workload="w", provider="a", model="m1", max_tokens=8000)
    assert _source(llm) == "pin"
    llm.close()


def test_route_source_policy(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b"))
    llm.register_workload(name="w")
    llm.set_workload_policy("w", {"fallback": [{"provider": "b", "model": "b-model"}]})
    llm.generate("hi", workload="w")
    assert _source(llm) == "policy"
    llm.close()


def test_route_source_on_a_refusal_row(tmp_path):
    from somm.errors import SommPrivacyViolation

    llm = _llm(tmp_path, Fake("openai"))
    llm.register_workload(name="secret", privacy_class=PrivacyClass.PRIVATE)
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="secret", provider="openai")
    assert _source(llm) == "pin"
    llm.close()


# ---------------------------------------------------------------------------
# 5. Rows that reached no route explain themselves


def test_exhausted_row_carries_the_attempted_route(tmp_path):
    llm = _llm(tmp_path, Fake("a"), Fake("b"))
    llm._tracker.mark_failure("a", cooldown_s=300)
    llm._tracker.mark_failure("b", cooldown_s=300)
    result = llm.generate("hi", workload="w")
    assert result.outcome == Outcome.EXHAUSTED
    row = _rows(llm)[-1]
    assert row["provider"] == ""
    assert row["error_kind"] == "SommProvidersExhausted"
    assert "a=cooled" in row["error_detail"] and "b=cooled" in row["error_detail"]
    route = _route(row)
    assert route["served"] is None
    assert route["skipped"] == [["a", "a-m", "cooled"], ["b", "b-m", "cooled"]]
    llm.close()


def test_exhausted_stream_row_carries_the_attempted_route(tmp_path):
    llm = _llm(tmp_path, Fake("a"))
    llm._tracker.mark_failure("a", cooldown_s=300)
    with pytest.raises(SommProvidersExhausted):
        list(llm.stream("hi", workload="w"))
    row = _rows(llm)[-1]
    assert row["outcome"] == "exhausted"
    assert _route(row)["skipped"] == [["a", "a-m", "cooled"]]
    assert "a=cooled" in row["error_detail"]
    llm.close()


def test_boundary_refusal_row_carries_the_refused_route(tmp_path):
    from somm.errors import SommPrivacyViolation

    llm = _llm(tmp_path, Fake("openai"))
    llm.register_workload(name="secret", privacy_class=PrivacyClass.PRIVATE)
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="secret", max_tokens=128)
    row = _rows(llm)[-1]
    assert row["error_kind"] == "SommPrivacyViolation"
    route = _route(row)
    assert route["served"] is None and route["max_tokens"] == 128
    assert route["skipped"] == [["openai", "openai-m", "boundary"]]
    llm.close()


# ---------------------------------------------------------------------------
# Adapters report the max_tokens they actually sent


def _patch_client(handler):
    transport = httpx.MockTransport(handler)

    class _MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    return _MockedClient


def test_openai_compat_reports_the_wire_max_tokens(monkeypatch):
    from somm.providers.minimax import MinimaxProvider

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    monkeypatch.setattr(httpx, "Client", _patch_client(handler))
    resp = MinimaxProvider(api_key="k").generate(SommRequest(prompt="hi", max_tokens=1024))
    assert resp.max_tokens_sent == 3072


def test_a_result_with_a_mark_hook_still_copies_pickles_and_serialises():
    """The hook closes over the client; a result must stay a plain value."""
    import copy
    import dataclasses
    import json
    import pickle
    import threading

    from somm_core.models import MarkHook, SommResult

    lock = threading.Lock()  # stands in for the client: cannot be deep-copied or pickled
    seen = []
    result = SommResult(
        text="x", provider="p", model="m", tokens_in=1, tokens_out=1, latency_ms=1,
        cost_usd=0.0, call_id="c1", outcome=Outcome.OK,
    )
    result._on_mark = MarkHook(lambda r, _lock=lock: seen.append(r.outcome))

    as_dict = dataclasses.asdict(result)
    assert as_dict["_on_mark"] is None
    json.dumps(as_dict, default=str)
    assert copy.deepcopy(result)._on_mark is None
    assert pickle.loads(pickle.dumps(result))._on_mark is None

    result.mark(Outcome.BAD_JSON)
    assert seen == [Outcome.BAD_JSON]

"""SommLLM.decide(): a typed decision recorded on the same ledger as LLM calls.

No network: the provider is a fake that answers in the documented wire shape
(https://docs.typesafe.ai/api/system-one.md), and the HTTP adapter is tested
against httpx.MockTransport.
"""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
from somm.client import SommLLM
from somm.decide import DecisionResult, record_verdict
from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommPrivacyViolation,
    SommProviderNotConfigured,
    SommRateLimited,
    SommTransientError,
)
from somm.providers import typesafe as ts
from somm.providers.typesafe import DecisionResponse, TypeSafeProvider
from somm_core import Outcome
from somm_core.calibration import calibration
from somm_core.config import Config
from somm_core.read import decision_calibration, search_calls
from somm_core.vocabulary import vocabulary

NOUL = {"type": "noul", "noul": 0.98}
CHOICE = {
    "type": "choice",
    "choice": "calm",
    "confidence": 0.7,
    "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05},
}
SCORE = {
    "type": "score",
    "score": 1.7,
    "confidence": 0.55,
    "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
}


class FakeDecider:
    name = "typesafe"
    default_model = "jev-latest"

    def __init__(self, answer=None, fail=None, served="jev-1.14.0", tokens_in=1000):
        self.answer, self.fail, self.served, self.tokens_in = answer or NOUL, fail, served, tokens_in
        self.calls = []

    def decide(self, state, question, model=None):
        self.calls.append((state, question, model))
        if self.fail is not None:
            raise self.fail
        return DecisionResponse(
            answer=self.answer, model=self.served, tokens_in=self.tokens_in, tokens_out=12,
            latency_ms=40, raw={},
        )


@pytest.fixture
def make_llm(tmp_path):
    made = []

    def _make(decider=None, key=None):
        cfg = Config()
        cfg.project = "decide_test"
        cfg.db_dir = tmp_path / ".somm"
        cfg.spool_dir = cfg.db_dir / "spool"
        cfg.typesafe_api_key = key
        llm = SommLLM(config=cfg, providers=[])
        if decider is not None:
            llm._decision_provider = decider
        made.append(llm)
        return llm

    yield _make
    for llm in made:
        llm.close()


def _row(llm, call_id):
    llm._writer.flush()
    return llm.repo.get_call(call_id)


def test_yes_no_records_a_decide_row(make_llm):
    fake = FakeDecider(NOUL)
    llm = make_llm(fake)
    r = llm.decide("I was charged twice.", ask="Is this message about billing?", workload="triage")
    assert isinstance(r, DecisionResult)
    assert (r.label, r.p, r.abstained, r.shape) == ("yes", 0.98, False, "yes_no")
    assert r.probabilities == pytest.approx({"yes": 0.98, "no": 0.02})
    assert fake.calls[0][1] == {"type": "noul", "instructions": "Is this message about billing?"}
    row = _row(llm, r.call_id)
    assert (row.kind, row.label, row.p_top, row.abstained) == ("decide", "yes", 0.98, False)
    assert row.provider == "typesafe" and row.model == "jev-1.14.0"
    assert json.loads(row.result_json)["probabilities"]["yes"] == 0.98
    assert row.tokens_in == 1000 and row.outcome == Outcome.OK


def test_cost_is_input_tokens_at_the_bundled_price(make_llm):
    llm = make_llm(FakeDecider(NOUL, tokens_in=1_000_000))
    r = llm.decide("x", ask="Is it?")
    assert r.cost_usd == pytest.approx(0.042)
    row = _row(llm, r.call_id)
    assert row.cost_source == "somm:model_intel" and row.cost_basis == "computed"


def test_choice_and_score_shapes(make_llm):
    fake = FakeDecider(CHOICE)
    llm = make_llm(fake)
    r = llm.decide("msg", ask="What is the tone?", labels={"angry": "upset", "calm": "", "excited": ""})
    assert (r.label, r.p, r.confidence) == ("calm", 0.8, 0.7)
    assert fake.calls[0][1]["type"] == "choice"

    fake.answer = SCORE
    r = llm.decide("msg", ask="How urgent?", scale=["Can wait", "This week", "Today"],
                   context={"customer": "vip"})
    assert (r.label, r.p, r.score) == ("Today", 0.8, 1.7)
    assert set(r.probabilities) == {"Can wait", "This week", "Today"}
    assert fake.calls[1][0] == {"text": "msg", "context": {"customer": "vip"}}
    assert fake.calls[1][1]["criteria"] == ["Can wait", "This week", "Today"]


def test_min_p_marks_abstained_and_records_it(make_llm):
    llm = make_llm(FakeDecider(CHOICE))
    r = llm.decide("msg", ask="Tone?", labels=["angry", "calm", "excited"], min_p=0.9)
    assert r.abstained is True and r.label == "calm"
    row = _row(llm, r.call_id)
    assert row.abstained is True
    assert json.loads(row.result_json)["min_p"] == 0.9


def test_bad_questions_are_refused_before_any_call(make_llm):
    fake = FakeDecider()
    llm = make_llm(fake)
    with pytest.raises(ValueError):
        llm.decide("x", ask="", labels=["a", "b"])
    with pytest.raises(ValueError):
        llm.decide("x", ask="q", labels=["a", "b"], scale=["lo", "hi"])
    with pytest.raises(ValueError):
        llm.decide("x", ask="q", labels=["only"])
    with pytest.raises(ValueError):
        llm.decide("x", ask="q", min_p=1.5)
    assert fake.calls == []


def test_missing_key_is_not_configured(make_llm):
    llm = make_llm(key=None)
    with pytest.raises(SommProviderNotConfigured):
        llm.decide("x", ask="Is it?")


def test_provider_error_is_recorded_then_raised(make_llm):
    llm = make_llm(FakeDecider(fail=SommRateLimited("429", model="jev-latest")))
    with pytest.raises(SommRateLimited) as info:
        llm.decide("x", ask="Is it?", workload="triage")
    row = _row(llm, info.value.call_id)
    assert row.kind == "decide" and row.outcome == Outcome.RATE_LIMIT
    assert row.label is None and row.cost_usd == 0.0


def test_malformed_answer_is_a_typed_error(make_llm):
    llm = make_llm(FakeDecider({"type": "choice"}))
    with pytest.raises(SommTransientError):
        llm.decide("x", ask="Tone?", labels=["a", "b"])


def test_data_boundary_refuses_and_records(make_llm):
    fake = FakeDecider()
    llm = make_llm(fake)
    llm.set_data_boundary("secret", "local")
    with pytest.raises(SommPrivacyViolation):
        llm.decide("x", ask="Is it?", workload="secret")
    assert fake.calls == []
    llm._writer.flush()
    rows = search_calls(llm.repo.db_path, "decide_test", workload="secret")
    assert rows and rows[0]["error_kind"] == "SommPrivacyViolation"


def test_verdicts_feed_calibration(make_llm):
    fake = FakeDecider(NOUL)
    llm = make_llm(fake)
    r1 = llm.decide("a", ask="Billing?", workload="triage")
    r1.verdict("yes")
    r2 = llm.decide("b", ask="Billing?", workload="triage")
    record_verdict(llm, r2.call_id, "no")  # a later verdict, by call id
    llm.decide("c", ask="Billing?", workload="triage")  # no verdict
    llm._writer.flush()
    out = decision_calibration(llm.repo.db_path, "decide_test", "triage")
    assert (out["n"], out["decisions"], out["unverified"]) == (2, 3, 1)
    assert out["accuracy"] == 0.5
    assert out["brier"] == pytest.approx(((0.98 - 1) ** 2 + 0.98**2) / 2)
    assert out["ece"] == pytest.approx(abs(0.98 - 0.5))


def test_search_calls_and_vocabulary_carry_the_new_columns(make_llm):
    llm = make_llm(FakeDecider(NOUL))
    r = llm.decide("a", ask="Billing?")
    llm._writer.flush()
    (row,) = search_calls(llm.repo.db_path, "decide_test")
    assert (row["kind"], row["label"], row["p_top"], row["abstained"]) == ("decide", "yes", 0.98, 0)
    assert row["id"] == r.call_id
    assert vocabulary()["call_kinds"] == ["decide"]
    with sqlite3.connect(llm.repo.db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(calls)")}
    assert {"kind", "label", "p_top", "abstained", "result_json"} <= cols


def test_calibration_math():
    assert calibration([])["brier"] is None
    out = calibration([(0.9, True), (0.9, True), (0.9, False), (0.2, False)], bins=10)
    assert out["n"] == 4 and out["accuracy"] == 0.5
    assert out["brier"] == pytest.approx((0.01 + 0.01 + 0.81 + 0.04) / 4)
    # bin 0.9: mean 0.9 vs 2/3 right; bin 0.2: mean 0.2 vs 0 right.
    assert out["ece"] == pytest.approx(0.75 * abs(0.9 - 2 / 3) + 0.25 * 0.2)
    assert calibration([(1.0, True)])["bins"][0]["hi"] == 1.0


# --- the HTTP adapter, against the documented wire shape -------------------


def _mock(monkeypatch, status, body, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    real = httpx.Client
    monkeypatch.setattr(
        ts.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def test_adapter_sends_documented_request(monkeypatch):
    seen = []
    _mock(monkeypatch, 200, {"model": "jev-1.14.0", "answers": {"q": NOUL},
                             "usage": {"input_tokens": 120, "output_tokens": 12}}, seen)
    resp = TypeSafeProvider(api_key="k").decide("state", {"type": "noul", "instructions": "?"})
    assert (resp.model, resp.tokens_in, resp.answer) == ("jev-1.14.0", 120, NOUL)
    (req,) = seen
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["Authorization"] == "Bearer k"
    assert json.loads(req.content) == {
        "model": "jev-latest", "state": "state",
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }


@pytest.mark.parametrize(
    "status,exc",
    [(401, SommAuthError), (422, SommBadRequest), (413, SommBadRequest),
     (429, SommRateLimited), (529, SommTransientError)],
)
def test_adapter_maps_statuses(monkeypatch, status, exc):
    _mock(monkeypatch, status, {"detail": "x"})
    with pytest.raises(exc):
        TypeSafeProvider(api_key="k").decide("s", {"type": "noul"})

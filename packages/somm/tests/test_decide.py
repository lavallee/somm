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
from somm.decide import DecisionResult, grade_decisions, record_verdict
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
from somm_core.pricing import write_intel
from somm_core.read import decision_calibration, decision_economics, search_calls
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


def test_adapter_sends_many_questions_long_context_and_timeout(monkeypatch):
    seen, timeouts = [], []
    body = {"model": "jev-1.14.0", "answers": {"a": NOUL, "b": CHOICE},
            "usage": {"input_tokens": 200, "output_tokens": 20}}

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=body)

    real = httpx.Client

    def client(**kw):
        timeouts.append(kw.get("timeout"))
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(ts.httpx, "Client", client)
    qs = {"a": {"type": "noul", "instructions": "?"},
          "b": {"type": "choice", "instructions": "?", "criteria": {"x": None, "y": None}}}
    resp = TypeSafeProvider(api_key="k").decide_many("s", qs, timeout=5.0, long_context=True)
    assert resp.answer == {"a": NOUL, "b": CHOICE} and resp.tokens_in == 200
    (req,) = seen
    assert json.loads(req.content)["questions"] == qs
    assert req.headers["TypeSafe-Danger-Zone-Long-Context"] == "true"
    assert timeouts == [5.0]
    TypeSafeProvider(api_key="k").decide_many("s", qs)
    assert "TypeSafe-Danger-Zone-Long-Context" not in seen[1].headers
    assert timeouts[1] == 20.0


# --- several questions in one request --------------------------------------


class FakeMany(FakeDecider):
    """Answers decide_many with ``answers`` keyed by question name."""

    def decide_many(self, state, questions, model=None, *, timeout=None, long_context=False):
        self.calls.append((state, questions, model, timeout, long_context))
        if self.fail is not None:
            raise self.fail
        return DecisionResponse(
            answer=self.answer, model=self.served, tokens_in=self.tokens_in, tokens_out=12,
            latency_ms=40, raw={},
        )


QUESTIONS = {
    "billing": {"ask": "Is this about billing?"},
    "tone": {"ask": "Tone?", "labels": ["angry", "calm", "excited"], "min_p": 0.9},
    "urgency": {"ask": "How urgent?", "scale": ["Can wait", "This week", "Today"]},
}
ANSWERS = {"billing": NOUL, "tone": CHOICE, "urgency": SCORE}


def test_decide_many_is_one_request_and_one_row(make_llm):
    fake = FakeMany(ANSWERS)
    llm = make_llm(fake)
    out = llm.decide_many("I was charged twice.", questions=QUESTIONS, workload="triage",
                          min_p=0.5, timeout=3.0, long_context=True)
    assert list(out) == ["billing", "tone", "urgency"]
    assert (out["billing"].label, out["billing"].abstained) == ("yes", False)
    assert (out["tone"].label, out["tone"].abstained) == ("calm", True)  # its own min_p 0.9
    assert (out["urgency"].label, out["urgency"].score) == ("Today", 1.7)
    assert {r.call_id for r in out.values()} == {out["billing"].call_id}
    assert out["tone"].question == "tone" and out["tone"].cost_usd == out["billing"].cost_usd
    (state, wire, _model, timeout, long_context) = fake.calls[0]
    assert state == "I was charged twice." and (timeout, long_context) == (3.0, True)
    assert wire["billing"] == {"type": "noul", "instructions": "Is this about billing?"}
    assert wire["urgency"]["criteria"] == ["Can wait", "This week", "Today"]

    row = _row(llm, out["billing"].call_id)
    assert row.kind == "decide" and (row.label, row.p_top, row.abstained) == (None, None, None)
    stored = json.loads(row.result_json)["questions"]
    assert stored["tone"]["min_p"] == 0.9 and stored["tone"]["abstained"] is True
    assert stored["billing"]["min_p"] == 0.5 and stored["urgency"]["label"] == "Today"
    llm._writer.flush()
    assert len(search_calls(llm.repo.db_path, "decide_test", workload="triage")) == 1


def test_decide_many_refuses_bad_questions_before_any_call(make_llm):
    fake = FakeMany(ANSWERS)
    llm = make_llm(fake)
    for bad in ({}, {"a": "Is it?"}, {"a": {"ask": "q", "labels": ["x"]}},
                {"a": {"ask": "q", "lables": ["x", "y"]}}, {"": {"ask": "q"}},
                {"a": {"ask": "q", "min_p": 2}}):
        with pytest.raises(ValueError):
            llm.decide_many("x", questions=bad)
    with pytest.raises(ValueError):
        llm.decide_many("x", questions={"a": {"ask": "q"}}, timeout=0)
    assert fake.calls == []


def test_decide_many_missing_answer_is_recorded_then_raised(make_llm):
    llm = make_llm(FakeMany({"billing": NOUL}))
    with pytest.raises(SommTransientError) as info:
        llm.decide_many("x", questions=QUESTIONS, workload="triage")
    row = _row(llm, info.value.call_id)
    assert row.kind == "decide" and row.outcome != Outcome.OK and row.result_json is None


def test_per_question_verdicts_feed_calibration(make_llm):
    llm = make_llm(FakeMany(ANSWERS))
    first = llm.decide_many("a", questions=QUESTIONS, workload="triage")
    first["tone"].verdict("calm")
    first["billing"].verdict("no")
    second = llm.decide_many("b", questions=QUESTIONS, workload="triage")
    record_verdict(llm, second["tone"].call_id, "angry", question="tone")
    llm._decision_provider.answer = NOUL
    llm.decide("c", ask="Billing?", workload="triage")  # a one-question row, no verdict
    llm._writer.flush()
    db = llm.repo.db_path
    tone = decision_calibration(db, "decide_test", "triage", question="tone")
    assert (tone["n"], tone["decisions"], tone["accuracy"]) == (2, 2, 0.5)
    assert tone["question"] == "tone"
    billing = decision_calibration(db, "decide_test", "triage", question="billing")
    assert (billing["n"], billing["accuracy"]) == (1, 0.0)
    single = decision_calibration(db, "decide_test", "triage")  # decide() rows only
    assert (single["n"], single["decisions"]) == (0, 1)
    with sqlite3.connect(db) as conn:
        fields = {r[0] for r in conn.execute("SELECT field FROM call_updates")}
    assert fields == {"verdict:tone", "verdict:billing"}


# --- grading a decision workload against an LLM ----------------------------


class ScriptedLLM:
    """A generation provider that answers the grader with fixed JSON."""

    name = "fake"

    def __init__(self, texts):
        self.texts, self.requests = list(texts), []

    def generate(self, request):
        from somm.providers.base import SommResponse

        self.requests.append(request)
        return SommResponse(text=self.texts.pop(0) if self.texts else "{}",
                            model=request.model or "fake-model", tokens_in=50, tokens_out=5,
                            latency_ms=5, raw={})

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        from somm.providers.base import ProviderHealth

        return ProviderHealth(available=True, detail="fake")

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


@pytest.fixture
def grading_llm(tmp_path):
    made = []

    def _make(decider, texts):
        cfg = Config()
        cfg.project = "decide_test"
        cfg.db_dir = tmp_path / ".somm"
        cfg.spool_dir = cfg.db_dir / "spool"
        gen = ScriptedLLM(texts)
        llm = SommLLM(config=cfg, providers=[gen])
        llm._decision_provider = decider
        llm.test_gen = gen
        made.append(llm)
        return llm

    yield _make
    for llm in made:
        llm.close()


def test_grading_needs_the_shadow_opt_in(grading_llm):
    llm = grading_llm(FakeDecider(NOUL), [])
    llm.decide("x", ask="Billing?", workload="triage")
    with pytest.raises(ValueError, match="enable_shadow"):
        grade_decisions(llm, "triage")


def test_grade_decisions_records_llm_verdicts_once(grading_llm):
    fake = FakeMany(ANSWERS)
    # Newest first: the decide() row is graded before the decide_many() row.
    llm = grading_llm(fake, ['{"q": "no"}',
                             '{"billing": "yes", "tone": "angry", "urgency": "Today"}'])
    llm.register_workload(name="triage")
    llm.enable_shadow("triage", "fake", "fake-model", sample_rate=1.0)
    many = llm.decide_many("I was charged twice.", questions=QUESTIONS, workload="triage")
    fake.answer = NOUL
    one = llm.decide("Is it spam?", ask="Spam?", workload="triage")

    out = grade_decisions(llm, "triage", limit=5)
    assert out["errors"] == []
    assert (out["graded"], out["stopped"]) == (2, None)
    assert out["grader"] == "fake/fake-model" and len(out["grader_call_ids"]) == 2

    llm._writer.flush()
    db = llm.repo.db_path
    with sqlite3.connect(db) as conn:
        updates = {(c, f): v for c, f, v in conn.execute(
            "SELECT call_id, field, value FROM call_updates")}
        judges = conn.execute(
            "SELECT source_call_id, observation_role FROM calls WHERE kind IS NULL"
        ).fetchall()
    mid = many["billing"].call_id
    assert updates[(mid, "verdict:tone")] == "angry"
    assert updates[(mid, "verdict:billing")] == "yes"
    assert updates[(one.call_id, "verdict")] == "no"
    assert updates[(mid, "verdict_by")] in out["grader_call_ids"]
    assert sorted(judges) == sorted([(mid, "shadow_judge"), (one.call_id, "shadow_judge")])
    # The grader saw the input and the allowed answers.
    first = llm.test_gen.requests[1]
    assert "I was charged twice." in first.prompt and "Today" in first.prompt
    assert "angry" in first.prompt and first.model == "fake-model"

    tone = decision_calibration(db, "decide_test", "triage", question="tone")
    assert (tone["n"], tone["accuracy"]) == (1, 0.0)  # model said calm, the LLM angry

    again = grade_decisions(llm, "triage", limit=5)
    assert again["graded"] == 0  # never the same row twice


def test_shadow_worker_skips_decide_rows_and_grader_calls(grading_llm):
    from somm.workers.shadow_eval import ShadowEvalWorker

    llm = grading_llm(FakeDecider(NOUL), ['{"q": "yes"}'])
    llm.register_workload(name="triage")
    llm.enable_shadow("triage", "fake", "fake-model", sample_rate=1.0)
    llm.decide("x", ask="Billing?", workload="triage")
    grade_decisions(llm, "triage")
    llm._writer.flush()
    with sqlite3.connect(llm.repo.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] >= 1
    assert ShadowEvalWorker(llm.repo, providers=[])._fetch_candidates() == []


def test_decision_economics(grading_llm):
    fake = FakeDecider(CHOICE, tokens_in=1_000_000)  # $0.042 a request
    llm = grading_llm(fake, ["route: auth", '{"q": "calm"}'])  # the escalation, the grader
    write_intel(llm.repo, provider="fake", model="fake-model", price_in_per_1m=1.0,
                price_out_per_1m=2.0, context_window=None, capabilities={}, source="test")
    llm.register_workload(name="route")
    llm.enable_shadow("route", "fake", "fake-model", sample_rate=1.0)
    ask = {"ask": "Tone?", "labels": ["angry", "calm", "excited"]}
    r1 = llm.decide("a", **ask, workload="route", min_p=0.9)  # abstains
    llm.generate("rescue", workload="route", provider="fake", parent_call_id=r1.call_id)
    llm.decide("b", **ask, workload="route")
    grade_decisions(llm, "route", limit=1)
    llm._writer.flush()
    out = decision_economics(llm.repo.db_path, "decide_test", "route")
    assert (out["decide_calls"], out["questions"], out["abstained"]) == (2, 2, 1)
    assert out["abstained_share"] == 0.5
    assert (out["verified"], out["agreement"]) == (1, 1.0)
    assert out["decide_cost"]["mean_usd"] == pytest.approx(0.042)
    assert out["escalations"]["calls"] == 1 and out["grader"]["calls"] == 1
    per = out["per_1000_inputs"]
    llm_call = (50 * 1.0 + 5 * 2.0) / 1e6  # the scripted provider's tokens at the seeded price
    assert out["grader"]["mean_usd"] == pytest.approx(llm_call)
    assert out["escalations"]["mean_usd"] == pytest.approx(llm_call)
    assert per["llm_only_basis"] == "grader"
    assert per["llm_only_usd"] == pytest.approx(1000 * llm_call)
    # Half the inputs escalated: 1000 x (decide + 0.5 x one LLM call).
    assert per["with_decide_usd"] == pytest.approx(1000 * (0.042 + 0.5 * llm_call))


def test_decision_economics_reports_unknown_cost(grading_llm):
    llm = grading_llm(FakeDecider(NOUL), ["route: auth"])  # no price for the fake route
    r = llm.decide("a", ask="Billing?", workload="route", min_p=0.99)
    llm.generate("rescue", workload="route", provider="fake", parent_call_id=r.call_id)
    llm._writer.flush()
    out = decision_economics(llm.repo.db_path, "decide_test", "route")
    assert out["escalations"]["calls"] == 1 and out["escalations"]["mean_usd"] == "unknown"
    assert out["per_1000_inputs"]["with_decide_usd"] == "unknown"
    assert out["per_1000_inputs"]["llm_only_usd"] == "unknown"


def test_decision_economics_empty_and_old_ledger(tmp_path):
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE calls (id TEXT, project TEXT)")
        conn.execute("CREATE TABLE workloads (id TEXT, name TEXT)")
        conn.execute("CREATE TABLE call_updates (call_id TEXT, field TEXT, value TEXT)")
    out = decision_economics(db, "p", "w")
    assert out["decide_calls"] == 0 and out["agreement"] is None
    assert out["decide_cost"]["mean_usd"] == "unknown"
    assert out["per_1000_inputs"]["with_decide_usd"] == "unknown"


def test_vocabulary_names_the_verdict_fields():
    fields = vocabulary()["call_update_fields"]
    assert {"verdict", "verdict_by"} <= set(fields["known"])
    assert fields["patterns"] == ["verdict:<question>"]


def test_grading_stops_at_the_shadow_budget(grading_llm):
    llm = grading_llm(FakeDecider(NOUL), ['{"q": "yes"}'])
    llm.register_workload(name="triage")
    llm.enable_shadow("triage", "fake", "fake-model", sample_rate=1.0, budget_usd_daily=0.0)
    llm.decide("x", ask="Billing?", workload="triage")
    out = grade_decisions(llm, "triage")
    assert (out["graded"], out["stopped"]) == (0, "budget")
    assert llm.test_gen.requests == []

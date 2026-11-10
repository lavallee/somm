"""Tests for online-eval sample capture on the call path (0.7.0)."""

from __future__ import annotations

import json
from pathlib import Path

from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommChunk, SommResponse
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="the answer", model="fake-m", tokens_in=3, tokens_out=2, latency_ms=1
        )

    def stream(self, request):
        yield SommChunk(text="the answer")
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _llm(tmp_path: Path) -> SommLLM:
    cfg = Config()
    cfg.project = "capture_test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(providers=[FakeProvider()], config=cfg)


def _samples(llm) -> list[tuple]:
    with llm.repo._open() as conn:
        return conn.execute(
            "SELECT call_id, prompt_body, response_body FROM samples"
        ).fetchall()


def _enable_shadow(llm, workload: str, rate: float = 1.0):
    wl = llm.repo.register_workload(name=workload, project=llm.config.project)
    llm.repo.set_shadow_config(
        wl.id,
        {"gold_provider": "fake", "gold_model": "fake-m", "sample_rate": rate,
         "budget_usd_daily": 1.0},
    )
    llm._shadow_cfg_cache.clear()
    return wl


def test_shadow_workload_captures_bodies(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    result = llm.generate(prompt="what is 2+2?", workload="graded")
    rows = _samples(llm)
    assert len(rows) == 1
    assert rows[0][0] == result.call_id
    assert rows[0][1] == "what is 2+2?"
    assert rows[0][2] == "the answer"


def test_no_shadow_config_means_no_capture(tmp_path):
    llm = _llm(tmp_path)
    llm.generate(prompt="hi", workload="plain")
    assert _samples(llm) == []


def test_rate_zero_never_captures(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=0.0)
    llm.generate(prompt="hi", workload="graded")
    assert _samples(llm) == []


def test_private_workload_never_captured(tmp_path):
    from somm_core import PrivacyClass

    llm = _llm(tmp_path)
    wl = llm.repo.register_workload(
        name="secret", project=llm.config.project, privacy_class=PrivacyClass.PRIVATE
    )
    llm.repo.set_shadow_config(
        wl.id,
        {"gold_provider": "fake", "gold_model": "fake-m", "sample_rate": 1.0,
         "budget_usd_daily": 1.0},
    )
    llm._shadow_cfg_cache.clear()
    # A private workload may only reach a local route (0.21 data boundary),
    # so serve it from a loopback endpoint: the call runs, nothing is captured.
    llm.providers[0].base_url = "http://127.0.0.1:11434"
    llm.generate(prompt="hi", workload="secret")
    assert _samples(llm) == []


def test_oversized_body_skipped(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    llm.generate(prompt="x" * 300_000, workload="graded")
    assert _samples(llm) == []


def test_messages_captured_as_json(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    llm.generate(
        prompt="ignored",
        messages=[{"role": "user", "content": "hello"}],
        workload="graded",
    )
    rows = _samples(llm)
    assert len(rows) == 1
    assert '"role": "user"' in rows[0][1]


def test_rate_partitions_deterministically(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=0.5)
    for _ in range(60):
        llm.generate(prompt="hi", workload="graded")
    n = len(_samples(llm))
    assert 10 < n < 50  # ~50% with slack; 0 or 60 would mean the gate is broken

def _request_json(llm) -> list:
    with llm.repo._open() as conn:
        return [r[0] for r in conn.execute("SELECT request_json FROM samples").fetchall()]


_PARAM_KEYS = {
    "system", "tools", "tool_choice", "response_format", "reasoning_effort",
    "temperature", "max_tokens",
}


def test_capture_records_the_request_parameters_a_replay_needs(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    tools = [{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}]
    llm.generate(
        prompt="hi", system="be brief", workload="graded", tools=tools,
        tool_choice="auto", temperature=0.4, max_tokens=99,
    )
    (raw,) = _request_json(llm)
    params = json.loads(raw)
    assert set(params) == _PARAM_KEYS
    assert params["system"] == "be brief"
    assert params["tools"][0]["name"] == "lookup" and params["tool_choice"] == "auto"
    assert (params["temperature"], params["max_tokens"]) == (0.4, 99)


def test_streams_capture_the_request_parameters_too(tmp_path):
    llm = _llm(tmp_path)
    _enable_shadow(llm, "graded", rate=1.0)
    assert "".join(llm.stream("hi", system="s", workload="graded")) == "the answer"
    (raw,) = _request_json(llm)
    assert json.loads(raw)["system"] == "s"


def test_a_sample_written_without_parameters_reads_back_null(tmp_path):
    llm = _llm(tmp_path)
    llm.repo.write_sample("old-call", "p", "r")  # the pre-0031 call shape
    assert _request_json(llm) == [None]
    assert _samples(llm) == [("old-call", "p", "r")]

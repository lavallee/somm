"""Tests for D2c library additions: extract_structured, prompt versioning,
provenance helper, parallel_slots."""

from __future__ import annotations

from pathlib import Path

import pytest
import somm
from somm.client import SommLLM
from somm.prompts import PromptNotFound
from somm.providers.base import ProviderHealth, SommResponse
from somm_core import Outcome
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "ok", model: str = "fake-m"):
        self._text = text
        self._model = model

    def generate(self, request):
        return SommResponse(
            text=self._text,
            model=self._model,
            tokens_in=3,
            tokens_out=2,
            latency_ms=5,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "d2c"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


# ---------------------------------------------------------------------------
# extract_structured


def test_extract_structured_happy_path(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text='{"name": "foo", "count": 3}')
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("parse this", workload="es_ok")
        assert data == {"name": "foo", "count": 3}
    finally:
        llm.close()


def test_extract_structured_handles_markdown_fence(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text='```json\n{"a": 1}\n```')
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("p", workload="es_fence")
        assert data == {"a": 1}
    finally:
        llm.close()


def test_extract_structured_returns_raw_on_parse_fail(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="this is definitely not json")
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("p", workload="es_fail")
        assert data["_somm_parse_err"] is True
        assert data["raw"] == "this is definitely not json"
    finally:
        llm.close()


def test_extract_structured_think_stripped(tmp_path):
    cfg = _tmp_config(tmp_path)
    # Upstream strip is in adapters; but our FakeProvider returns raw —
    # the library-side parse still handles <think> via extract_json.
    fake = FakeProvider(text='<think>plan</think>{"ok": true}')
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        data = llm.extract_structured("p", workload="es_think")
        assert data == {"ok": True}
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Prompt versioning


def test_register_prompt_first_is_v1(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p = llm.register_prompt(workload="pv", body="Extract contacts from {text}")
        assert p.version == "v1"
    finally:
        llm.close()


def test_register_prompt_is_idempotent(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="pv", body="same body")
        p2 = llm.register_prompt(workload="pv", body="same body")
        assert p1.id == p2.id
        assert p1.version == p2.version == "v1"
    finally:
        llm.close()


def test_register_prompt_minor_bump_on_body_change(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="pv", body="first body")
        assert p1.version == "v1"
        p2 = llm.register_prompt(workload="pv", body="second body")
        assert p2.version == "v1.1"
        p3 = llm.register_prompt(workload="pv", body="third body")
        assert p3.version == "v1.2"
    finally:
        llm.close()


def test_register_prompt_major_bump_explicit(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.register_prompt(workload="pv", body="v1 body")
        p = llm.register_prompt(workload="pv", body="breaking change", bump="major")
        assert p.version == "v2"
    finally:
        llm.close()


def test_register_prompt_explicit_version(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p = llm.register_prompt(workload="pv", body="body", bump="v3")
        assert p.version == "v3"
    finally:
        llm.close()


def test_get_prompt_latest_and_pinned(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.register_prompt(workload="pv", body="body-1")
        llm.register_prompt(workload="pv", body="body-2")
        llm.register_prompt(workload="pv", body="body-3")

        latest = llm.prompt("pv", version="latest")
        assert latest.body == "body-3"

        pinned = llm.prompt("pv", version="v1")
        assert pinned.body == "body-1"
    finally:
        llm.close()


def test_get_prompt_missing_raises(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        with pytest.raises(PromptNotFound):
            llm.prompt("nonexistent_workload", version="v1")
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Provenance


def test_provenance_shape(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="hi")
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        r = llm.generate("hi", workload="prov")
        prov = somm.provenance(r)
        assert prov["schema"] == 1
        assert prov["call_id"] == r.call_id
        assert prov["provider"] == "fake"
        assert prov["model"] == "fake-m"
        assert prov["tokens_in"] == 3
        assert prov["tokens_out"] == 2
        assert prov["outcome"] == Outcome.OK.value
        assert isinstance(prov["stamped_at"], str)
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# parallel_slots


def test_parallel_slots_single_provider_fills(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider()
    fake.name = "ollama"
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        assert llm.parallel_slots(4) == ["ollama"] * 4
    finally:
        llm.close()


def test_parallel_slots_stripes_across_providers(tmp_path):
    cfg = _tmp_config(tmp_path)
    p1 = FakeProvider()
    p1.name = "ollama"
    p2 = FakeProvider()
    p2.name = "openrouter"
    llm = SommLLM(config=cfg, providers=[p1, p2])
    try:
        slots = llm.parallel_slots(6)
        # Ollama hint=2, openrouter=4 → 6 slots proportional: 2 ollama, 4 openrouter
        assert len(slots) == 6
        assert slots.count("ollama") == 2
        assert slots.count("openrouter") == 4
        # First two positions interleave (no stampede)
        assert slots[0] != slots[1] or slots[:2] == ["ollama", "ollama"]
    finally:
        llm.close()


def test_parallel_slots_zero(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        assert llm.parallel_slots(0) == []
    finally:
        llm.close()


def test_parallel_slots_skips_cooled(tmp_path):
    cfg = _tmp_config(tmp_path)
    p1 = FakeProvider()
    p1.name = "ollama"
    p2 = FakeProvider()
    p2.name = "openrouter"
    llm = SommLLM(config=cfg, providers=[p1, p2])
    try:
        llm._tracker.mark_failure("ollama", cooldown_s=300)
        slots = llm.parallel_slots(4)
        assert all(s == "openrouter" for s in slots)
    finally:
        llm.close()

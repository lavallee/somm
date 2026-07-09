"""Prompt-version binding on call telemetry."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import somm.client as client_mod
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommChunk, SommResponse
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text='{"ok": true}',
            model=request.model or "fake-m",
            tokens_in=3,
            tokens_out=2,
            latency_ms=5,
        )

    def stream(self, request):
        yield SommChunk(text="hello", done=False)
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "prompt-bind"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _call_prompt_id(db_path: Path, call_id: str) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT prompt_id FROM calls WHERE id = ?",
            (call_id,),
        ).fetchone()
    assert row is not None
    return row[0]


def test_generate_prompt_object_stamps_prompt_id(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        prompt = llm.register_prompt(workload="claims", body="Extract claim JSON")
        result = llm.generate(prompt, workload="claims")
    finally:
        llm.close()

    assert _call_prompt_id(cfg.db_path, result.call_id) == prompt.id


def test_generate_string_hash_match_stamps_prompt_id(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        prompt = llm.register_prompt(workload="claims", body="Extract claim JSON")
        result = llm.generate(prompt.body, workload="claims")
    finally:
        llm.close()

    assert _call_prompt_id(cfg.db_path, result.call_id) == prompt.id


def test_generate_unregistered_string_leaves_prompt_id_null(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        result = llm.generate("not registered", workload="claims")
    finally:
        llm.close()

    assert _call_prompt_id(cfg.db_path, result.call_id) is None


def test_stream_prompt_object_stamps_prompt_id(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        prompt = llm.register_prompt(workload="claims", body="Stream claim JSON")
        pieces = list(llm.stream(prompt, workload="claims"))
    finally:
        llm.close()

    assert "".join(pieces) == "hello"
    with sqlite3.connect(cfg.db_path) as conn:
        row = conn.execute("SELECT prompt_id FROM calls").fetchone()
    assert row[0] == prompt.id


def test_prompt_id_hash_match_cache_avoids_second_lookup(tmp_path, monkeypatch):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    real_lookup = client_mod.prompt_ids_for_workload
    calls = 0

    def counted_lookup(repo, workload_id):
        nonlocal calls
        calls += 1
        return real_lookup(repo, workload_id)

    monkeypatch.setattr(client_mod, "prompt_ids_for_workload", counted_lookup)
    try:
        prompt = llm.register_prompt(workload="claims", body="Cached body")
        first = llm.generate(prompt.body, workload="claims")
        second = llm.generate(prompt.body, workload="claims")
    finally:
        llm.close()

    assert calls == 1
    assert _call_prompt_id(cfg.db_path, first.call_id) == prompt.id
    assert _call_prompt_id(cfg.db_path, second.call_id) == prompt.id

"""Phase-2 telemetry columns: session/parent hierarchy, cache tokens,
citations, and streaming TTFT land on the calls row and SommResult."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommChunk, SommEmbedResponse, SommResponse
from somm_core.config import Config


class RawProvider:
    """Fake provider whose response.raw carries whatever usage/citation
    shape a test wants to exercise."""

    name = "fake"

    def __init__(self, *, text: str = "ok", raw: dict | None = None):
        self.text = text
        self.raw = raw

    def generate(self, request):
        return SommResponse(
            text=self.text,
            model=request.model or "fake-model",
            tokens_in=7,
            tokens_out=3,
            latency_ms=5,
            raw=self.raw,
        )

    def stream(self, request):  # pragma: no cover - overridden where needed
        yield SommChunk(text=self.text, done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "telemetry"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _row(llm: SommLLM, call_id: str) -> dict:
    llm._writer.flush(timeout=2.0)
    with sqlite3.connect(llm.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT ttft_ms, session_id, parent_call_id, cache_tokens_in, "
            "cache_tokens_out, citations_json FROM calls WHERE id = ?",
            (call_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def test_generate_persists_session_parent_cache_and_citations(tmp_path):
    raw = {
        "usage": {
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 12,
        },
        "citations": ["https://example.com/a", "https://example.com/b"],
    }
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[RawProvider(raw=raw)])
    try:
        result = llm.generate(
            "hi",
            workload="w",
            provider="fake",
            session_id="sess-1",
            parent_call_id="parent-1",
        )
        row = _row(llm, result.call_id)
    finally:
        llm.close()

    assert row["session_id"] == "sess-1"
    assert row["parent_call_id"] == "parent-1"
    assert row["cache_tokens_in"] == 40
    assert row["cache_tokens_out"] == 12
    assert "example.com/a" in row["citations_json"]
    assert result.cache_tokens_in == 40
    assert result.citations == ["https://example.com/a", "https://example.com/b"]
    assert row["ttft_ms"] is None  # non-stream


def test_generate_extracts_openai_cache_tokens_and_null_defaults(tmp_path):
    raw = {"usage": {"prompt_tokens_details": {"cached_tokens": 25}}}
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[RawProvider(raw=raw)])
    try:
        result = llm.generate("hi", workload="w", provider="fake")
        row = _row(llm, result.call_id)
    finally:
        llm.close()
    assert row["cache_tokens_in"] == 25
    assert row["cache_tokens_out"] is None
    assert row["citations_json"] is None

    # A plain call with no usage/citations leaves everything NULL.
    llm2 = SommLLM(config=_tmp_config(tmp_path / "b"), providers=[RawProvider(raw=None)])
    try:
        r2 = llm2.generate("hi", workload="w", provider="fake")
        row2 = _row(llm2, r2.call_id)
    finally:
        llm2.close()
    assert row2["cache_tokens_in"] is None
    assert row2["session_id"] is None


def test_stream_records_ttft(tmp_path):
    class SlowStream(RawProvider):
        def stream(self, request):
            time.sleep(0.02)
            yield SommChunk(text="pong", done=True)

    llm = SommLLM(config=_tmp_config(tmp_path), providers=[SlowStream()])
    try:
        pieces = list(llm.stream("hi", workload="w", provider="fake"))
        # find the call the stream recorded
        llm._writer.flush(timeout=2.0)
        with sqlite3.connect(llm.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT ttft_ms FROM calls ORDER BY ts DESC LIMIT 1"
            ).fetchone()
    finally:
        llm.close()

    assert "".join(pieces) == "pong"
    assert row["ttft_ms"] is not None
    assert row["ttft_ms"] >= 0


def test_embed_threads_session_parent_to_call_row(tmp_path):
    class FakeOllama(RawProvider):
        name = "ollama"
        default_embed_model = "embed-m"

        def embed(self, request):
            return SommEmbedResponse(
                embedding=[0.1, 0.2],
                model="embed-m",
                tokens_in=2,
                latency_ms=3,
            )

    llm = SommLLM(config=_tmp_config(tmp_path), providers=[FakeOllama()])
    try:
        result = llm.embed(
            "embed me",
            workload="w",
            session_id="sess-embed",
            parent_call_id="parent-embed",
        )
        row = _row(llm, result.call_id)
    finally:
        llm.close()

    assert row["session_id"] == "sess-embed"
    assert row["parent_call_id"] == "parent-embed"

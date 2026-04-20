"""Streaming tests — ThinkStreamStripper, ollama stream, OpenAI SSE stream,
SommLLM.stream() with buffered think-strip across chunks.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommChunk, SommResponse
from somm_core.config import Config
from somm_core.parse import ThinkStreamStripper

# ---------------------------------------------------------------------------
# ThinkStreamStripper unit tests


def test_stripper_no_think_passthrough():
    s = ThinkStreamStripper()
    out1 = s.feed("hello ")
    out2 = s.feed("world")
    tail = s.flush()
    assert out1 + out2 + tail == "hello world"


def test_stripper_whole_think_in_one_chunk():
    s = ThinkStreamStripper()
    out = s.feed("<think>planning</think>answer")
    tail = s.flush()
    assert out + tail == "answer"


def test_stripper_think_across_two_chunks():
    s = ThinkStreamStripper()
    out1 = s.feed("hello <thin")
    out2 = s.feed("k>reasoning</think>world")
    tail = s.flush()
    combined = out1 + out2 + tail
    assert combined == "hello world"


def test_stripper_think_across_many_chunks():
    s = ThinkStreamStripper()
    chunks = ["pre ", "<think>a", "b", "c</thi", "nk>", "post"]
    collected = []
    for c in chunks:
        collected.append(s.feed(c))
    collected.append(s.flush())
    assert "".join(collected) == "pre post"


def test_stripper_two_think_blocks():
    s = ThinkStreamStripper()
    chunks = ["<think>one</think>A<think>two</think>B"]
    out = s.feed(chunks[0]) + s.flush()
    assert out == "AB"


def test_stripper_unclosed_think_emitted_at_flush():
    """If stream ends mid-think, leak rather than silently drop."""
    s = ThinkStreamStripper()
    out = s.feed("ok<think>leaked")
    tail = s.flush()
    assert "ok" in (out + tail)
    # capped marks that we had to flush unclosed
    assert s.capped is True


def test_stripper_holds_back_potential_tag_prefix():
    """Emits nothing if buffer ends with '<th' — could be start of <think>."""
    s = ThinkStreamStripper()
    out1 = s.feed("hi <th")
    # "<th" is held back — not emitted yet
    assert out1 == "hi "
    out2 = s.feed("ink>planning</think> done")
    tail = s.flush()
    assert (out1 + out2 + tail) == "hi  done"


def test_stripper_lookahead_cap():
    """Unbounded think block larger than lookahead window — emit + mark."""
    s = ThinkStreamStripper(lookahead_bytes=16)
    out = s.feed("<think>" + "x" * 100)
    tail = s.flush()
    assert s.capped is True
    # The think content gets emitted verbatim after cap (safer than dropping).
    assert "x" in (out + tail)


# ---------------------------------------------------------------------------
# SommLLM.stream() — fake streaming provider


class FakeStreamingProvider:
    name = "fake-stream"

    def __init__(self, chunks: list[str]):
        self._chunks = list(chunks)

    def generate(self, request):
        return SommResponse(
            text="".join(self._chunks),
            model="fake-m",
            tokens_in=3,
            tokens_out=2,
            latency_ms=5,
        )

    def stream(self, request):
        for c in self._chunks:
            yield SommChunk(text=c, done=False)
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(text) // 4)


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "streams"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


def test_stream_yields_text_and_writes_telemetry(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeStreamingProvider(["hello ", "world", "!"])
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        pieces = list(llm.stream("say hi", workload="stream_ok"))
        assert "".join(pieces) == "hello world!"
    finally:
        llm.close()

    # Telemetry row landed
    stats = llm.repo.stats_by_workload("streams", since_days=1)
    assert len(stats) == 1
    assert stats[0]["workload"] == "stream_ok"
    assert stats[0]["n_calls"] == 1


def test_stream_strips_think_across_chunks(tmp_path):
    cfg = _tmp_config(tmp_path)
    chunks = ["pre ", "<thin", "k>reason</think>", "post"]
    fake = FakeStreamingProvider(chunks)
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        out = "".join(llm.stream("hi", workload="stream_think"))
    finally:
        llm.close()
    assert out == "pre post"


def test_stream_empty_marks_outcome(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeStreamingProvider([""])  # single empty chunk
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        out = "".join(llm.stream("hi", workload="stream_empty"))
        assert out == ""
    finally:
        llm.close()
    # Row written with outcome=empty
    call = [c for c in _all_calls(llm.repo) if c.workload_id][-1]
    from somm_core import Outcome

    assert call.outcome == Outcome.EMPTY


def test_stream_provider_error_propagates(tmp_path):
    cfg = _tmp_config(tmp_path)

    class BustedProvider:
        name = "busted"

        def generate(self, request):  # pragma: no cover
            raise RuntimeError("boom")

        def stream(self, request):
            yield SommChunk(text="partial ", done=False)
            raise RuntimeError("mid-stream boom")

        def health(self):
            return ProviderHealth(available=True)

        def models(self):
            return []

        def estimate_tokens(self, text, model):
            return 1

    llm = SommLLM(config=cfg, providers=[BustedProvider()])
    try:
        with pytest.raises(RuntimeError, match="mid-stream"):
            list(llm.stream("hi", workload="stream_err"))
    finally:
        llm.close()
    # Telemetry still written (outcome=upstream_error)
    from somm_core import Outcome

    call = [c for c in _all_calls(llm.repo) if c.workload_id][-1]
    assert call.outcome == Outcome.UPSTREAM_ERROR


# ---------------------------------------------------------------------------
# Live ollama streaming (skipped if ollama down)


def _ollama_live() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_test_model() -> str | None:
    """Same as test_smoke._ollama_test_model — pick whatever is installed
    so the test runs on any dev machine with ollama available."""
    import os

    env = os.environ.get("SOMM_OLLAMA_MODEL")
    if env:
        return env
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        r.raise_for_status()
        models = r.json().get("models") or []
        if models:
            return models[0].get("name") or models[0].get("model")
    except Exception:
        return None
    return None


@pytest.mark.skipif(not _ollama_live(), reason="no local ollama")
def test_ollama_live_stream(tmp_path):
    model = _ollama_test_model()
    if not model:
        pytest.skip("ollama reachable but has no models installed")
    cfg = _tmp_config(tmp_path)
    cfg.ollama_model = model
    llm = SommLLM(config=cfg)
    try:
        pieces = list(
            llm.stream(
                "Reply with exactly: pong",
                workload="live_stream",
                max_tokens=8,
                temperature=0.0,
            )
        )
        text = "".join(pieces)
    finally:
        llm.close()

    # Don't assert on exact text — model may vary. But we should have received
    # at least one chunk and the row should be logged.
    assert text  # non-empty


# ---------------------------------------------------------------------------


def _all_calls(repo):
    from datetime import datetime

    from somm_core.models import Call, Outcome

    with repo._open() as conn:
        rows = conn.execute(
            "SELECT id, ts, project, workload_id, prompt_id, provider, model, "
            "tokens_in, tokens_out, latency_ms, cost_usd, outcome, error_kind, "
            "prompt_hash, response_hash FROM calls ORDER BY ts"
        ).fetchall()
    return [
        Call(
            id=r[0],
            ts=datetime.fromisoformat(r[1]),
            project=r[2],
            workload_id=r[3],
            prompt_id=r[4],
            provider=r[5],
            model=r[6],
            tokens_in=r[7],
            tokens_out=r[8],
            latency_ms=r[9],
            cost_usd=r[10],
            outcome=Outcome(r[11]),
            error_kind=r[12],
            prompt_hash=r[13],
            response_hash=r[14],
        )
        for r in rows
    ]

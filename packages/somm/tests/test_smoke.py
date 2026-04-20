"""D1 smoke tests.

Verifies end-to-end skeleton: schema applied, writer queue persists calls,
workload auto-registration in observe mode, strict mode raises, ollama live
path (skipped if ollama is down).
"""

from __future__ import annotations

import os
from datetime import UTC
from pathlib import Path

import httpx
import pytest
from somm.client import SommLLM, SommStrictMode
from somm.providers.base import ProviderHealth, SommResponse
from somm.telemetry import drain_spool
from somm_core import Outcome
from somm_core.config import Config
from somm_core.repository import Repository

# ---------------------------------------------------------------------------
# Test doubles


class FakeProvider:
    """Deterministic provider for tests — no network."""

    name = "fake"

    def __init__(self, text: str = "hi", tokens_in: int = 4, tokens_out: int = 1) -> None:
        self._text = text
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out

    def generate(self, request):
        return SommResponse(
            text=self._text,
            model=request.model or "fake-model",
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            latency_ms=5,
            raw={"fake": True},
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True, detail="fake")

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return len(text) // 4


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "smoke"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


# ---------------------------------------------------------------------------
# Schema + repository


def test_schema_applies_and_perms_enforced(tmp_path):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    # DB file exists
    assert cfg.db_path.exists()
    # Perms: 600 on file, 700 on dir
    assert oct(cfg.db_path.stat().st_mode)[-3:] == "600"
    assert oct(cfg.db_path.parent.stat().st_mode)[-3:] == "700"
    # Schema version table populated
    stats = repo.stats_by_workload("smoke", since_days=1)
    assert stats == []


# ---------------------------------------------------------------------------
# Library end-to-end with fake provider


def test_observe_mode_auto_registers_and_writes(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="hello JSON")
    llm = SommLLM(config=cfg, providers=[fake])

    result = llm.generate("say hi in json", workload="ad_hoc_test")

    assert result.text == "hello JSON"
    assert result.provider == "fake"
    assert result.outcome == Outcome.OK
    assert result.call_id

    llm.close()

    call = llm.repo.get_call(result.call_id)
    assert call is not None
    assert call.provider == "fake"
    assert call.project == "smoke"
    # Auto-registered workload exists
    wl = llm.repo.workload_by_name("ad_hoc_test", "smoke")
    assert wl is not None


def test_strict_mode_raises_on_unregistered_workload(tmp_path):
    cfg = _tmp_config(tmp_path)
    cfg.mode = "strict"
    fake = FakeProvider()
    llm = SommLLM(config=cfg, providers=[fake])

    with pytest.raises(SommStrictMode) as exc_info:
        llm.generate("test", workload="never_registered")

    assert "SOMM_WORKLOAD_UNREGISTERED" in str(exc_info.value)
    assert "Problem:" in str(exc_info.value)
    assert "Fix:" in str(exc_info.value)
    assert "Docs:" in str(exc_info.value)

    llm.close()


def test_strict_mode_allows_registered_workload(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider()
    llm = SommLLM(config=cfg, providers=[fake])

    llm.repo.register_workload(name="known", project="smoke")

    # Flip to strict on the same repo
    llm.config.mode = "strict"
    result = llm.generate("ok", workload="known")
    assert result.outcome == Outcome.OK

    llm.close()


def test_empty_response_marks_outcome(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="")
    llm = SommLLM(config=cfg, providers=[fake])

    result = llm.generate("prompt", workload="empty_test")
    assert result.outcome == Outcome.EMPTY

    llm.close()


def test_result_mark_updates_outcome(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="bad json: }")
    llm = SommLLM(config=cfg, providers=[fake])

    result = llm.generate("prompt", workload="mark_test")
    result.mark(Outcome.BAD_JSON)
    assert result.outcome == Outcome.BAD_JSON
    # TODO D2: persist mark via call_updates; for D1 it's an in-result signal.

    llm.close()


def test_stats_by_workload_rolls_up(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="ok", tokens_in=10, tokens_out=5)
    llm = SommLLM(config=cfg, providers=[fake])

    for _ in range(3):
        llm.generate("p", workload="rollup_test")

    llm.close()  # flush the queue

    stats = llm.repo.stats_by_workload("smoke", since_days=1)
    assert len(stats) == 1
    row = stats[0]
    assert row["workload"] == "rollup_test"
    assert row["n_calls"] == 3
    assert row["tokens_in"] == 30
    assert row["tokens_out"] == 15


# ---------------------------------------------------------------------------
# Parse helpers


def test_strip_think_block():
    from somm_core.parse import strip_think_block

    assert strip_think_block("<think>planning</think>hello") == "hello"
    assert strip_think_block("no think here") == "no think here"
    assert strip_think_block("<think>a\nb\nc</think> done") == "done"


def test_extract_json_variants():
    from somm_core.parse import extract_json

    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert extract_json('prose before {"a": 1} and after') == {"a": 1}
    assert extract_json("nothing parseable here") is None


def test_extract_json_with_control_chars():
    """Literal control bytes inside strings break json.loads — fallback strips."""
    from somm_core.parse import extract_json

    # Null byte embedded mid-string (seen from some local models)
    assert extract_json('{"msg": "hi\x00there"}') == {"msg": "hi there"}
    # Bell + vertical tab
    assert extract_json('{"x": "a\x07b\x0bc"}') == {"x": "a b c"}


def test_extract_json_with_unescaped_newlines():
    """Unescaped \\n inside a JSON string — flatten-whitespace fallback recovers."""
    from somm_core.parse import extract_json

    raw = '{"text": "line one\nline two\nline three"}'
    parsed = extract_json(raw)
    assert parsed is not None
    assert "line one" in parsed["text"]
    assert "line three" in parsed["text"]


# ---------------------------------------------------------------------------
# Spool fallback


def test_spool_drain_roundtrip(tmp_path):
    """Write a spool file, drain it into SQLite, confirm rows land."""
    import json
    import uuid
    from datetime import datetime

    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    spool = cfg.spool_dir
    spool.mkdir(parents=True, exist_ok=True)

    call_id = str(uuid.uuid4())
    ts = datetime.now(UTC).isoformat()
    row = {
        "id": call_id,
        "ts": ts,
        "project": "smoke",
        "workload_id": None,
        "prompt_id": None,
        "provider": "fake",
        "model": "fake-1",
        "tokens_in": 1,
        "tokens_out": 1,
        "latency_ms": 1,
        "cost_usd": 0.0,
        "outcome": "ok",
        "error_kind": None,
        "prompt_hash": "abcdef0123456789",
        "response_hash": "fedcba9876543210",
    }
    spill = spool / "20260101T000000000000.jsonl"
    spill.write_text(json.dumps(row) + "\n")

    drained = drain_spool(repo, spool)
    assert drained == 1
    assert not spill.exists()

    got = repo.get_call(call_id)
    assert got is not None
    assert got.provider == "fake"


# ---------------------------------------------------------------------------
# Live ollama path (skipped if unavailable)


def _ollama_live() -> bool:
    url = os.environ.get("SOMM_OLLAMA_URL", "http://localhost:11434")
    try:
        r = httpx.get(f"{url}/api/tags", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_live(), reason="no local ollama")
def test_ollama_live_generate(tmp_path):
    """Real call to local ollama. Verifies the full happy path end-to-end."""
    cfg = _tmp_config(tmp_path)
    cfg.ollama_model = os.environ.get("SOMM_OLLAMA_MODEL", "gemma4:e4b")
    llm = SommLLM(config=cfg)

    result = llm.generate(
        prompt="Reply with exactly: pong",
        workload="live_ping",
        max_tokens=8,
        temperature=0.0,
    )
    llm.close()

    assert result.provider == "ollama"
    assert result.call_id
    assert result.tokens_out >= 0
    # Don't assert on text content — models vary. Just require a row landed.
    call = llm.repo.get_call(result.call_id)
    assert call is not None
    assert call.provider == "ollama"

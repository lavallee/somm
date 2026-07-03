"""SommLLM.embed() — local ollama, no fallback, telemetry mirrors generate()."""

from __future__ import annotations

from pathlib import Path

import pytest
from somm.client import SommLLM
from somm.providers.base import (
    ProviderHealth,
    SommEmbedResponse,
    SommResponse,
)
from somm_core import Outcome
from somm_core.config import Config


class FakeOllamaProvider:
    """Stand-in for OllamaProvider in tests. Mirrors the duck-typed
    SommProvider surface plus an embed() method."""

    name = "ollama"
    default_embed_model = "nomic-embed-text"

    def __init__(self, vector: list[float] | None = None, fail: Exception | None = None):
        self._vector = vector or [0.1, 0.2, 0.3]
        self._fail = fail
        self.last_request = None

    def generate(self, request):  # pragma: no cover — embed tests don't exercise generate
        return SommResponse(text="", model="x", tokens_in=0, tokens_out=0, latency_ms=0)

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return len(text) // 4 or 1

    def embed(self, request):
        self.last_request = request
        if self._fail is not None:
            raise self._fail
        return SommEmbedResponse(
            embedding=list(self._vector),
            model=request.model or self.default_embed_model,
            tokens_in=len(request.text) // 4 or 1,
            latency_ms=2,
            raw={"fake": True},
        )


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "embed_test"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def test_embed_happy_path_returns_vector(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeOllamaProvider(vector=[0.5, 0.5, 0.5, 0.5])
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        result = llm.embed("hello world", workload="vault_tiling")
        assert result.outcome == Outcome.OK
        assert result.embedding == [0.5, 0.5, 0.5, 0.5]
        assert result.dim == 4
        assert result.provider == "ollama"
        assert result.model == "nomic-embed-text"
        assert result.tokens_in > 0
        assert result.error_detail is None
        assert result.call_id  # uuid set
    finally:
        llm.close()


def test_embed_custom_model_honored(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeOllamaProvider()
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        result = llm.embed("x", workload="ad_hoc", model="all-minilm")
        assert result.model == "all-minilm"
        assert fake.last_request.model == "all-minilm"
    finally:
        llm.close()


def test_embed_failure_returns_upstream_error(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeOllamaProvider(fail=RuntimeError("connection refused"))
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        result = llm.embed("hi", workload="ad_hoc")
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert result.embedding == []
        assert result.dim == 0
        assert result.error_kind == "RuntimeError"
        assert "connection refused" in result.error_detail
    finally:
        llm.close()


def test_embed_telemetry_row_lands_in_calls_db(tmp_path):
    """Every embed call writes a row to calls.sqlite, parity with generate."""
    cfg = _tmp_config(tmp_path)
    fake = FakeOllamaProvider(vector=[0.1] * 10)
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        result = llm.embed("payload", workload="vault_tiling")
        llm._writer.flush(timeout=2.0)
    finally:
        llm.close()

    import sqlite3
    db_path = cfg.db_dir / "calls.sqlite"
    assert db_path.exists()
    rows = sqlite3.connect(db_path).execute(
        "SELECT id, provider, model, outcome, tokens_in, tokens_out FROM calls"
    ).fetchall()
    assert len(rows) == 1
    row_id, provider, model, outcome, tokens_in, tokens_out = rows[0]
    assert row_id == result.call_id
    assert provider == "ollama"
    assert model == "nomic-embed-text"
    assert outcome == "ok"
    assert tokens_out == 0  # embeddings have no output tokens
    assert tokens_in > 0


def test_embed_failure_telemetry_row_carries_error_detail(tmp_path):
    cfg = _tmp_config(tmp_path)
    fake = FakeOllamaProvider(fail=ValueError("bad input"))
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        llm.embed("payload", workload="ad_hoc")
        llm._writer.flush(timeout=2.0)
    finally:
        llm.close()

    import sqlite3
    db_path = cfg.db_dir / "calls.sqlite"
    rows = sqlite3.connect(db_path).execute(
        "SELECT outcome, error_kind, error_detail FROM calls"
    ).fetchall()
    assert len(rows) == 1
    outcome, error_kind, error_detail = rows[0]
    assert outcome == "upstream_error"
    assert error_kind == "ValueError"
    assert "bad input" in error_detail


def test_embed_raises_when_no_ollama_provider_configured(tmp_path):
    """Without ollama in the chain, embed() refuses cleanly. The error
    message points at $SOMM_PROVIDER_ORDER as the likely culprit."""
    cfg = _tmp_config(tmp_path)

    class NotOllama:
        name = "fake"
        def generate(self, r):  # pragma: no cover
            raise NotImplementedError
        def stream(self, r):  # pragma: no cover
            yield
        def health(self):
            return ProviderHealth(available=True)
        def models(self):
            return []
        def estimate_tokens(self, t, m):
            return 1

    llm = SommLLM(config=cfg, providers=[NotOllama()])
    try:
        with pytest.raises(ValueError, match="ollama provider"):
            llm.embed("hi", workload="ad_hoc")
    finally:
        llm.close()


# ---- live ollama (auto-skip when unreachable) -----------------------------


def _ollama_reachable(url: str | None = None, model: str = "nomic-embed-text") -> bool:
    import os

    url = url or os.environ.get("SOMM_OLLAMA_URL", "http://127.0.0.1:11434")
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.5) as r:
            data = json.loads(r.read().decode())
        names = {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}
        return any(n == model or n.startswith(f"{model}:") for n in names)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


@pytest.mark.skipif(
    not _ollama_reachable(),
    reason="local ollama not reachable or nomic-embed-text not pulled",
)
def test_embed_against_live_ollama(tmp_path):
    """Smoke test the real ollama path. Auto-skips when ollama isn't running
    or the model isn't pulled — matches the existing live-provider pattern
    in this test suite."""
    from somm.providers.ollama import OllamaProvider

    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[OllamaProvider()])
    try:
        result = llm.embed("the quick brown fox", workload="ad_hoc")
        detail = (result.error_detail or "").lower()
        if "503" in detail or "server busy" in detail:
            pytest.skip("local ollama contended (busy/503)")
        assert result.outcome == Outcome.OK
        assert result.dim > 0
        assert all(isinstance(v, float) for v in result.embedding)
        assert result.provider == "ollama"
        assert result.model == "nomic-embed-text"
    finally:
        llm.close()

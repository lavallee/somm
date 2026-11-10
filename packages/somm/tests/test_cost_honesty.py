"""Recorded cost is honest: one price table, cache tokens at cache rates,
unknown prices recorded as unknown rather than $0, and streamed rows priced
like generated ones."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommChunk, SommResponse
from somm_core.config import Config
from somm_core.pricing import backfill_costs, cost_for_call, sync_bundled_pricing, write_intel
from somm_core.repository import Repository


class UsageProvider:
    """Fake provider that reports a fixed usage block, Anthropic-shaped:
    ``tokens_in`` is the whole prompt, cache slices included."""

    def __init__(self, name: str = "anthropic", model: str = "test-model", raw: dict | None = None):
        self.name = name
        self.model = model
        self.raw = raw

    def generate(self, request):
        return SommResponse(
            text="ok",
            model=request.model or self.model,
            tokens_in=1_000_000,
            tokens_out=0,
            latency_ms=5,
            raw=self.raw,
        )

    def stream(self, request):
        yield SommChunk(text="ok", done=False)
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


def _config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "cost"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _row(llm: SommLLM, call_id: str) -> dict:
    llm._writer.flush(timeout=2.0)
    with sqlite3.connect(llm.config.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
    assert row is not None
    return dict(row)


_CACHED_USAGE = {
    "usage": {
        "input_tokens": 100_000,
        "cache_read_input_tokens": 800_000,
        "cache_creation_input_tokens": 100_000,
    }
}


# ---- cache tokens ------------------------------------------------------------


def test_cache_slices_priced_at_their_own_rates(tmp_path):
    repo = Repository(tmp_path / "c.sqlite")
    write_intel(
        repo, "anthropic", "m", 4.0, 20.0, None, None, "manual",
        cache_read_per_1m=0.4, cache_write_per_1m=5.0,
    )
    # 100k uncached at $4, 800k read at $0.40, 100k written at $5.
    cost = cost_for_call(
        repo, "anthropic", "m", 1_000_000, 0, cache_read_tokens=800_000, cache_write_tokens=100_000
    )
    assert cost == pytest.approx(0.4 + 0.32 + 0.5)


def test_missing_cache_rate_falls_back_to_the_input_rate(tmp_path):
    repo = Repository(tmp_path / "c.sqlite")
    write_intel(repo, "openai", "m", 2.0, 8.0, None, None, "manual")
    cost = cost_for_call(repo, "openai", "m", 1_000_000, 0, cache_read_tokens=500_000)
    assert cost == pytest.approx(2.0)


def test_bundle_carries_cache_rates(tmp_path):
    repo = Repository(tmp_path / "c.sqlite")
    sync_bundled_pricing(repo)
    with repo._open() as conn:
        read, write = conn.execute(
            "SELECT price_cache_read_per_1m, price_cache_write_per_1m FROM model_intel "
            "WHERE provider = 'anthropic' AND model = 'claude-opus-5-5'"
        ).fetchone()
    assert read is not None and read < 4.0
    assert write is not None and write > 4.0


def test_generate_prices_anthropic_cache_tokens(tmp_path):
    llm = SommLLM(config=_config(tmp_path), providers=[UsageProvider(raw=_CACHED_USAGE)])
    try:
        write_intel(
            llm.repo, "anthropic", "test-model", 4.0, 20.0, None, None, "manual",
            cache_read_per_1m=0.4, cache_write_per_1m=5.0,
        )
        result = llm.generate("hi", workload="w", provider="anthropic")
        row = _row(llm, result.call_id)
    finally:
        llm.close()
    # Was $4.00: every prompt token at the full input rate.
    assert result.cost_usd == pytest.approx(1.22)
    assert row["cost_usd"] == pytest.approx(1.22)


def test_backfill_uses_cache_rates(tmp_path):
    llm = SommLLM(config=_config(tmp_path), providers=[UsageProvider(raw=_CACHED_USAGE)])
    try:
        result = llm.generate("hi", workload="w", provider="anthropic")
        _row(llm, result.call_id)
        write_intel(
            llm.repo, "anthropic", "test-model", 4.0, 20.0, None, None, "manual",
            cache_read_per_1m=0.4, cache_write_per_1m=5.0,
        )
        n, total = backfill_costs(llm.repo)
        row = _row(llm, result.call_id)
    finally:
        llm.close()
    assert n == 1
    assert total == pytest.approx(1.22)
    assert row["cost_usd"] == pytest.approx(1.22)

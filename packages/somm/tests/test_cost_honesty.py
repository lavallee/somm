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


# ---- unknown is not zero -------------------------------------------------------


def test_self_hosted_endpoint_records_unknown_not_zero(tmp_path):
    """A self-hosted endpoint has no price: the row says unknown, not $0."""
    llm = SommLLM(config=_config(tmp_path), providers=[UsageProvider(name="gpu-box")])
    try:
        result = llm.generate("hi", workload="w", provider="gpu-box")
        row = _row(llm, result.call_id)
    finally:
        llm.close()
    assert result.cost_usd is None
    assert row["cost_usd"] is None
    assert (row["cost_basis"], row["cost_kind"], row["cost_accuracy"], row["cost_source"]) == (
        "unknown",
        "unknown",
        "unknown",
        "unknown:no-price",
    )


def test_free_routes_stay_zero_and_say_why(tmp_path):
    llm = SommLLM(
        config=_config(tmp_path),
        providers=[
            UsageProvider(name="ollama", model="qwen-local"),
            UsageProvider(name="openrouter", model="vendor/some-model:free"),
        ],
    )
    try:
        local = llm.generate("hi", workload="w", provider="ollama")
        free = llm.generate("hi", workload="w", provider="openrouter")
        local_row, free_row = _row(llm, local.call_id), _row(llm, free.call_id)
    finally:
        llm.close()
    assert local_row["cost_usd"] == 0.0
    assert local_row["cost_source"] == "local-included-unpriced"
    assert free_row["cost_usd"] == 0.0
    assert free_row["cost_source"] == "openrouter:free-model"


def test_budget_counts_known_costs_only(tmp_path):
    """Unknown-cost calls do not count toward a daily cap (they are skipped,
    not priced at $0); a priced call still trips it. No crash on NULL."""
    cfg = _config(tmp_path)
    cfg.budget_fail_closed = True
    llm = SommLLM(
        config=cfg, providers=[UsageProvider(name="gpu-box"), UsageProvider(name="anthropic")]
    )
    try:
        llm.repo.register_workload(name="capped", project=cfg.project, budget_cap_usd_daily=0.5)
        for _ in range(3):
            assert llm.generate("hi", workload="capped", provider="gpu-box").cost_usd is None
            llm._writer.flush(timeout=2.0)
        write_intel(llm.repo, "anthropic", "test-model", 1.0, 1.0, None, None, "manual")
        assert llm.generate("hi", workload="capped", provider="anthropic").cost_usd == 1.0
        llm._writer.flush(timeout=2.0)
        from somm.errors import SommBudgetExceeded

        with pytest.raises(SommBudgetExceeded):
            llm.generate("hi", workload="capped", provider="gpu-box")
    finally:
        llm.close()


def test_frontier_mean_cost_skips_unknown(tmp_path):
    """An unpriced model's mean cost is unknown, not $0, so it cannot look free."""
    llm = SommLLM(config=_config(tmp_path), providers=[UsageProvider(name="gpu-box")])
    try:
        llm.generate("hi", workload="w", provider="gpu-box")
        llm._writer.flush(timeout=2.0)
        wl = llm.repo.workload_by_name("w", llm.config.project)
        rows = llm.repo.workload_frontier(wl.id, since_days=1)
    finally:
        llm.close()
    assert rows[0]["mean_cost_per_ok_call"] is None


def test_search_calls_keeps_unknown_cost(tmp_path):
    from somm_core.read import search_calls

    llm = SommLLM(config=_config(tmp_path), providers=[UsageProvider(name="gpu-box")])
    try:
        llm.generate("hi", workload="w", provider="gpu-box")
        llm._writer.flush(timeout=2.0)
        rows = search_calls(llm.config.db_path, llm.config.project)
    finally:
        llm.close()
    assert rows[0]["cost_usd"] is None


def test_migration_0029_lets_an_existing_ledger_hold_null(tmp_path):
    """0001 declared cost_usd NOT NULL; 0029 relaxes it in place, keeping rows,
    and a connection opened before the migration sees the new definition."""
    from somm_core.schema import _list_migrations, ensure_schema
    from somm_core.version import SCHEMA_VERSION

    path = tmp_path / "v28.sqlite"
    insert = (
        "INSERT INTO calls (id, ts, project, provider, model, cost_usd, prompt_hash, "
        "response_hash) VALUES (?, '2026-11-01', 'p', 'x', 'y', ?, 'h', 'h')"
    )
    with sqlite3.connect(path) as conn:
        for version, migration in _list_migrations():
            if version > 28:
                continue
            conn.executescript(migration.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        conn.execute(insert, ("old", 0.25))
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("null-before", None))
    early = sqlite3.connect(path)
    with sqlite3.connect(path) as conn:
        assert ensure_schema(conn) == SCHEMA_VERSION
        assert ensure_schema(conn) == SCHEMA_VERSION  # idempotent
    early.execute(insert, ("null-after", None))
    early.commit()
    assert early.execute("SELECT id, cost_usd FROM calls ORDER BY id").fetchall() == [
        ("null-after", None),
        ("old", 0.25),
    ]
    assert early.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    early.close()


# ---- streamed rows -----------------------------------------------------------


class DefaultModelStreamer(UsageProvider):
    """Streams under its default model, naming it on the final chunk the way
    the real adapters do."""

    def stream(self, request):
        yield SommChunk(text="ok", done=False)
        yield SommChunk(text="", done=True, model=request.model or self.model)


def test_stream_without_a_model_is_priced_like_generate(tmp_path):
    """No model given: the row used to say model=<provider name>, which has no
    price. It now names the model the provider served and prices it."""
    llm = SommLLM(
        config=_config(tmp_path), providers=[DefaultModelStreamer(name="anthropic")]
    )
    try:
        write_intel(llm.repo, "anthropic", "test-model", 1.0, 1.0, None, None, "manual")
        assert "".join(llm.stream("hello world", workload="w", provider="anthropic")) == "ok"
        llm._writer.flush(timeout=2.0)
        with sqlite3.connect(llm.config.db_path) as conn:
            model, cost, basis = conn.execute(
                "SELECT model, cost_usd, cost_basis FROM calls"
            ).fetchone()
    finally:
        llm.close()
    assert model == "test-model"
    assert cost is not None and cost > 0
    assert basis == "computed"


def test_openai_compat_stream_names_its_model(monkeypatch):
    import httpx
    from somm.providers.base import SommRequest
    from somm.providers.openai import OpenAIProvider

    body = 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'

    class _Client(httpx.Client):
        def __init__(self, *a, **kw):
            kw.pop("transport", None)
            transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
            super().__init__(*a, transport=transport, **kw)

    monkeypatch.setattr(httpx, "Client", _Client)
    provider = OpenAIProvider(api_key="k", default_model="gpt-test")
    chunks = list(provider.stream(SommRequest(prompt="hi")))
    assert chunks[-1].done and chunks[-1].model == "gpt-test"


def test_ollama_stream_names_its_model(monkeypatch):
    import httpx
    from somm.providers.base import SommRequest
    from somm.providers.ollama import OllamaProvider

    body = '{"message": {"content": "hi"}, "done": false}\n{"done": true}\n'

    class _Client(httpx.Client):
        def __init__(self, *a, **kw):
            kw.pop("transport", None)
            transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
            super().__init__(*a, transport=transport, **kw)

    monkeypatch.setattr(httpx, "Client", _Client)
    provider = OllamaProvider(base_url="http://127.0.0.1:1", default_model="qwen-local")
    chunks = list(provider.stream(SommRequest(prompt="hi")))
    assert chunks[-1].done and chunks[-1].model == "qwen-local"

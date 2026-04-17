"""Tests for the model-intel worker + pricing helpers."""

from __future__ import annotations

from pathlib import Path

import httpx
from somm_core import cost_for_call, list_intel, write_intel
from somm_core.config import Config
from somm_core.repository import Repository
from somm_service.workers.model_intel import (
    STATIC_PRICING,
    ModelIntelWorker,
)


def _tmp_repo(tmp_path: Path) -> Repository:
    cfg = Config()
    cfg.db_dir = tmp_path / ".somm"
    return Repository(cfg.db_path)


# ---------------------------------------------------------------------------
# pricing helpers


def test_write_and_list_intel_roundtrip(tmp_path):
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo,
        provider="openai",
        model="gpt-4o-mini",
        price_in_per_1m=0.15,
        price_out_per_1m=0.6,
        context_window=128_000,
        capabilities={"foo": "bar"},
        source="static",
    )
    rows = list_intel(repo, provider="openai")
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "gpt-4o-mini"
    assert r["price_in_per_1m"] == 0.15
    assert r["capabilities"] == {"foo": "bar"}


def test_write_intel_upsert(tmp_path):
    repo = _tmp_repo(tmp_path)
    write_intel(repo, "openai", "gpt-4o-mini", 0.15, 0.6, 128_000, None, "v1")
    write_intel(repo, "openai", "gpt-4o-mini", 0.20, 0.8, 128_000, None, "v2")
    rows = list_intel(repo, provider="openai")
    assert len(rows) == 1
    assert rows[0]["price_in_per_1m"] == 0.20
    assert rows[0]["source"] == "v2"


def test_cost_for_call_computes_from_prices(tmp_path):
    repo = _tmp_repo(tmp_path)
    write_intel(repo, "openai", "gpt-4o-mini", 0.15, 0.6, 128_000, None, "static")
    # 1000 in-tokens @ $0.15/1M + 500 out-tokens @ $0.6/1M
    # = 0.15 * 1000 / 1e6 + 0.6 * 500 / 1e6 = 0.00015 + 0.0003 = 0.00045
    cost = cost_for_call(repo, "openai", "gpt-4o-mini", 1000, 500)
    assert cost == round(0.00045, 8)


def test_cost_for_call_returns_zero_when_missing(tmp_path):
    repo = _tmp_repo(tmp_path)
    assert cost_for_call(repo, "unknown", "unknown", 1000, 500) == 0.0


def test_cost_for_call_zero_for_local(tmp_path):
    repo = _tmp_repo(tmp_path)
    write_intel(repo, "ollama", "gemma4:e4b", 0.0, 0.0, None, None, "ollama-local")
    assert cost_for_call(repo, "ollama", "gemma4:e4b", 10000, 5000) == 0.0


# ---------------------------------------------------------------------------
# ModelIntelWorker — static + mocked http


def test_run_once_populates_static_pricing(tmp_path, monkeypatch):
    """When OpenRouter + ollama fail, static still lands."""
    repo = _tmp_repo(tmp_path)
    worker = ModelIntelWorker(repo)

    # Monkey-patch httpx.Client so BOTH openrouter + ollama fail.
    class _FailClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **kw):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "Client", _FailClient)
    summary = worker.run_once()
    assert summary["static"] == len(STATIC_PRICING)
    assert summary["openrouter"] == 0
    assert summary["ollama"] == 0
    assert len(summary["errors"]) == 2

    # Static entries landed
    rows = list_intel(repo, provider="anthropic")
    assert any(r["model"] == "claude-haiku-4-5-20251001" for r in rows)


def test_openrouter_scrape(tmp_path, monkeypatch):
    """OpenRouter response shape parsed + written."""
    repo = _tmp_repo(tmp_path)

    or_response = {
        "data": [
            {
                "id": "google/gemma-3-27b-it:free",
                "context_length": 8192,
                "pricing": {"prompt": "0", "completion": "0"},
                "top_provider": {"max_completion_tokens": 2048},
                "architecture": {"modality": "text"},
                "modality": "text",
            },
            {
                "id": "meta-llama/llama-3.3-70b-instruct",
                "context_length": 131072,
                "pricing": {"prompt": "0.0000006", "completion": "0.0000008"},
                "top_provider": {},
                "architecture": {"modality": "text"},
                "modality": "text",
            },
        ]
    }

    def handler(request):
        if "openrouter.ai" in str(request.url):
            return httpx.Response(200, json=or_response)
        # ollama probe
        raise httpx.ConnectError("no ollama in this test")

    _patch_client_with(handler, monkeypatch)

    worker = ModelIntelWorker(repo)
    summary = worker.run_once()
    assert summary["openrouter"] == 2

    rows = list_intel(repo, provider="openrouter")
    models = {r["model"]: r for r in rows}
    assert "google/gemma-3-27b-it:free" in models
    assert models["google/gemma-3-27b-it:free"]["price_in_per_1m"] == 0.0
    # $0.0000006 per token → $0.6 per 1M tokens
    assert round(models["meta-llama/llama-3.3-70b-instruct"]["price_in_per_1m"], 4) == 0.6
    assert round(models["meta-llama/llama-3.3-70b-instruct"]["price_out_per_1m"], 4) == 0.8


def test_ollama_probe(tmp_path, monkeypatch):
    """Ollama /api/tags models written with price=0."""
    repo = _tmp_repo(tmp_path)

    def handler(request):
        if "openrouter.ai" in str(request.url):
            raise httpx.ConnectError("no openrouter in this test")
        # ollama
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "gemma4:e4b",
                        "size": 9608350718,
                        "details": {"family": "gemma", "parameter_size": "e4b"},
                    },
                    {
                        "name": "qwen2.5:7b",
                        "size": 4000000000,
                        "details": {"family": "qwen2.5", "parameter_size": "7B"},
                    },
                ]
            },
        )

    _patch_client_with(handler, monkeypatch)
    worker = ModelIntelWorker(repo, openrouter_url="https://openrouter.ai/fail")
    summary = worker.run_once()
    assert summary["ollama"] == 2

    rows = list_intel(repo, provider="ollama")
    names = {r["model"] for r in rows}
    assert "gemma4:e4b" in names
    assert "qwen2.5:7b" in names
    # local = free
    for r in rows:
        assert r["price_in_per_1m"] == 0.0
        assert r["price_out_per_1m"] == 0.0
        assert r["source"] == "ollama-local"


# ---------------------------------------------------------------------------
# Integration: library.generate() computes cost via model_intel


def test_library_generate_computes_cost_from_intel(tmp_path, monkeypatch):
    """After an intel refresh, .generate() writes non-zero cost_usd."""
    from somm.client import SommLLM
    from somm.providers.base import ProviderHealth, SommResponse

    repo = _tmp_repo(tmp_path)
    # Seed intel with a known price
    write_intel(
        repo,
        "fake",
        "fake-m",
        price_in_per_1m=1.0,
        price_out_per_1m=4.0,
        context_window=None,
        capabilities=None,
        source="test",
    )

    cfg = Config()
    cfg.project = "cost-test"
    cfg.db_dir = tmp_path / ".somm"

    class FakeProvider:
        name = "fake"

        def generate(self, request):
            return SommResponse(
                text="ok",
                model="fake-m",
                tokens_in=10_000,
                tokens_out=1_000,
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

    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        result = llm.generate("hi", workload="cost_w")
        # 10000 in @ $1/1M + 1000 out @ $4/1M = 0.01 + 0.004 = 0.014
        assert abs(result.cost_usd - 0.014) < 1e-9
    finally:
        llm.close()


# ---------------------------------------------------------------------------


def _patch_client_with(handler, monkeypatch):
    transport = httpx.MockTransport(handler)

    class _MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MockedClient)

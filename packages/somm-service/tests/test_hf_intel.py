"""HuggingFace intel worker — canonicalization, enrichment merge, feature flag."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from somm_core import write_intel
from somm_core.config import Config
from somm_core.pricing import list_intel
from somm_core.repository import Repository
from somm_service.workers.hf_intel import (
    HF_PIPELINE_MAP,
    HuggingFaceIntelWorker,
    canonical_hf_id,
)


def _tmp_repo(tmp_path: Path) -> Repository:
    cfg = Config()
    cfg.db_dir = tmp_path / ".somm"
    return Repository(cfg.db_path)


# ---------------------------------------------------------------------------
# canonical_hf_id


def test_canonical_hf_id_strips_tier_suffix():
    assert canonical_hf_id("openrouter", "google/gemma-3-27b-it:free") == "google/gemma-3-27b-it"
    assert canonical_hf_id("openrouter", "meta-llama/llama-3.3-70b:nitro") == "meta-llama/llama-3.3-70b"


def test_canonical_hf_id_skips_proprietary_orgs():
    """Anthropic/OpenAI don't publish weights on HF — skip 404-bound fetches."""
    assert canonical_hf_id("openrouter", "anthropic/claude-sonnet-4-6") is None
    assert canonical_hf_id("openrouter", "openai/gpt-4o-mini") is None


def test_canonical_hf_id_skips_meta_routers():
    assert canonical_hf_id("openrouter", "openrouter/auto") is None
    assert canonical_hf_id("openrouter", "openrouter/free") is None


def test_canonical_hf_id_skips_non_openrouter_providers():
    """Ollama/static mapping deferred — no canonical alias table yet."""
    assert canonical_hf_id("ollama", "llama3.2-vision:11b") is None
    assert canonical_hf_id("anthropic", "claude-opus-4-7") is None


def test_canonical_hf_id_rejects_malformed_ids():
    assert canonical_hf_id("openrouter", "bad-no-slash") is None


# ---------------------------------------------------------------------------
# Feature flag


def test_worker_disabled_by_default(tmp_path):
    repo = _tmp_repo(tmp_path)
    worker = HuggingFaceIntelWorker(repo)
    summary = worker.run_once()
    assert summary["disabled"] is True
    assert summary["enriched"] == 0


def test_worker_honours_env_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMM_ENABLE_HF_INTEL", "1")
    repo = _tmp_repo(tmp_path)
    worker = HuggingFaceIntelWorker(repo)
    assert worker.enabled is True


# ---------------------------------------------------------------------------
# Enrichment


def test_enriches_openrouter_row_with_pipeline_tag(tmp_path, monkeypatch):
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo, "openrouter", "google/gemma-3-27b-it:free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=262_144,
        capabilities={"modality": "text+image->text"},
        source="openrouter",
    )

    def handler(request):
        assert "google/gemma-3-27b-it" in str(request.url)
        assert ":free" not in str(request.url)   # suffix stripped
        return httpx.Response(
            200,
            json={
                "id": "google/gemma-3-27b-it",
                "pipeline_tag": "image-text-to-text",
                "tags": ["conversational", "multimodal", "vision"],
            },
        )

    _patch_client_with(handler, monkeypatch)

    worker = HuggingFaceIntelWorker(repo, enabled=True)
    summary = worker.run_once()
    assert summary["enriched"] == 1

    caps = list_intel(repo, provider="openrouter")[0]["capabilities"]
    # Primary OpenRouter data preserved.
    assert caps["modality"] == "text+image->text"
    # HF block merged under its own sub-key.
    hf = caps["hf"]
    assert hf["pipeline_tag"] == "image-text-to-text"
    assert "conversational" in hf["tags"]
    # Derived modality hints from the pipeline_tag map.
    assert hf["input_modalities"] == ["image", "text"]
    assert hf["output_modalities"] == ["text"]


def test_enrichment_merge_preserves_existing_keys(tmp_path, monkeypatch):
    """Second run must not clobber unrelated capability keys."""
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo, "openrouter", "some/model",
        0.0, 0.0, 100_000,
        capabilities={"modality": "text->text", "custom": {"foo": "bar"}},
        source="openrouter",
    )

    def handler(request):
        return httpx.Response(200, json={"id": "some/model", "pipeline_tag": "text-generation", "tags": []})

    _patch_client_with(handler, monkeypatch)

    HuggingFaceIntelWorker(repo, enabled=True).run_once()
    caps = list_intel(repo, provider="openrouter")[0]["capabilities"]
    assert caps["modality"] == "text->text"
    assert caps["custom"] == {"foo": "bar"}
    assert caps["hf"]["pipeline_tag"] == "text-generation"


def test_enrichment_skips_404(tmp_path, monkeypatch):
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo, "openrouter", "ghost/model",
        0.0, 0.0, 100_000, {"modality": "text->text"}, "openrouter",
    )

    def handler(request):
        return httpx.Response(404, json={"error": "not found"})

    _patch_client_with(handler, monkeypatch)
    summary = HuggingFaceIntelWorker(repo, enabled=True).run_once()
    assert summary["enriched"] == 0
    assert summary["skipped"] >= 1


def test_enrichment_tolerates_http_errors(tmp_path, monkeypatch):
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo, "openrouter", "flaky/model",
        0.0, 0.0, 100_000, None, "openrouter",
    )

    def handler(request):
        raise httpx.ConnectError("network down")

    _patch_client_with(handler, monkeypatch)
    summary = HuggingFaceIntelWorker(repo, enabled=True).run_once()
    assert summary["errors"] >= 1
    # Row still present, capabilities unchanged.
    caps = list_intel(repo, provider="openrouter")[0]["capabilities"]
    assert caps is None or "hf" not in caps


def test_enrichment_skips_proprietary(tmp_path, monkeypatch):
    """Anthropic rows should never trigger HF fetches."""
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo, "anthropic", "claude-opus-4-7",
        15.0, 75.0, 200_000, None, "static",
    )

    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(500)   # would fail if reached

    _patch_client_with(handler, monkeypatch)
    summary = HuggingFaceIntelWorker(repo, enabled=True).run_once()
    assert summary["enriched"] == 0
    assert calls == []


def test_pipeline_map_covers_core_tags():
    """Regression guard for the static map — missing a common tag means
    silent capability loss."""
    for tag in (
        "text-generation", "image-text-to-text", "text-to-image",
        "text-to-speech", "automatic-speech-recognition",
    ):
        assert tag in HF_PIPELINE_MAP


# ---------------------------------------------------------------------------


def _patch_client_with(handler, monkeypatch):
    transport = httpx.MockTransport(handler)

    class _MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MockedClient)

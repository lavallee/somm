"""Multimodal prompt widening + capability-aware routing."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from somm.capabilities import model_has_capability, provider_can_serve
from somm.client import SommLLM
from somm.errors import SommNoCapableProvider
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)
from somm.routing import ProviderHealthTracker, Router
from somm_core.config import Config
from somm_core.parse import (
    estimate_prompt_tokens,
    image_prompt,
    infer_capabilities,
    prompt_preview,
    stable_hash,
    text_prompt,
)
from somm_core.pricing import write_intel
from somm_core.repository import Repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "mm"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _tmp_repo(tmp_path: Path) -> Repository:
    cfg = Config()
    cfg.project = "mm"
    cfg.db_dir = tmp_path / ".somm"
    return Repository(cfg.db_path)


class RecordingProvider:
    """Captures the request it receives; returns a canned response."""

    def __init__(self, name: str = "rec", default_model: str = "rec-1") -> None:
        self.name = name
        self.default_model = default_model
        self.last_request: SommRequest | None = None

    def generate(self, request: SommRequest) -> SommResponse:
        self.last_request = request
        return SommResponse(
            text="ok",
            model=request.model or self.default_model,
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            raw=None,
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:  # pragma: no cover
        yield SommChunk(text="ok", done=True)

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def models(self) -> list[SommModel]:
        return []

    def estimate_tokens(self, text, model):
        return estimate_prompt_tokens(text)


# ---------------------------------------------------------------------------
# Helpers: text_prompt / image_prompt / infer_capabilities / preview
# ---------------------------------------------------------------------------


def test_text_prompt_builds_single_block():
    assert text_prompt("hi") == [{"type": "text", "text": "hi"}]


def test_image_prompt_embeds_base64():
    blocks = image_prompt("caption this", b"\x89PNG\r\n", media_type="image/png")
    assert blocks[0] == {"type": "text", "text": "caption this"}
    assert blocks[1]["type"] == "image"
    src = blocks[1]["source"]
    assert src["type"] == "base64"
    assert src["media_type"] == "image/png"
    assert src["data"]  # non-empty


def test_image_prompt_url_variant():
    blocks = image_prompt("see", url="https://example.com/x.png")
    assert blocks[1]["source"] == {"type": "url", "url": "https://example.com/x.png"}


def test_image_prompt_requires_bytes_or_url():
    with pytest.raises(ValueError):
        image_prompt("nothing")


def test_infer_capabilities_on_text_only():
    assert infer_capabilities("just text") == []
    assert infer_capabilities(text_prompt("still text")) == []


def test_infer_capabilities_picks_up_vision():
    caps = infer_capabilities(image_prompt("look", b"img"))
    assert caps == ["vision"]


def test_stable_hash_handles_list_prompt():
    listed = image_prompt("hi", b"img")
    h1 = stable_hash(listed)
    h2 = stable_hash(listed)
    assert h1 == h2
    assert h1 != stable_hash("plain string")


def test_prompt_preview_elides_image_payload():
    blocks = image_prompt("look at this", b"x" * 300)
    preview = prompt_preview(blocks)
    assert "look at this" in preview
    assert "[IMAGE" in preview
    # Never leaks the base64 body itself
    assert "xxxxxxxx" not in preview


def test_estimate_prompt_tokens_adds_per_image_cost():
    only_text = estimate_prompt_tokens("a" * 40)
    with_image = estimate_prompt_tokens(image_prompt("a" * 40, b"img"))
    assert with_image > only_text + 100  # image addend dominates


# ---------------------------------------------------------------------------
# Pass-through: providers receive list-shaped content unchanged
# ---------------------------------------------------------------------------


def test_request_passes_list_prompt_to_provider(tmp_path):
    cfg = _tmp_config(tmp_path)
    rec = RecordingProvider()
    llm = SommLLM(config=cfg, providers=[rec])

    blocks = image_prompt("describe", b"bytes")
    llm.generate(blocks, workload="vision_test")
    llm.close()

    assert rec.last_request is not None
    assert rec.last_request.prompt == blocks
    assert "vision" in rec.last_request.capabilities_required


def test_workload_capabilities_merge_with_auto_inferred(tmp_path):
    cfg = _tmp_config(tmp_path)
    rec = RecordingProvider()
    llm = SommLLM(config=cfg, providers=[rec])

    llm.register_workload(name="long_ctx", capabilities_required=["long_context"])
    llm.generate("hi", workload="long_ctx")
    llm.close()

    caps = rec.last_request.capabilities_required
    assert "long_context" in caps


# ---------------------------------------------------------------------------
# Capability lookup against model_intel
# ---------------------------------------------------------------------------


def test_model_has_capability_vision_by_name(tmp_path):
    repo = _tmp_repo(tmp_path)
    # claude-opus-4-7 is name-hint vision-capable even without intel row.
    assert model_has_capability(repo, "anthropic", "claude-opus-4-7", "vision") is True


def test_model_has_capability_unknown_ollama_returns_none(tmp_path):
    repo = _tmp_repo(tmp_path)
    assert (
        model_has_capability(repo, "ollama", "some-random-model", "vision") is None
    )


def test_model_has_capability_openrouter_modality(tmp_path):
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo,
        provider="openrouter",
        model="vision-route/x",
        price_in_per_1m=0,
        price_out_per_1m=0,
        context_window=None,
        capabilities={"modality": "text+image->text"},
        source="test",
    )
    assert model_has_capability(repo, "openrouter", "vision-route/x", "vision") is True


def test_model_has_capability_explicit_flag_wins(tmp_path):
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo,
        provider="custom",
        model="m",
        price_in_per_1m=0,
        price_out_per_1m=0,
        context_window=None,
        capabilities={"vision": True},
        source="test",
    )
    assert model_has_capability(repo, "custom", "m", "vision") is True


def test_provider_can_serve_returns_reason(tmp_path):
    repo = _tmp_repo(tmp_path)
    # openai/gpt-4o-mini: vision via name hint → ok.
    ok, _ = provider_can_serve(repo, "openai", "gpt-4o-mini", ["vision"])
    assert ok


# ---------------------------------------------------------------------------
# Router: capability filtering
# ---------------------------------------------------------------------------


class StubProvider(RecordingProvider):
    def __init__(self, name: str, default_model: str) -> None:
        super().__init__(name=name, default_model=default_model)


def test_router_raises_when_no_provider_capable(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    # Seed model_intel so the lookup is decisive.
    write_intel(
        repo,
        provider="ollama",
        model="text-only",
        price_in_per_1m=0,
        price_out_per_1m=0,
        context_window=None,
        capabilities={"vision": False},
        source="test",
    )
    p = StubProvider("ollama", "text-only")
    router = Router([p], tr)

    with pytest.raises(SommNoCapableProvider) as exc:
        router.dispatch(
            SommRequest(
                prompt="hi",
                capabilities_required=["vision"],
            )
        )
    assert exc.value.required == ["vision"]
    assert exc.value.skipped and exc.value.skipped[0][0] == "ollama"


def test_router_skips_incapable_provider_and_uses_capable(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    write_intel(
        repo,
        provider="ollama",
        model="text-only",
        price_in_per_1m=0,
        price_out_per_1m=0,
        context_window=None,
        capabilities={"vision": False},
        source="test",
    )
    p_text = StubProvider("ollama", "text-only")
    # OpenAI gpt-4o-mini — name-hint vision capable.
    p_vision = StubProvider("openai", "gpt-4o-mini")
    router = Router([p_text, p_vision], tr)

    result = router.dispatch(
        SommRequest(prompt="hi", capabilities_required=["vision"])
    )
    assert result.provider == "openai"


def test_router_uses_full_chain_when_no_caps_required(tmp_path):
    """Baseline: unchanged routing semantics for non-multimodal calls."""
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p = StubProvider("ollama", "text-only")
    router = Router([p], tr)

    result = router.dispatch(SommRequest(prompt="hi"))
    assert result.provider == "ollama"

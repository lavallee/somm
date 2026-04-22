"""HuggingFace-Hub intel worker — supplements `model_intel` with per-model
`pipeline_tag` and derived input/output modalities.

Why this exists: OpenRouter's `/api/v1/models` mostly exposes an input
modality string (`"text+image->text"`) but its `output_modalities`
coverage is spotty, so downstream filters "vision model that outputs
text" can't reliably exclude audio-gen models with image inputs
(`google/lyria-3-pro-preview`). HuggingFace publishes a canonical
`pipeline_tag` per model that maps unambiguously to modality pairs —
`text-generation`, `image-text-to-text`, `text-to-image`,
`text-to-speech`, etc. — and we layer that over the primary intel.

Only *enriches* existing rows; never creates new ones. If a model is
on HF but not on OpenRouter/Ollama/static we have no pricing or
context for it anyway, so there's nothing useful to add.

Feature-flagged by default via `enabled=False` in the constructor or
`SOMM_ENABLE_HF_INTEL=1` in the env. Off by default until coverage
stabilises; failures never block the rest of a refresh cycle.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import httpx
from somm_core.pricing import merge_intel_capabilities

if TYPE_CHECKING:
    from somm_core.repository import Repository


_log = logging.getLogger("somm.workers.hf_intel")


# Canonical pipeline_tag → modality map. Add entries as new HF task types
# appear; unknown tags fall through unmapped (we still store the raw tag).
# Input/output keys mirror OpenRouter's `architecture` shape.
HF_PIPELINE_MAP: dict[str, dict[str, list[str]]] = {
    "text-generation":               {"input": ["text"],          "output": ["text"]},
    "text2text-generation":          {"input": ["text"],          "output": ["text"]},
    "conversational":                {"input": ["text"],          "output": ["text"]},
    "image-text-to-text":            {"input": ["image", "text"], "output": ["text"]},
    "image-to-text":                 {"input": ["image"],         "output": ["text"]},
    "visual-question-answering":     {"input": ["image", "text"], "output": ["text"]},
    "document-question-answering":   {"input": ["image", "text"], "output": ["text"]},
    "automatic-speech-recognition":  {"input": ["audio"],         "output": ["text"]},
    "text-to-image":                 {"input": ["text"],          "output": ["image"]},
    "image-to-image":                {"input": ["image"],         "output": ["image"]},
    "text-to-video":                 {"input": ["text"],          "output": ["video"]},
    "text-to-speech":                {"input": ["text"],          "output": ["audio"]},
    "text-to-audio":                 {"input": ["text"],          "output": ["audio"]},
    "audio-to-audio":                {"input": ["audio"],         "output": ["audio"]},
    "audio-text-to-text":            {"input": ["audio", "text"], "output": ["text"]},
    "image-classification":          {"input": ["image"],         "output": ["text"]},
    "object-detection":              {"input": ["image"],         "output": ["text"]},
    "feature-extraction":            {"input": ["text"],          "output": ["embedding"]},
    "sentence-similarity":           {"input": ["text"],          "output": ["embedding"]},
}


# Model-id suffixes OpenRouter uses that aren't part of the HF id.
_OR_TIER_SUFFIXES = (":free", ":nitro", ":beta", ":extended", ":online")


def canonical_hf_id(provider: str, model: str) -> str | None:
    """Best-effort mapping from a somm (provider, model) pair to an HF id.

    Conservative — returns None rather than guessing for providers where
    HF ids aren't a meaningful canonical (Anthropic, OpenAI, Minimax).
    Ollama mapping is deferred: family→HF id is n-to-many and needs a
    curated alias table we don't have yet.
    """
    if provider != "openrouter":
        return None
    base = model
    for suf in _OR_TIER_SUFFIXES:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    # Some OpenRouter ids are themselves non-HF (e.g. `openrouter/auto`,
    # `anthropic/claude-*`). Filter out the meta-routers outright.
    org, _, _name = base.partition("/")
    if not org or not _name:
        return None
    if org in {"openrouter"}:
        return None
    # Anthropic / OpenAI / Google Gemini proprietary models aren't on HF —
    # skipping saves a round-trip that will 404 anyway.
    if org in {"anthropic", "openai"}:
        return None
    return base


class HuggingFaceIntelWorker:
    """Fetches `pipeline_tag` + `tags` for known models and merges them
    into `model_intel.capabilities_json` under the `hf` sub-key.

    Runs *after* the primary intel refresh so it enriches fresh rows.
    Skips models with no plausible HF id (proprietary, meta-routers).
    """

    name = "hf_intel"

    def __init__(
        self,
        repo: Repository,
        *,
        enabled: bool | None = None,
        base_url: str = "https://huggingface.co/api/models",
        timeout: float = 5.0,
        max_models: int | None = None,
    ) -> None:
        self.repo = repo
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_models = max_models
        if enabled is None:
            enabled = os.environ.get("SOMM_ENABLE_HF_INTEL", "").strip() in {"1", "true", "yes"}
        self.enabled = enabled

    def run_once(self) -> dict:
        """Enrich every eligible model_intel row with HF metadata.

        Returns a summary: models enriched, skipped (no HF id), errored.
        Errors per-model are swallowed — HF rate limits and 404s should
        not poison the cache.
        """
        if not self.enabled:
            return {"enriched": 0, "skipped": 0, "errors": 0, "disabled": True}

        with self.repo._open() as conn:
            rows = conn.execute(
                "SELECT provider, model, capabilities_json FROM model_intel"
            ).fetchall()

        enriched = skipped = errored = 0
        seen_hf_ids: set[str] = set()   # dedup multi-tier variants
        with httpx.Client(timeout=self.timeout) as client:
            for provider, model, _caps_raw in rows:
                if self.max_models is not None and enriched >= self.max_models:
                    break
                hf_id = canonical_hf_id(provider, model)
                if not hf_id:
                    skipped += 1
                    continue
                # `:free` and `:nitro` point at the same HF model — fetch once.
                if hf_id in seen_hf_ids:
                    # Still merge the cached entry to the alias row.
                    pass
                try:
                    meta = self._fetch(client, hf_id)
                except Exception as e:  # noqa: BLE001
                    _log.debug("hf fetch failed for %s: %s", hf_id, e)
                    errored += 1
                    continue
                if meta is None:
                    skipped += 1
                    continue
                seen_hf_ids.add(hf_id)
                delta = _shape_capabilities(hf_id, meta)
                merge_intel_capabilities(self.repo, provider, model, delta)
                enriched += 1
        return {
            "enriched": enriched,
            "skipped": skipped,
            "errors": errored,
            "disabled": False,
        }

    def _fetch(self, client: httpx.Client, hf_id: str) -> dict | None:
        """Return the HF model record, or None on 404/invalid response."""
        url = f"{self.base_url}/{hf_id}"
        r = client.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else None


def _shape_capabilities(hf_id: str, meta: dict) -> dict:
    """Build the `{"hf": {...}}` delta for merge_intel_capabilities."""
    pipeline = meta.get("pipeline_tag")
    tags = meta.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    # Store a bounded slice of tags; HF sometimes returns 50+. We keep the
    # first 16 so the capabilities_json doesn't balloon.
    tags = [t for t in tags if isinstance(t, str)][:16]

    hf_block: dict = {"id": hf_id, "pipeline_tag": pipeline, "tags": tags}
    mapped = HF_PIPELINE_MAP.get(pipeline or "") if isinstance(pipeline, str) else None
    if mapped:
        hf_block["input_modalities"] = mapped["input"]
        hf_block["output_modalities"] = mapped["output"]
    return {"hf": hf_block}

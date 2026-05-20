"""Capability lookup against `model_intel.capabilities_json`.

The router consults this to skip (provider, model) pairs that can't serve a
request's required capabilities *before* making the network call. Unknown
models fall through as capable (same behavior as pre-capability somm).

Capability sources per provider:

- **OpenRouter**: populates `modality` (e.g. `"text+image->text"`) and
  `architecture` (incl. `input_modalities`) on every model. We derive
  `vision` from either field.
- **Anthropic / OpenAI**: no list-models API for pricing, so capabilities
  come from the static pricing seed. Vision is inferred from model name
  as a conservative starter — opus/sonnet/haiku 4.x and gpt-4o* support
  images natively.
- **Ollama**: family-based inference; `llava`, `bakllava`, `llama3.2-vision`
  and similar carry vision. Unknown models fall through as capable.
- **Minimax**: single default model; treated as capability-unknown → allow.

Adding new capability tokens (`tool_use`, `json_mode`, `thinking`, …) is
just a matter of teaching `model_has_capability` to look them up — no
schema change needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository


_VISION_NAME_HINTS: tuple[str, ...] = (
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "llava",
    "bakllava",
    "vision",
    "gemini",
)


# Models that consume "thinking" / reasoning tokens before emitting visible
# output. Workloads that declare ``capabilities_required=["thinking"]`` get
# routed to one of these; mechanical workloads (copyedit, simple polish)
# explicitly omit it and get steered to a non-thinking variant.
#
# Inverse: NON-thinking variants of these families exist (deepseek-v4-flash,
# claude-sonnet-4-6 without extended thinking, gemini-2.5-flash). The
# `_NON_THINKING_NAME_HINTS` list below excludes models that look like they
# want thinking but explicitly don't.
_THINKING_NAME_HINTS: tuple[str, ...] = (
    "deepseek-v4-pro",
    "deepseek-reasoner",
    "claude-opus-4-7",  # Opus families default to extended thinking
    "claude-opus-4-6",
    "o1-",       # OpenAI o1 family
    "o3-",       # OpenAI o3 family
    "gemini-2.5-pro",
    "gemini-3-pro",
    "qwq-",
    "magistral",
)

_NON_THINKING_NAME_HINTS: tuple[str, ...] = (
    "-flash",
    "-mini",
    "-haiku",  # Haiku family is non-thinking by default
    "deepseek-chat",
    "deepseek-coder",
    "gpt-4o",
    "gpt-4.1",
)


# Model families that support function/tool calling. Workloads that declare
# ``capabilities_required=["tools"]`` (e.g. deepagents orchestrators) route
# only to these. Tool calling is near-universal across modern frontier and
# mid-tier models, so the list is generous and unknown models fall through
# as capable (None) rather than being blocked — there is no negative case.
# Adapters that *can't* serve tools raise SommBadRequest at call time
# instead of being filtered here.
_TOOLS_NAME_HINTS: tuple[str, ...] = (
    "claude-3",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4",
    "gpt-4",       # gpt-4o, gpt-4.1, gpt-4-turbo …
    "gpt-5",
    "o1-",
    "o3-",
    "o4-",
    "gemini-1.5",
    "gemini-2",
    "gemini-3",
    "llama-3.1",
    "llama-3.2",
    "llama-3.3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "llama-4",
    "qwen2.5",
    "qwen3",
    "deepseek-chat",
    "deepseek-v3",
    "deepseek-v4",
    "mistral",
    "mixtral",
)


def _openrouter_has_vision(caps: dict) -> bool:
    modality = caps.get("modality") or ""
    if isinstance(modality, str) and "image" in modality.lower():
        return True
    arch = caps.get("architecture") or {}
    if isinstance(arch, dict):
        inputs = arch.get("input_modalities") or []
        if isinstance(inputs, list) and any(
            isinstance(m, str) and "image" in m.lower() for m in inputs
        ):
            return True
    return False


def model_has_capability(
    repo: Repository,
    provider: str,
    model: str,
    capability: str,
) -> bool | None:
    """Return True/False if we know, None if capability data is unavailable.

    Callers treat `None` as "allow — let the provider try" per the
    capability-aware routing proposal.
    """
    if not capability:
        return True

    with repo._open() as conn:
        row = conn.execute(
            "SELECT capabilities_json FROM model_intel "
            "WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()

    caps: dict | None = None
    if row and row[0]:
        try:
            parsed = json.loads(row[0])
            if isinstance(parsed, dict):
                caps = parsed
        except json.JSONDecodeError:
            caps = None

    # Explicit per-capability flag wins if present ({"vision": true}).
    if caps is not None and capability in caps and isinstance(caps[capability], bool):
        return caps[capability]

    if capability == "vision":
        if provider == "openrouter" and caps is not None:
            return _openrouter_has_vision(caps)
        lowered = model.lower()
        if any(h in lowered for h in _VISION_NAME_HINTS):
            return True
        if provider == "ollama":
            fam = (caps or {}).get("family") if caps is not None else None
            if isinstance(fam, str) and any(h in fam.lower() for h in _VISION_NAME_HINTS):
                return True
            # Unknown ollama model → capability-unknown, let it try.
            return None
        if caps is None:
            return None
        return False

    if capability == "thinking":
        # Thinking-tier models reason before emitting visible text. Routing
        # workloads that declare needs_thinking="yes" to one of these is the
        # difference between a calibrated answer and an empty response (the
        # 8K-budget all-eaten-by-reasoning failure mode from 2026-05-06).
        lowered = model.lower()
        if any(h in lowered for h in _NON_THINKING_NAME_HINTS):
            return False
        if any(h in lowered for h in _THINKING_NAME_HINTS):
            return True
        return None  # unknown — let the provider try

    if capability == "tools":
        # Function/tool calling. Known tool-capable families return True;
        # everything else falls through as None (allow) — never False, since
        # a provider that genuinely can't serve tools raises SommBadRequest
        # at call time rather than being pre-filtered here.
        lowered = model.lower()
        if any(h in lowered for h in _TOOLS_NAME_HINTS):
            return True
        return None

    if capability == "non-thinking":
        # Inverse: workloads that DON'T need thinking explicitly steer away
        # from reasoning models, which would burn budget on a mechanical task.
        # E.g. a copyeditor pass on already-clean prose doesn't need v4-pro.
        lowered = model.lower()
        if any(h in lowered for h in _NON_THINKING_NAME_HINTS):
            return True
        if any(h in lowered for h in _THINKING_NAME_HINTS):
            return False
        return None

    # Unknown capability — don't block.
    return None


def provider_can_serve(
    repo: Repository,
    provider: str,
    model: str,
    required: list[str],
) -> tuple[bool, str]:
    """Return (ok, reason). Reason is empty on ok=True."""
    for cap in required:
        verdict = model_has_capability(repo, provider, model, cap)
        if verdict is False:
            return False, f"missing_capability:{cap}"
    return True, ""


def model_output_modalities(
    repo: Repository,
    provider: str,
    model: str,
) -> list[str] | None:
    """Return the set of output modalities this model can produce, or None
    when we have no signal.

    Signal sources, in order of preference:
      1. OpenRouter `architecture.output_modalities` — list of strings.
      2. OpenRouter `modality` scalar (`"text+image->text"`) — parse the
         right-hand side.
      3. HuggingFace `hf.output_modalities` — set by the HF intel worker
         from `pipeline_tag`.

    Returns lowercased modality tokens (`"text"`, `"image"`, `"audio"`,
    `"video"`, `"embedding"`). Callers that want to filter for "outputs
    text" should check membership against a requested set.
    """
    import json

    with repo._open() as conn:
        row = conn.execute(
            "SELECT capabilities_json FROM model_intel "
            "WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()

    if not row or not row[0]:
        return None
    try:
        caps = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(caps, dict):
        return None

    # 1. Direct OpenRouter architecture.output_modalities
    arch = caps.get("architecture") or {}
    if isinstance(arch, dict):
        out = arch.get("output_modalities")
        if isinstance(out, list) and out:
            normalised = [m.lower() for m in out if isinstance(m, str)]
            if normalised:
                return normalised

    # 2. OpenRouter scalar modality "in+out->out"
    modality = caps.get("modality")
    if isinstance(modality, str) and "->" in modality:
        _, _, after = modality.partition("->")
        parts = [p.strip().lower() for p in after.split("+") if p.strip()]
        if parts:
            return parts

    # 3. HuggingFace enrichment
    hf = caps.get("hf") or {}
    if isinstance(hf, dict):
        out = hf.get("output_modalities")
        if isinstance(out, list) and out:
            normalised = [m.lower() for m in out if isinstance(m, str)]
            if normalised:
                return normalised

    return None

"""Canonical model identities.

The same weights show up under several identifiers: ``claude-opus-5[1m]`` and
``claude-opus-5`` on a Claude seat, ``google/gemma-3-27b-it:free`` and
``google/gemma-3-27b-it`` on OpenRouter, ``claude-cli`` and ``claude_cli`` as
the provider. Rollups and the sommelier treated each spelling as a separate
model, which both inflated candidate lists and hid "you already use this".

``model_aliases`` has existed since schema 19 and was never populated in
production. This module derives the aliases that follow from rules — the
ones a person would never bother to type — and leaves anything genuinely
ambiguous alone (``sonnet`` is a moving nickname, not a model).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository

# Suffixes that change serving tier or routing but not the weights.
_OPENROUTER_TIERS = re.compile(r":(free|nitro|beta|extended|online|floor|thinking)$")
_CONTEXT_SUFFIX = re.compile(r"\[\d+[mk]\]$")  # claude-opus-5[1m]
_PROVIDER_SPELLING = {
    "claude_cli": "claude-cli",
    "codex_cli": "codex-cli",
}

# Nicknames that resolve to a different concrete model over time. Aliasing
# them to a concrete id would be a guess, so they stay distinct and the
# doctor lists them as unresolved.
_MOVING_NICKNAMES = {"sonnet", "opus", "haiku", "codex-cli", "default"}


def canonical_provider(provider: str) -> str:
    return _PROVIDER_SPELLING.get(provider, provider)


def canonicalize(provider: str, model: str) -> str | None:
    """The canonical id for a (provider, model) route, or ``None`` when the
    route is already canonical or cannot be resolved by rule."""
    provider = canonical_provider(provider)
    name = model.strip()
    if not name or name.startswith("*") or name in _MOVING_NICKNAMES:
        return None
    base = _CONTEXT_SUFFIX.sub("", name)
    if provider == "openrouter":
        base = _OPENROUTER_TIERS.sub("", base)
    if provider == "claude-cli" or (provider == "openrouter" and base.startswith("anthropic/")):
        # A Claude seat and OpenRouter's Anthropic wrapper serve Anthropic's
        # own models; the canonical identity is the first-party route.
        family_model = base.removeprefix("anthropic/")
        canonical = f"anthropic/{family_model}"
    elif provider == "openrouter" and base.startswith("openai/"):
        canonical = base
    else:
        canonical = f"{provider}/{base}"
    if canonical == f"{provider}/{name}":
        return None
    return canonical


def seed_model_aliases(repo: Repository, *, since_days: int = 90) -> int:
    """Insert rule-derived aliases for routes seen in the ledger or in
    ``model_intel``. Rule rows never overwrite a manual alias. Returns the
    number of rows written."""
    import json

    existing = repo.model_alias_map()
    written = 0
    catalog: dict[tuple[str, str], str] = {}
    with repo._open() as conn:
        pairs = set(
            conn.execute(
                "SELECT DISTINCT provider, model FROM calls WHERE ts >= datetime('now', ?)",
                (f"-{int(since_days)} days",),
            ).fetchall()
        )
        intel = conn.execute(
            "SELECT provider, model, capabilities_json FROM model_intel"
        ).fetchall()
    for provider, model, caps_json in intel:
        pairs.add((provider, model))
        try:
            caps = json.loads(caps_json) if caps_json else {}
        except ValueError:
            caps = {}
        canonical = (caps.get("models_dev") or {}).get("canonical_model_id") if isinstance(caps, dict) else None
        if isinstance(canonical, str) and canonical and canonical != f"{provider}/{model}":
            catalog[(provider, model)] = canonical
    for provider, model in sorted(pairs):
        if not provider or not model or (provider, model) in existing:
            continue
        # A catalog's own canonical id beats a spelling rule.
        canonical = catalog.get((provider, model)) or canonicalize(provider, model)
        if canonical is None:
            continue
        source = "models_dev" if (provider, model) in catalog else "rule"
        repo.set_model_alias(canonical, provider, model, source=source)
        written += 1
    return written


__all__ = ["canonical_provider", "canonicalize", "seed_model_aliases"]

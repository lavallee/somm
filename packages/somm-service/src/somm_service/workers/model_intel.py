"""Model-intel worker — scrapes OpenRouter, probes ollama, seeds static pricing.

Sources:
  - OpenRouter /api/v1/models — authoritative pricing + context windows.
    Public endpoint (no API key required for listing).
  - ollama /api/tags — locally-installed models; price=0.
  - Static pricing table — anthropic/openai/minimax (no reliable public
    pricing API). Hand-curated starter; refresh on each somm release.

Writes to `model_intel` via `somm_core.pricing.write_intel`. Failures on
any source don't poison the cache — last-good entries remain valid.

Run manually: `somm admin refresh-intel`
Or automatically via the service's scheduler loop (D3d).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from somm_core.pricing import write_intel

if TYPE_CHECKING:
    from somm_core.repository import Repository


_log = logging.getLogger("somm.workers.model_intel")


# Static pricing (per 1M tokens, USD) for commercial providers with no
# public pricing API. Keep this short and refresh on each somm release.
# Source: provider pricing pages as of Jan 2026.
STATIC_PRICING: dict[tuple[str, str], dict] = {
    ("anthropic", "claude-haiku-4-5-20251001"): {
        "price_in_per_1m": 1.0,
        "price_out_per_1m": 5.0,
        "context_window": 200_000,
    },
    ("anthropic", "claude-sonnet-4-6"): {
        "price_in_per_1m": 3.0,
        "price_out_per_1m": 15.0,
        "context_window": 200_000,
    },
    ("anthropic", "claude-opus-4-7"): {
        "price_in_per_1m": 15.0,
        "price_out_per_1m": 75.0,
        "context_window": 200_000,
    },
    ("openai", "gpt-4o-mini"): {
        "price_in_per_1m": 0.15,
        "price_out_per_1m": 0.6,
        "context_window": 128_000,
    },
    ("openai", "gpt-4o"): {
        "price_in_per_1m": 2.5,
        "price_out_per_1m": 10.0,
        "context_window": 128_000,
    },
    ("minimax", "MiniMax-M2"): {
        "price_in_per_1m": 0.0,
        "price_out_per_1m": 0.0,
        "context_window": 200_000,
    },
}


class ModelIntelWorker:
    """Refreshes the `model_intel` table from OpenRouter + ollama + static."""

    name = "model_intel"

    def __init__(
        self,
        repo: Repository,
        openrouter_url: str = "https://openrouter.ai/api/v1/models",
        ollama_url: str = "http://localhost:11434",
        timeout: float = 10.0,
    ) -> None:
        self.repo = repo
        self.openrouter_url = openrouter_url
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout

    def run_once(self) -> dict:
        """Refresh intel from all sources. Returns a summary dict."""
        summary = {
            "openrouter": 0,
            "ollama": 0,
            "static": 0,
            "errors": [],
        }
        for fn, key in (
            (self._refresh_static, "static"),
            (self._refresh_openrouter, "openrouter"),
            (self._refresh_ollama, "ollama"),
        ):
            try:
                summary[key] = fn()
            except Exception as e:
                _log.warning("model_intel %s failed: %s", key, e)
                summary["errors"].append(f"{key}: {e}")
        return summary

    # ------------------------------------------------------------------

    def _refresh_static(self) -> int:
        n = 0
        for (provider, model), data in STATIC_PRICING.items():
            write_intel(
                self.repo,
                provider=provider,
                model=model,
                price_in_per_1m=data["price_in_per_1m"],
                price_out_per_1m=data["price_out_per_1m"],
                context_window=data.get("context_window"),
                capabilities=data.get("capabilities"),
                source="static",
            )
            n += 1
        return n

    def _refresh_openrouter(self) -> int:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(self.openrouter_url)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            raise RuntimeError(f"openrouter fetch failed: {e}") from e

        n = 0
        for m in data.get("data", []):
            model_id = m.get("id")
            if not model_id:
                continue
            pricing = m.get("pricing") or {}
            # OpenRouter gives prices as string dollars-per-token; convert to
            # per-1M. Unknown-pricing sentinels ("-1" on meta-models) parse to
            # None and propagate through as NULL so downstream comparisons
            # don't treat them as cheap.
            _in = _parse_price_per_token(pricing.get("prompt"))
            _out = _parse_price_per_token(pricing.get("completion"))
            price_in = _in * 1_000_000 if _in is not None else None
            price_out = _out * 1_000_000 if _out is not None else None
            ctx = m.get("context_length")
            caps = {
                "top_provider": m.get("top_provider"),
                "architecture": m.get("architecture"),
                "modality": m.get("modality"),
            }
            write_intel(
                self.repo,
                provider="openrouter",
                model=model_id,
                price_in_per_1m=price_in,
                price_out_per_1m=price_out,
                context_window=ctx,
                capabilities=caps,
                source="openrouter",
            )
            n += 1
        return n

    def _refresh_ollama(self) -> int:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(f"{self.ollama_url}/api/tags")
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            raise RuntimeError(f"ollama probe failed: {e}") from e

        n = 0
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if not name:
                continue
            details = m.get("details") or {}
            ctx = None
            # Ollama's details occasionally expose context via modelfile;
            # leave as None for now — library still works without it.
            caps = {
                "size": m.get("size"),
                "family": details.get("family"),
                "format": details.get("format"),
                "parameter_size": details.get("parameter_size"),
            }
            write_intel(
                self.repo,
                provider="ollama",
                model=name,
                price_in_per_1m=0.0,
                price_out_per_1m=0.0,
                context_window=ctx,
                capabilities=caps,
                source="ollama-local",
            )
            n += 1
        return n


def _parse_price_per_token(raw) -> float | None:
    """OpenRouter gives prices as strings like '0.0000006'. Returns USD per
    token, or None when the price is unknown / sentinel.

    OpenRouter uses "-1" as a sentinel for variable or dynamic pricing (e.g.,
    `openrouter/auto`, `openrouter/bodybuilder` — router meta-models whose
    effective price depends on which backend is chosen at inference time).
    Treat those as unknown rather than literal negative prices — otherwise
    downstream "cheaper than X" logic will pick them as ultra-cheap.
    """
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v

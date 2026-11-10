"""Cost provenance: what a ``cost_usd`` figure is, and where it came from.

Every ``calls`` row carries five provenance columns beside ``cost_usd`` so
downstream accounting can keep reported, computed, marginal, and notional
values distinct. Three paths in the library (``generate``, ``embed``,
``stream``), the shadow-eval worker, and the harness ledger all stamp them;
this module is the one implementation.

Hot-path note: ``plans.toml`` and ``model_intel`` are consulted for every
computed cost. Both are cached here — plans by file mtime, pricing versions
for a short TTL — because the previous version re-read the TOML and ran a
query per call.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from somm_core.plans import load_plans, plan_for, plans_path
from somm_core.pricing import free_route, known_prices

if TYPE_CHECKING:
    from somm_core.repository import Repository

_PLANS_LOCK = threading.Lock()
_plans_cache: tuple[Path, float | None, dict] | None = None

# cost_source of a call whose price somm did not know. Its cost_usd is 0.0
# (the column is NOT NULL), so this source is the one way to tell an unknown
# price from a free one.
UNKNOWN_COST_SOURCE = "unknown:no-price"


def cost_is_known(cost_source: str | None) -> bool:
    """False for a row whose 0.0 means "price unknown", not "free"."""
    return not (cost_source or "").startswith("unknown:")


def known_cost_sql(alias: str = "") -> str:
    """SQL predicate twin of ``cost_is_known``; ``alias`` like ``"c."``."""
    return f"COALESCE({alias}cost_source, '') NOT LIKE 'unknown:%'"

_PRICING_VERSION_TTL_S = 300.0
_pricing_lock = threading.Lock()
_pricing_cache: dict[tuple[str, str, str], tuple[float, str | None]] = {}


def cached_plans() -> dict:
    """``load_plans()`` re-read only when ``plans.toml`` changes on disk."""
    global _plans_cache
    path = plans_path()
    try:
        mtime: float | None = os.stat(path).st_mtime
    except OSError:
        mtime = None
    with _PLANS_LOCK:
        if _plans_cache is not None and _plans_cache[0] == path and _plans_cache[1] == mtime:
            return _plans_cache[2]
    plans = load_plans(path) if mtime is not None else {}
    with _PLANS_LOCK:
        _plans_cache = (path, mtime, plans)
    return plans


def computed_cost_kind(provider: str) -> str:
    """Classify a token-times-price value without confusing it for an invoice.

    A metered plan (a coding seat, a token plan) has ~0 marginal dollars inside
    its quota, so its computed cost is *notional* list price; a PAYG key's is
    *marginal* spend.
    """
    try:
        mode = plan_for(provider, cached_plans()).mode
    except Exception:
        mode = "payg"
    return "notional" if mode == "metered" else "marginal"


def pricing_version(repo: Repository, provider: str, model: str) -> str | None:
    """``source@last_seen`` of the model_intel row a computed cost used."""
    key = (str(repo.db_path), provider, model)
    now = time.monotonic()
    with _pricing_lock:
        hit = _pricing_cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    with repo._open() as conn:
        row = conn.execute(
            "SELECT source, last_seen FROM model_intel WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()
    version = f"{row[0]}@{row[1]}" if row else None
    with _pricing_lock:
        if len(_pricing_cache) > 4096:
            _pricing_cache.clear()
        _pricing_cache[key] = (now + _PRICING_VERSION_TTL_S, version)
    return version


def cost_provenance(
    repo: Repository,
    provider: str,
    model: str,
    *,
    reported: bool,
    source: str | None,
) -> tuple[str, str, str, str | None, str | None]:
    """``(cost_basis, cost_kind, cost_accuracy, cost_source, pricing_version)``.

    A reported figure is what the provider said it charged. An Ollama call is
    ``included``: zero metered price, not zero economic cost. A call with no
    known price (recorded as 0.0) is ``unknown`` throughout, with source
    ``unknown:no-price``. Anything else is computed from ``model_intel`` and
    says which pricing row it used.
    """
    if provider == "chatgpt":
        return "unknown", "included", "unknown", "chatgpt-subscription", None
    if reported:
        return "reported", "unknown", "unknown", source, None
    if provider == "ollama":
        return "unknown", "included", "unknown", "local-included-unpriced", None
    if known_prices(repo, provider, model) is None:
        free = free_route(provider or "", model or "")
        if free is not None:
            return "computed", computed_cost_kind(provider), "estimated", free, None
        return "unknown", "unknown", "unknown", UNKNOWN_COST_SOURCE, None
    return (
        "computed",
        computed_cost_kind(provider),
        "estimated",
        source or "somm:model_intel",
        pricing_version(repo, provider, model),
    )


def reset_caches() -> None:
    """Test hook."""
    global _plans_cache
    with _PLANS_LOCK:
        _plans_cache = None
    with _pricing_lock:
        _pricing_cache.clear()

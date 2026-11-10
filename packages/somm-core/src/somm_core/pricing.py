"""Cost calculation from the `model_intel` cache.

Library calls `cost_for_call(repo, provider, model, tokens_in, tokens_out)`
after each LLM call. If (provider, model) is in `model_intel`, returns the
computed USD cost. If not, returns 0.0 and the call's cost_source says
"unknown:no-price"; `backfill_costs` fills it once a price exists.

Prices in `model_intel` are per 1M tokens: input, output, and the
prompt-cache read and write rates where the source publishes them.
Local Ollama and OpenRouter `:free` routes cost 0.0 (see `free_route`).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository

# Providers known to charge per-token. Missing pricing for these is a bug.
_PAID_PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "openai", "gemini", "deepseek", "minimax", "perplexity"}
)

# Track which (provider, model) pairs have already emitted a missing-pricing
# warning so we only warn once per process.
_warned_missing_pricing: set[tuple[str, str]] = set()

_PRICE_CACHE_TTL_SECONDS = 600.0
_price_cache_lock = threading.Lock()
_price_cache: dict[tuple[str, str, str], tuple[float, tuple | None]] = {}


def _price_cache_key(repo: Repository, provider: str, model: str) -> tuple[str, str, str]:
    return (str(repo.db_path), provider, model)


def _clear_price_cache(db_path: Path | str | None = None) -> None:
    """Clear cached model prices, optionally scoped to one database path."""
    with _price_cache_lock:
        if db_path is None:
            _price_cache.clear()
            return
        db_key = str(Path(db_path))
        for key in [key for key in _price_cache if key[0] == db_key]:
            _price_cache.pop(key, None)


# The bundled snapshot (data/pricing_bundle.json) is somm's one price table.
# Earlier releases also carried two hand-written tables (a seed here and a
# "static" one in the model-intel worker) that drifted from the bundle and
# from each other. Rows those tables wrote into existing databases are owned
# by somm, so the bundle now overwrites them.
_SOMM_OWNED_SOURCES = ("somm_seed", "static", "litellm_bundle")


def seed_known_pricing(repo: Repository) -> None:
    """Deprecated alias of :func:`sync_bundled_pricing`, kept for callers."""
    sync_bundled_pricing(repo)


def sync_bundled_pricing(repo: Repository) -> int:
    """Sync the wheel's bundled pricing snapshot into `model_intel`.

    The bundle (data/pricing_bundle.json, generated from LiteLLM's price
    file by scripts/update_pricing_bundle.py) covers every provider somm
    routes to, so cost tracking works offline out of the box — including
    on databases that predate the bundle.

    Rules:
    - Rows missing from `model_intel` are inserted (source="litellm_bundle").
    - Rows whose source is "somm_seed", "static" or "litellm_bundle" are
      updated: those came from somm's own tables, and the bundle replaces them.
    - Rows from any other source (manual edits, live refreshes, workers)
      are never touched.
    - A crc32 of the bundle is stored in SQLite's `PRAGMA user_version`;
      when it matches, the sync is a single-pragma no-op, so calling this
      on every library init is cheap.

    Returns the number of rows written. Never raises — pricing sync must
    not break `somm.llm()`.
    """
    import json
    import zlib

    _clear_price_cache(repo.db_path)
    try:
        from importlib import resources

        data = (resources.files("somm_core") / "data" / "pricing_bundle.json").read_bytes()
    except Exception:
        return 0

    # The sync rules are part of the fingerprint, so a database synced under
    # older rules re-syncs once even when the bundle bytes are unchanged.
    rules = ",".join(_SOMM_OWNED_SOURCES).encode()
    fingerprint = zlib.crc32(rules + data) & 0x7FFFFFFF  # keep it positive for SQLite
    try:
        with repo._open() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] == fingerprint:
                return 0
            existing = {
                (r[0], r[1]): r[2]
                for r in conn.execute("SELECT provider, model, source FROM model_intel")
            }

        bundle = json.loads(data)
        written = 0
        for m in bundle.get("models", []):
            key = (m["provider"], m["model"])
            source = existing.get(key, "")
            if key in existing and source not in _SOMM_OWNED_SOURCES:
                continue
            write_intel(
                repo,
                provider=m["provider"],
                model=m["model"],
                price_in_per_1m=m.get("price_in_per_1m"),
                price_out_per_1m=m.get("price_out_per_1m"),
                context_window=m.get("context_window"),
                capabilities=m.get("capabilities"),
                source="litellm_bundle",
                cache_read_per_1m=m.get("cache_read_per_1m"),
                cache_write_per_1m=m.get("cache_write_per_1m"),
            )
            written += 1

        with repo._open() as conn:
            conn.execute(f"PRAGMA user_version = {fingerprint}")
        return written
    except Exception:
        return 0


def _cost_sql(c: str) -> str:
    """``price_tokens`` as SQL over a calls alias joined to ``model_intel mi``."""
    read = f"MAX(COALESCE({c}.cache_tokens_in, 0), 0)"
    write = f"MAX(COALESCE({c}.cache_tokens_out, 0), 0)"
    return (
        f"(MAX({c}.tokens_in - {read} - {write}, 0) * COALESCE(mi.price_in_per_1m, 0)"
        f" + {read} * COALESCE(mi.price_cache_read_per_1m, mi.price_in_per_1m, 0)"
        f" + {write} * COALESCE(mi.price_cache_write_per_1m, mi.price_in_per_1m, 0)"
        f" + {c}.tokens_out * COALESCE(mi.price_out_per_1m, 0)) / 1000000.0"
    )


def backfill_costs(
    repo: Repository,
    since_days: int | None = None,
    dry_run: bool = False,
) -> tuple[int, float]:
    """Recompute cost_usd for calls logged at $0 that now have pricing.

    Historical calls made before pricing intel existed for their
    (provider, model) carry cost_usd=0 forever unless recomputed. This
    joins those rows against current `model_intel` and rewrites the cost
    from the recorded token counts. Only rows whose cost is NULL/0 and
    whose model now has a nonzero price are touched — priced rows and
    genuinely-free models are left alone.

    Returns (rows_affected, total_usd). With dry_run=True nothing is
    written; the return value reports what would change.
    """
    _clear_price_cache(repo.db_path)
    window = ""
    params: list = []
    if since_days is not None:
        window = " AND c.ts >= datetime('now', ?)"
        params.append(f"-{int(since_days)} days")

    select_sql = f"""
        SELECT COUNT(*),
               COALESCE(SUM({_cost_sql("c")}), 0)
        FROM calls c
        JOIN model_intel mi ON mi.provider = c.provider AND mi.model = c.model
        WHERE (c.cost_usd IS NULL OR c.cost_usd = 0)
          AND c.origin = 'native'
          AND c.cost_basis != 'reported'
          AND (mi.price_in_per_1m > 0 OR mi.price_out_per_1m > 0)
          AND (c.tokens_in > 0 OR c.tokens_out > 0){window}
    """
    with repo._open() as conn:
        n, total = conn.execute(select_sql, params).fetchone()
        if dry_run or not n:
            return (n, round(total or 0.0, 6))
        conn.execute(
            f"""
            UPDATE calls SET
                cost_usd = ROUND({_cost_sql("calls")}, 8),
                cost_basis = 'computed',
                cost_accuracy = 'estimated',
                cost_source = 'somm:model_intel',
                pricing_version = mi.source || '@' || mi.last_seen
            FROM model_intel mi
            WHERE mi.provider = calls.provider AND mi.model = calls.model
              AND (calls.cost_usd IS NULL OR calls.cost_usd = 0)
              AND calls.origin = 'native'
              AND calls.cost_basis != 'reported'
              AND (mi.price_in_per_1m > 0 OR mi.price_out_per_1m > 0)
              AND (calls.tokens_in > 0 OR calls.tokens_out > 0)
              {window.replace("c.ts", "calls.ts") if window else ""}
            """,
            params,
        )
    return (n, round(total or 0.0, 6))


# (input, output, cache read, cache write) per 1M tokens; None = not published.
Prices = tuple[float | None, float | None, float | None, float | None]


def model_prices(repo: Repository, provider: str, model: str) -> Prices | None:
    """The ``model_intel`` prices for (provider, model), or None with no row.

    Cached per database for ten minutes; ``write_intel`` invalidates.
    """
    key = _price_cache_key(repo, provider, model)
    now = time.monotonic()
    with _price_cache_lock:
        cached = _price_cache.get(key)
        if cached and cached[0] > now:
            return cached[1]
    with repo._open() as conn:
        row = conn.execute(
            "SELECT price_in_per_1m, price_out_per_1m, "
            "price_cache_read_per_1m, price_cache_write_per_1m "
            "FROM model_intel WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()
    prices: Prices | None = tuple(row) if row else None  # type: ignore[assignment]
    with _price_cache_lock:
        _price_cache[key] = (now + _PRICE_CACHE_TTL_SECONDS, prices)
    return prices


def price_tokens(
    prices: Prices,
    tokens_in: int,
    tokens_out: int,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> float:
    """USD for one call at ``prices``.

    ``tokens_in`` is the whole prompt, cache reads and writes included, as
    every adapter records it. Cached slices are priced at their own rates;
    a model whose source publishes no cache rate falls back to the full
    input rate, which is what the provider charges when it does no caching
    and never understates spend.
    """
    price_in, price_out, cache_read, cache_write = prices
    price_in = price_in or 0.0
    price_out = price_out or 0.0
    read = max(int(cache_read_tokens or 0), 0)
    write = max(int(cache_write_tokens or 0), 0)
    uncached = max(int(tokens_in) - read - write, 0)
    cost = (
        uncached * price_in
        + read * (price_in if cache_read is None else cache_read)
        + write * (price_in if cache_write is None else cache_write)
        + int(tokens_out) * price_out
    ) / 1_000_000.0
    return round(cost, 8)


def free_route(provider: str, model: str) -> str | None:
    """Why a route costs nothing per token, or None when it is not free.

    Local Ollama has no metered price; an OpenRouter ``:free`` variant is
    listed at zero. These record 0.0. Anything else without a known price
    records 0.0 with cost_source "unknown:no-price".
    """
    if provider == "ollama":
        return "local-included-unpriced"
    if provider == "openrouter" and model.endswith(":free"):
        return "openrouter:free-model"
    return None


def known_prices(repo: Repository, provider: str, model: str) -> Prices | None:
    """``model_prices`` when they price anything, else None (unknown)."""
    if not provider or not model:
        return None
    prices = model_prices(repo, provider, model)
    if prices is None or (prices[0] is None and prices[1] is None):
        return None
    return prices


def cost_for_call(
    repo: Repository,
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    *,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> float:
    """USD cost for one call; 0.0 when somm does not know the price.

    An unknown price (a self-hosted endpoint, an unlisted model, a call with
    no model name) returns 0.0 like a free route, so the number alone cannot
    tell them apart: ``cost_provenance`` stamps it ``cost_source =
    "unknown:no-price"``, and readers use ``somm_core.cost.cost_is_known``.
    A ``model_intel`` row wins (that is how a local box gets a rate). The
    first unknown price for a known-paid provider warns on stderr; refresh
    intel or ``write_intel`` a price, then ``somm backfill-costs``.
    """
    prices = known_prices(repo, provider, model)
    if prices is None:
        if free_route(provider or "", model or "") is not None:
            return 0.0
        if provider in _PAID_PROVIDERS:
            key = (provider, model)
            if key not in _warned_missing_pricing:
                _warned_missing_pricing.add(key)
                print(
                    f"[somm] WARNING: no pricing data for {provider}/{model} "
                    f"— cost recorded as unknown. Run `somm-serve admin refresh-intel` "
                    f"or add pricing via write_intel().",
                    file=sys.stderr,
                )
        return 0.0
    return price_tokens(prices, tokens_in, tokens_out, cache_read_tokens, cache_write_tokens)


def write_intel(
    repo: Repository,
    provider: str,
    model: str,
    price_in_per_1m: float | None,
    price_out_per_1m: float | None,
    context_window: int | None,
    capabilities: dict | list | None,
    source: str,
    *,
    cache_read_per_1m: float | None = None,
    cache_write_per_1m: float | None = None,
) -> None:
    """Upsert a row into `model_intel`.

    The cache rates are replaced with the rest of the row: a source that
    publishes none leaves them NULL, and cost falls back to the input rate.

    ``last_seen`` is refreshed on every write; ``first_seen`` is set on the
    first and kept, so "did this land after the model I use?" has an answer.
    """
    import json

    caps_json = json.dumps(capabilities) if capabilities is not None else None
    with repo._open() as conn:
        conn.execute(
            """
            INSERT INTO model_intel
                (provider, model, price_in_per_1m, price_out_per_1m,
                 price_cache_read_per_1m, price_cache_write_per_1m,
                 context_window, capabilities_json, source, last_seen, first_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(provider, model) DO UPDATE SET
                price_in_per_1m = excluded.price_in_per_1m,
                price_out_per_1m = excluded.price_out_per_1m,
                price_cache_read_per_1m = excluded.price_cache_read_per_1m,
                price_cache_write_per_1m = excluded.price_cache_write_per_1m,
                context_window = excluded.context_window,
                capabilities_json = excluded.capabilities_json,
                source = excluded.source,
                last_seen = CURRENT_TIMESTAMP,
                first_seen = COALESCE(model_intel.first_seen, excluded.first_seen)
            """,
            (
                provider,
                model,
                price_in_per_1m,
                price_out_per_1m,
                cache_read_per_1m,
                cache_write_per_1m,
                context_window,
                caps_json,
                source,
            ),
        )
    _clear_price_cache(repo.db_path)


def merge_intel_capabilities(
    repo: Repository,
    provider: str,
    model: str,
    delta: dict,
) -> bool:
    """Recursively merge `delta` into `model_intel.capabilities_json` for
    an existing row. Returns True on success, False when no row exists.

    Sources layered on top of the primary intel (HuggingFace pipeline_tag,
    LMArena quality, canirun.ai feasibility …) write under their own
    sub-key so they compose without stomping each other. Primary-source
    workers (OpenRouter, Ollama) keep using `write_intel` — the sub-keys
    survive because downstream workers run after, and re-enrich on the
    next cycle.
    """
    import json

    with repo._open() as conn:
        row = conn.execute(
            "SELECT capabilities_json FROM model_intel WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()
        if row is None:
            return False
        existing: dict = {}
        if row[0]:
            try:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    existing = parsed
            except json.JSONDecodeError:
                existing = {}
        merged = _deep_merge(existing, delta)
        conn.execute(
            "UPDATE model_intel SET capabilities_json = ?, "
            "last_seen = CURRENT_TIMESTAMP "
            "WHERE provider = ? AND model = ?",
            (json.dumps(merged), provider, model),
        )
    return True


def _deep_merge(base: dict, delta: dict) -> dict:
    """Merge nested dicts — delta wins on leaf conflicts, dicts recurse."""
    out = dict(base)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def list_intel(repo: Repository, provider: str | None = None) -> list[dict]:
    """Return all known model_intel rows, optionally filtered by provider."""
    import json

    with repo._open() as conn:
        if provider:
            rows = conn.execute(
                "SELECT provider, model, price_in_per_1m, price_out_per_1m, "
                "context_window, capabilities_json, source, last_seen, "
                "price_cache_read_per_1m, price_cache_write_per_1m "
                "FROM model_intel WHERE provider = ? ORDER BY model",
                (provider,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT provider, model, price_in_per_1m, price_out_per_1m, "
                "context_window, capabilities_json, source, last_seen, "
                "price_cache_read_per_1m, price_cache_write_per_1m "
                "FROM model_intel ORDER BY provider, model"
            ).fetchall()
    return [
        {
            "provider": r[0],
            "model": r[1],
            "price_in_per_1m": r[2],
            "price_out_per_1m": r[3],
            "context_window": r[4],
            "capabilities": json.loads(r[5]) if r[5] else None,
            "source": r[6],
            "last_seen": r[7],
            "price_cache_read_per_1m": r[8],
            "price_cache_write_per_1m": r[9],
        }
        for r in rows
    ]

"""Cost calculation from the `model_intel` cache.

Library calls `cost_for_call(repo, provider, model, tokens_in, tokens_out)`
after each LLM call. If (provider, model) is in `model_intel`, returns the
computed USD cost. If not, returns 0.0 — graceful degradation; the agent
or user can refresh intel later.

Prices in `model_intel` are per 1M tokens, stored as input/output pair.
Local / free models have price=0 and cost_usd stays 0.0.
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
_price_cache: dict[tuple[str, str, str], tuple[float, tuple[float, float] | None]] = {}


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


# Hardcoded pricing for major providers, used by seed_known_pricing().
# Format: (provider, model, price_in_per_1m, price_out_per_1m, capabilities)
#
# `capabilities` is a dict merged into model_intel.capabilities_json, or None.
# The named frontier rows below all support tool calling, so they declare
# {"tools": True} — this makes capability-filtered tool workloads
# (capabilities_required=["tools"]) route to them positively rather than
# relying on name-hint inference. Wildcard rows (model="*") carry no caps:
# capability lookup is exact-match, so a wildcard never matches a real
# (provider, model) pair anyway.
#
# Prior-generation Anthropic model IDs are kept alongside current ones so
# projects still pinning older snapshots continue to cost-track correctly.
# Run `somm-serve admin refresh-intel` to pull live pricing.
_TOOLS: dict[str, bool] = {"tools": True}
_KNOWN_PRICING: list[tuple[str, str, float, float, dict | None]] = [
    # Current Anthropic (Claude 4.5–4.7 family)
    ("anthropic", "claude-haiku-4-5-20251001", 0.80, 4.00, _TOOLS),
    ("anthropic", "claude-sonnet-4-6", 3.00, 15.00, _TOOLS),
    ("anthropic", "claude-opus-4-7", 15.00, 75.00, _TOOLS),
    # Prior Anthropic snapshots
    ("anthropic", "claude-sonnet-4-20250514", 3.00, 15.00, _TOOLS),
    ("anthropic", "claude-opus-4-20250514", 15.00, 75.00, _TOOLS),
    ("openai", "gpt-4o-mini", 0.15, 0.60, _TOOLS),
    ("openai", "gpt-4o", 2.50, 10.00, _TOOLS),
    ("ollama", "*", 0.0, 0.0, None),
    ("openrouter", "*:free", 0.0, 0.0, None),
    ("minimax", "*", 0.0, 0.0, None),
]


def seed_known_pricing(repo: Repository) -> None:
    """Populate model_intel with hardcoded pricing if the table is empty.

    Only seeds when the table has zero rows — never overwrites manually
    set prices. Called from SommLLM.__init__ on first use.
    """
    _clear_price_cache(repo.db_path)
    with repo._open() as conn:
        count = conn.execute("SELECT COUNT(*) FROM model_intel").fetchone()[0]
    if count > 0:
        return

    for provider, model, price_in, price_out, capabilities in _KNOWN_PRICING:
        write_intel(
            repo,
            provider=provider,
            model=model,
            price_in_per_1m=price_in,
            price_out_per_1m=price_out,
            context_window=None,
            capabilities=capabilities,
            source="somm_seed",
        )


def sync_bundled_pricing(repo: Repository) -> int:
    """Sync the wheel's bundled pricing snapshot into `model_intel`.

    The bundle (data/pricing_bundle.json, generated from LiteLLM's price
    file by scripts/update_pricing_bundle.py) covers every provider somm
    routes to, so cost tracking works offline out of the box — including
    on databases that predate the bundle.

    Rules:
    - Rows missing from `model_intel` are inserted (source="litellm_bundle").
    - Rows whose source is "somm_seed" or "litellm_bundle" are updated —
      the bundle is fresher than either.
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

        data = (
            resources.files("somm_core") / "data" / "pricing_bundle.json"
        ).read_bytes()
    except Exception:
        return 0

    fingerprint = zlib.crc32(data) & 0x7FFFFFFF  # keep it positive for SQLite
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
            if key in existing and source not in ("somm_seed", "litellm_bundle"):
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
            )
            written += 1

        with repo._open() as conn:
            conn.execute(f"PRAGMA user_version = {fingerprint}")
        return written
    except Exception:
        return 0


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
               COALESCE(SUM((c.tokens_in * mi.price_in_per_1m
                             + c.tokens_out * mi.price_out_per_1m) / 1000000.0), 0)
        FROM calls c
        JOIN model_intel mi ON mi.provider = c.provider AND mi.model = c.model
        WHERE (c.cost_usd IS NULL OR c.cost_usd = 0)
          AND (mi.price_in_per_1m > 0 OR mi.price_out_per_1m > 0)
          AND (c.tokens_in > 0 OR c.tokens_out > 0){window}
    """
    with repo._open() as conn:
        n, total = conn.execute(select_sql, params).fetchone()
        if dry_run or not n:
            return (n, round(total or 0.0, 6))
        conn.execute(
            f"""
            UPDATE calls SET cost_usd = ROUND(
                (calls.tokens_in * mi.price_in_per_1m
                 + calls.tokens_out * mi.price_out_per_1m) / 1000000.0, 8)
            FROM model_intel mi
            WHERE mi.provider = calls.provider AND mi.model = calls.model
              AND (calls.cost_usd IS NULL OR calls.cost_usd = 0)
              AND (mi.price_in_per_1m > 0 OR mi.price_out_per_1m > 0)
              AND (calls.tokens_in > 0 OR calls.tokens_out > 0)
              {window.replace("c.ts", "calls.ts") if window else ""}
            """,
            params,
        )
    return (n, round(total or 0.0, 6))


def cost_for_call(
    repo: Repository,
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """Return USD cost for a single call.

    Returns 0.0 if pricing data is missing for (provider, model) — the
    call is still logged; the agent can backfill cost when intel refreshes.

    If the provider is known-paid (anthropic, openai) and no pricing row
    exists, emits a stderr warning on the first occurrence.
    """
    if not provider or not model:
        return 0.0
    key = _price_cache_key(repo, provider, model)
    now = time.monotonic()
    with _price_cache_lock:
        cached = _price_cache.get(key)
        if cached and cached[0] > now:
            prices = cached[1]
            if prices is None:
                return 0.0
            price_in, price_out = prices
            return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000.0, 8)
    with repo._open() as conn:
        row = conn.execute(
            "SELECT price_in_per_1m, price_out_per_1m "
            "FROM model_intel WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()
    if not row:
        with _price_cache_lock:
            _price_cache[key] = (now + _PRICE_CACHE_TTL_SECONDS, None)
        if provider in _PAID_PROVIDERS:
            key = (provider, model)
            if key not in _warned_missing_pricing:
                _warned_missing_pricing.add(key)
                print(
                    f"[somm] WARNING: no pricing data for {provider}/{model} "
                    f"— cost will be $0. Run `somm-serve admin refresh-intel` "
                    f"or add pricing via write_intel().",
                    file=sys.stderr,
                )
        return 0.0
    price_in, price_out = row[0] or 0.0, row[1] or 0.0
    with _price_cache_lock:
        _price_cache[key] = (now + _PRICE_CACHE_TTL_SECONDS, (price_in, price_out))
    cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000.0
    return round(cost, 8)


def write_intel(
    repo: Repository,
    provider: str,
    model: str,
    price_in_per_1m: float | None,
    price_out_per_1m: float | None,
    context_window: int | None,
    capabilities: dict | list | None,
    source: str,
) -> None:
    """Upsert a row into `model_intel`. Uses CURRENT_TIMESTAMP for last_seen."""
    import json

    caps_json = json.dumps(capabilities) if capabilities is not None else None
    with repo._open() as conn:
        conn.execute(
            """
            INSERT INTO model_intel
                (provider, model, price_in_per_1m, price_out_per_1m,
                 context_window, capabilities_json, source, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider, model) DO UPDATE SET
                price_in_per_1m = excluded.price_in_per_1m,
                price_out_per_1m = excluded.price_out_per_1m,
                context_window = excluded.context_window,
                capabilities_json = excluded.capabilities_json,
                source = excluded.source,
                last_seen = CURRENT_TIMESTAMP
            """,
            (provider, model, price_in_per_1m, price_out_per_1m, context_window, caps_json, source),
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
            "SELECT capabilities_json FROM model_intel "
            "WHERE provider = ? AND model = ?",
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
                "context_window, capabilities_json, source, last_seen "
                "FROM model_intel WHERE provider = ? ORDER BY model",
                (provider,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT provider, model, price_in_per_1m, price_out_per_1m, "
                "context_window, capabilities_json, source, last_seen "
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
        }
        for r in rows
    ]

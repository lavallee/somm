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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository

# Providers known to charge per-token. Missing pricing for these is a bug.
_PAID_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai"})

# Track which (provider, model) pairs have already emitted a missing-pricing
# warning so we only warn once per process.
_warned_missing_pricing: set[tuple[str, str]] = set()

# Hardcoded pricing for major providers, used by seed_known_pricing().
# Format: (provider, model, price_in_per_1m, price_out_per_1m)
#
# Prior-generation Anthropic model IDs are kept alongside current ones so
# projects still pinning older snapshots continue to cost-track correctly.
# Run `somm-serve admin refresh-intel` to pull live pricing.
_KNOWN_PRICING: list[tuple[str, str, float, float]] = [
    # Current Anthropic (Claude 4.5–4.7 family)
    ("anthropic", "claude-haiku-4-5-20251001", 0.80, 4.00),
    ("anthropic", "claude-sonnet-4-6", 3.00, 15.00),
    ("anthropic", "claude-opus-4-7", 15.00, 75.00),
    # Prior Anthropic snapshots
    ("anthropic", "claude-sonnet-4-20250514", 3.00, 15.00),
    ("anthropic", "claude-opus-4-20250514", 15.00, 75.00),
    ("openai", "gpt-4o-mini", 0.15, 0.60),
    ("openai", "gpt-4o", 2.50, 10.00),
    ("ollama", "*", 0.0, 0.0),
    ("openrouter", "*:free", 0.0, 0.0),
    ("minimax", "*", 0.0, 0.0),
]


def seed_known_pricing(repo: Repository) -> None:
    """Populate model_intel with hardcoded pricing if the table is empty.

    Only seeds when the table has zero rows — never overwrites manually
    set prices. Called from SommLLM.__init__ on first use.
    """
    with repo._open() as conn:
        count = conn.execute("SELECT COUNT(*) FROM model_intel").fetchone()[0]
    if count > 0:
        return

    for provider, model, price_in, price_out in _KNOWN_PRICING:
        write_intel(
            repo,
            provider=provider,
            model=model,
            price_in_per_1m=price_in,
            price_out_per_1m=price_out,
            context_window=None,
            capabilities=None,
            source="somm_seed",
        )


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
    with repo._open() as conn:
        row = conn.execute(
            "SELECT price_in_per_1m, price_out_per_1m "
            "FROM model_intel WHERE provider = ? AND model = ?",
            (provider, model),
        ).fetchone()
    if not row:
        if provider in _PAID_PROVIDERS:
            key = (provider, model)
            if key not in _warned_missing_pricing:
                _warned_missing_pricing.add(key)
                print(
                    f"[somm] WARNING: no pricing data for {provider}/{model} "
                    f"— cost will be $0. Run `somm intel refresh` or add "
                    f"pricing via write_intel().",
                    file=sys.stderr,
                )
        return 0.0
    price_in, price_out = row[0] or 0.0, row[1] or 0.0
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

"""Cost calculation from the `model_intel` cache.

Library calls `cost_for_call(repo, provider, model, tokens_in, tokens_out)`
after each LLM call. If (provider, model) is in `model_intel`, returns the
computed USD cost. If not, returns 0.0 — graceful degradation; the agent
or user can refresh intel later.

Prices in `model_intel` are per 1M tokens, stored as input/output pair.
Local / free models have price=0 and cost_usd stays 0.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository


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

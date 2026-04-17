"""Provenance helper — stable dict you can stamp on output rows.

Users typically do:

    row["llm_provenance"] = somm.provenance(result)

and persist that dict alongside whatever the LLM produced. The schema is
stable so future re-questioning (which model? cost? when?) doesn't require
re-joining against calls.sqlite.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core import SommResult


def provenance(result: SommResult) -> dict:
    """Return a stable provenance dict for stamping on output data.

    Shape is schema-versioned via `schema` key so downstream consumers can
    migrate. All values are JSON-serializable.
    """
    return {
        "schema": 1,
        "call_id": result.call_id,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "outcome": result.outcome.value,
        "stamped_at": datetime.now(UTC).isoformat(),
    }

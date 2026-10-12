"""Supported read-only queries over a somm ledger.

Every surface that reports on telemetry — the CLI, the MCP server, the web
service, and external tools such as Milton or a fleet dashboard — needs the
same handful of rollups. Until now each wrote its own SQL against
``calls.sqlite``, and the copies drifted (one labelled an unregistered
workload ``(unregistered)``, another ``ad_hoc``). This module is the one
place those queries live. Functions here never write.

Column-stability contract: the dict keys returned by these functions are
additive. A key, once published, keeps its name and meaning; new keys may
appear. Consumers should read by key, never by position.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

UNREGISTERED = "(unregistered)"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def spend_today(
    db_path: Path,
    project: str,
    default_cap: float | None,
) -> list[dict]:
    """Today's (UTC) budget-eligible spend per workload, highest first.

    Each row: ``{"workload": str, "spent_usd": float, "cap_usd": float | None}``.
    ``cap_usd`` is the workload's ``budget_cap_usd_daily``, else ``default_cap``
    (the config-level ceiling), else ``None`` when neither is set.
    """
    with _connect_ro(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(w.name, ?) AS workload,
                SUM(c.cost_usd) AS spent_usd,
                MAX(w.budget_cap_usd_daily) AS cap_usd
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            WHERE c.project = ?
              AND date(c.ts) = date('now')
              AND c.budget_eligible != 0
            GROUP BY COALESCE(w.name, ?)
            ORDER BY spent_usd DESC
            """,
            (UNREGISTERED, project, UNREGISTERED),
        ).fetchall()

    result = []
    for name, spent, cap in rows:
        effective_cap = cap if cap is not None else default_cap
        result.append(
            {
                "workload": name,
                "spent_usd": float(spent or 0.0),
                "cap_usd": effective_cap,
            }
        )
    return result

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


# ---------------------------------------------------------------------------
# Call search — the one query behind `somm tail`, MCP `somm_search_calls`, and
# the service's /api/calls and session views.

_CALL_COLUMNS = (
    "id", "ts", "workload", "provider", "model", "tokens_in", "tokens_out",
    "latency_ms", "cost_usd", "outcome", "error_kind", "prompt_hash", "response_hash",
    "session_id", "parent_call_id", "correlation_id", "ttft_ms", "cache_tokens_in",
    "cache_tokens_out", "reasoning_tokens", "origin", "call_site", "client_version",
)


def search_calls(
    conn_or_path,
    project: str,
    *,
    since_days: int | None = 7,
    since: str | None = None,
    workload: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    outcome: str | None = None,
    q: str | None = None,
    origin: str | None = None,
    limit: int = 50,
    newest_first: bool = True,
) -> list[dict]:
    """Filtered rows from ``calls`` joined to their workload name.

    ``conn_or_path`` is an open ``sqlite3.Connection`` or a database path (opened
    read-only). ``since`` (ISO timestamp, exclusive) overrides ``since_days``.
    ``workload`` matches the registered name or the workload id. ``q`` is a
    substring search across id, provider, model, workload, session, and
    correlation ids. Every row carries the keys in ``_CALL_COLUMNS``; an
    unregistered workload reads ``(unregistered)``.
    """
    clauses = ["c.project = ?"]
    params: list[object] = [project]
    if since is not None:
        clauses.append("c.ts > ?")
        params.append(since)
    elif since_days is not None:
        clauses.append("c.ts >= datetime('now', ?)")
        params.append(f"-{int(since_days)} days")
    if workload:
        clauses.append("(w.name = ? OR c.workload_id = ?)")
        params.extend([workload, workload])
    if provider:
        clauses.append("c.provider = ?")
        params.append(provider)
    if model:
        clauses.append("c.model = ?")
        params.append(model)
    if outcome:
        clauses.append("c.outcome = ?")
        params.append(outcome)
    if origin:
        clauses.append("c.origin = ?")
        params.append(origin)
    if q:
        like = f"%{q}%"
        clauses.append(
            "(c.id LIKE ? OR c.provider LIKE ? OR c.model LIKE ? OR "
            "COALESCE(w.name, '') LIKE ? OR COALESCE(c.session_id, '') LIKE ? OR "
            "COALESCE(c.correlation_id, '') LIKE ?)"
        )
        params.extend([like] * 6)
    order = "DESC" if newest_first else "ASC"
    params.append(int(limit))
    sql = f"""
        SELECT c.id, c.ts, COALESCE(w.name, ?) AS workload, c.provider, c.model,
               c.tokens_in, c.tokens_out, c.latency_ms, c.cost_usd, c.outcome,
               c.error_kind, c.prompt_hash, c.response_hash, c.session_id,
               c.parent_call_id, c.correlation_id, c.ttft_ms, c.cache_tokens_in,
               c.cache_tokens_out, c.reasoning_tokens, c.origin, c.call_site,
               c.client_version
        FROM calls c
        LEFT JOIN workloads w ON w.id = c.workload_id
        WHERE {" AND ".join(clauses)}
        ORDER BY c.ts {order}
        LIMIT ?
    """
    params.insert(0, UNREGISTERED)
    rows = _execute(conn_or_path, sql, params)
    out = []
    for r in rows:
        row = dict(zip(_CALL_COLUMNS, r, strict=True))
        row["cost_usd"] = row["cost_usd"] or 0.0
        out.append(row)
    return out


def _execute(conn_or_path, sql: str, params) -> list:
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path.execute(sql, params).fetchall()
    with _connect_ro(Path(conn_or_path)) as conn:
        return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Fleet health — what `somm doctor --global` reports about a shared ledger.

SENTINEL_MODEL_PREFIX = "*"


def loop_state(db_path: Path) -> dict:
    """Whether the intelligence loop has produced anything in one ledger.

    Samples, eval results, and recommendations live only in a project's own
    ledger — the shared mirror carries calls and workloads, by design — so a
    fleet view must ask each ledger.
    """
    with _connect_ro(db_path) as conn:

        def count(sql: str) -> int:
            try:
                return conn.execute(sql).fetchone()[0]
            except sqlite3.Error:
                return 0

        def scalar(sql: str):
            try:
                return conn.execute(sql).fetchone()[0]
            except sqlite3.Error:
                return None

        return {
            "db_path": str(db_path),
            "samples": count("SELECT COUNT(*) FROM samples"),
            "eval_results": count("SELECT COUNT(*) FROM eval_results"),
            "recommendations": count("SELECT COUNT(*) FROM recommendations"),
            "recommendations_open": count(
                "SELECT COUNT(*) FROM recommendations "
                "WHERE applied_at IS NULL AND dismissed_at IS NULL"
            ),
            "decisions": count("SELECT COUNT(*) FROM decisions"),
            "worker_heartbeats": count("SELECT COUNT(*) FROM worker_heartbeat"),
            "last_worker_success": scalar("SELECT MAX(last_success_at) FROM worker_heartbeat"),
            "shadow_configured_workloads": count(
                "SELECT COUNT(*) FROM workloads WHERE shadow_config_json IS NOT NULL"
            ),
        }


def fleet_summary(
    db_path: Path, *, since_days: int = 7, ledgers: list[Path] | None = None
) -> dict:
    """Facts about a ledger that only make sense across projects.

    - ``clients``: rows per (project, client_version) in the window, so an
      operator can see which projects write with an old library;
    - ``loop``: the intelligence loop's state in *this* ledger (see
      :func:`loop_state`);
    - ``ledgers``: the same per registered project ledger (``ledgers``
      defaults to every path in ``~/.somm/registry.json``), because a mirror
      never carries samples, evals, or recommendations and would otherwise
      report a loop as dormant that is alive in a project;
    - ``sentinel_rows``: rows whose model is a routing sentinel or blank,
      which older clients wrote for exhaustion and which pollute rollups;
    - ``provider_spellings``: providers that differ only by ``-`` vs ``_``;
    - ``unaliased``: (provider, model) pairs seen in the window with no row in
      ``model_aliases``.
    """
    window = f"-{int(since_days)} days"
    with _connect_ro(db_path) as conn:
        clients = conn.execute(
            "SELECT project, COALESCE(client_version, '(pre-0.17)') AS client_version, "
            "COUNT(*) FROM calls WHERE ts >= datetime('now', ?) "
            "GROUP BY project, client_version ORDER BY project, 3 DESC",
            (window,),
        ).fetchall()

        sentinel_rows = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE ts >= datetime('now', ?) "
            "AND (model = '' OR model LIKE ? ESCAPE '\\')",
            (window, SENTINEL_MODEL_PREFIX.replace("*", "\\*") + "%"),
        ).fetchone()[0]
        providers = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT provider FROM calls WHERE ts >= datetime('now', ?)", (window,)
            )
        ]
        pairs = conn.execute(
            "SELECT DISTINCT c.provider, c.model FROM calls c "
            "LEFT JOIN model_aliases a ON a.provider = c.provider AND a.model = c.model "
            "WHERE c.ts >= datetime('now', ?) AND c.model != '' AND c.model NOT LIKE '*%' "
            "AND a.canonical_id IS NULL ORDER BY 1, 2",
            (window,),
        ).fetchall()

    by_norm: dict[str, set[str]] = {}
    for p in providers:
        if p:
            by_norm.setdefault(p.replace("_", "-"), set()).add(p)
    spellings = {k: sorted(v) for k, v in by_norm.items() if len(v) > 1}
    loop = loop_state(Path(db_path))
    if ledgers is None:
        try:
            from somm_core.registry import fleet_db_paths

            ledgers = [p for p in fleet_db_paths() if Path(p).resolve() != Path(db_path).resolve()]
        except Exception:  # noqa: BLE001 - the registry is optional
            ledgers = []
    per_ledger = []
    for path in ledgers:
        try:
            per_ledger.append(loop_state(Path(path)))
        except sqlite3.Error:
            continue
    return {
        "window_days": since_days,
        "clients": [
            {"project": r[0], "client_version": r[1], "calls": r[2]} for r in clients
        ],
        "loop": loop,
        "ledgers": per_ledger,
        "sentinel_rows": sentinel_rows,
        "provider_spellings": spellings,
        "unaliased": [(r[0], r[1]) for r in pairs],
    }

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

import json
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from somm_core.cost import cost_is_known, known_cost_sql
from somm_core.version import SCHEMA_VERSION

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
                -- A workload id is shared by every project that declares the
                -- same name and schemas; only this project's row sets its cap.
                MAX(CASE WHEN w.project = c.project THEN w.budget_cap_usd_daily END) AS cap_usd
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
    "latency_ms", "cost_usd", "outcome", "error_kind", "error_detail", "prompt_hash", "response_hash",
    "session_id", "parent_call_id", "correlation_id", "ttft_ms", "cache_tokens_in",
    "cache_tokens_out", "reasoning_tokens", "origin", "call_site", "client_version", "routing_json",
    "data_tier", "cost_source",
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
    unregistered workload reads ``(unregistered)``. A column the ledger's schema
    predates reads ``None``. Three derived keys are added: ``cost_known`` (see
    ``somm_core.cost.cost_is_known``), and ``route_source`` and ``route`` parsed
    from ``routing_json`` (``None`` when absent or unreadable).
    """
    with _ledger(conn_or_path) as conn:
        cols = _columns(conn, "calls")
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
            clauses.append(f"{_col('origin', cols)} = ?")
            params.append(origin)
        if q:
            like = f"%{q}%"
            clauses.append(
                "(c.id LIKE ? OR c.provider LIKE ? OR c.model LIKE ? OR "
                f"COALESCE(w.name, '') LIKE ? OR COALESCE({_col('session_id', cols)}, '') LIKE ? "
                f"OR COALESCE({_col('correlation_id', cols)}, '') LIKE ?)"
            )
            params.extend([like] * 6)
        order = "DESC" if newest_first else "ASC"
        sql = f"""
            SELECT {_select_calls(cols)}
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            WHERE {" AND ".join(clauses)}
            ORDER BY c.ts {order}
            LIMIT ?
        """
        rows = conn.execute(sql, [UNREGISTERED, *params, int(limit)]).fetchall()
    return [_call_row(r) for r in rows]


@contextmanager
def _ledger(conn_or_path):
    """Yield an open connection: the caller's own, or a read-only one we close."""
    if isinstance(conn_or_path, sqlite3.Connection):
        yield conn_or_path
        return
    conn = _connect_ro(Path(conn_or_path))
    try:
        yield conn
    finally:
        conn.close()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names of ``table``; empty when the table does not exist."""
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _col(name: str, cols: set[str], alias: str = "c") -> str:
    """``alias.name`` when the ledger has the column, else ``NULL``."""
    return f"{alias}.{name}" if name in cols else "NULL"


def _select_calls(cols: set[str]) -> str:
    return ", ".join(
        "COALESCE(w.name, ?) AS workload" if n == "workload" else f"{_col(n, cols)} AS {n}"
        for n in _CALL_COLUMNS
    )


def _call_row(r) -> dict:
    row = dict(zip(_CALL_COLUMNS, r, strict=True))
    row["cost_usd"] = row["cost_usd"] or 0.0
    row["cost_known"] = cost_is_known(row["cost_source"])
    row["route_source"], row["route"] = _parse_routing(row["routing_json"])
    return row


def _parse_routing(raw) -> tuple[str | None, object]:
    """``(route_source, route)`` from a ``routing_json`` value, else ``(None, None)``."""
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    return data.get("route_source"), data.get("route")


def get_calls(conn_or_path, ids: list[str]) -> list[dict]:
    """Rows for exactly these call ids (any project), shaped like ``search_calls``.

    Ids that are not in the ledger are simply absent from the result.
    """
    ids = list(dict.fromkeys(ids))
    out: list[dict] = []
    with _ledger(conn_or_path) as conn:
        cols = _columns(conn, "calls")
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT {_select_calls(cols)} FROM calls c "
                f"LEFT JOIN workloads w ON w.id = c.workload_id WHERE c.id IN ({marks})",
                [UNREGISTERED, *chunk],
            ).fetchall()
            out.extend(_call_row(r) for r in rows)
    return out


# ---------------------------------------------------------------------------
# Fleet health — what `somm doctor --global` reports about a shared ledger.

SENTINEL_MODEL_PREFIX = "*"


def loop_state(db_path: Path, project: str | None = None) -> dict:
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
            "project": project,
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
    - ``ledgers``: the same per registered project ledger, labelled with the
      project (``ledgers`` defaults to ``~/.somm/registry.json``), because a mirror
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
    labelled: list[tuple[str | None, Path]]
    if ledgers is None:
        try:
            from somm_core.registry import registered_ledgers

            labelled = [
                (project, path)
                for project, path in registered_ledgers()
                if Path(path).resolve() != Path(db_path).resolve()
            ]
        except Exception:  # noqa: BLE001 - the registry is optional
            labelled = []
    else:
        labelled = [(None, Path(p)) for p in ledgers]
    per_ledger = []
    for project, path in labelled:
        try:
            per_ledger.append(loop_state(Path(path), project))
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


# ---------------------------------------------------------------------------
# North-star numbers — VISION.md's decisions recalled, evidence-routed share,
# and recommendations applied, computed from rows the ledger already holds.

# Values of ``routing_json.route_source``, and whether each means the route was
# decided by somm's evidence rather than by the caller or the default chain.
ROUTE_SOURCES = {
    "recommendation": True,
    "policy": True,
    "learned_override": True,
    "pin": False,
    "default": False,
}
RECOMMENDATION_WINDOW_DAYS = 14


def north_star(db_path: Path, *, since_days: int = 30) -> dict:
    """The three north-star numbers for one ledger, each with its denominator.

    Every entry carries ``measurable``; when it is false, ``reason`` says why.

    - ``decisions_recalled``: decisions with ``recalled_from`` set, over all
      decisions. A decision bound to a workload id that has no calls in this
      ledger is test residue and is counted in ``excluded`` instead.
    - ``evidence_routed``: of production calls in the last ``since_days`` whose
      ``routing_json`` names a ``route_source``, how many were decided by a
      recommendation, a workload policy, or a learned override. Calls without
      the marker are ``unmarked``, never counted as pins.
    - ``recommendations_applied``: of recommendations whose fate is known
      (closed, or open for fourteen days), how many were applied or dismissed
      within fourteen days of being created.
    """
    with _connect_ro(db_path) as conn:
        return {
            "decisions_recalled": _decisions_recalled(conn),
            "evidence_routed": _evidence_routed(conn, since_days),
            "recommendations_applied": _recommendations_applied(conn),
        }


def _decisions_recalled(conn: sqlite3.Connection) -> dict:
    try:
        recalled, total, excluded = conn.execute(
            """
            SELECT COALESCE(SUM(residue = 0 AND recalled_from IS NOT NULL), 0),
                   COALESCE(SUM(residue = 0), 0),
                   COALESCE(SUM(residue), 0)
            FROM (
                SELECT d.recalled_from,
                       (d.workload_id IS NOT NULL AND NOT EXISTS (
                           SELECT 1 FROM calls c
                           WHERE c.workload_id = d.workload_id AND c.project = d.project
                       )) AS residue
                FROM decisions d
            )
            """
        ).fetchone()
    except sqlite3.Error:
        return {"measurable": False, "reason": "this ledger predates recall tracking (schema 28)"}
    out = {"measurable": total > 0, "recalled": recalled, "decisions": total, "excluded": excluded}
    if not total:
        out["reason"] = "no decisions recorded"
    return out


def _evidence_routed(conn: sqlite3.Connection, since_days: int) -> dict:
    try:
        rows = conn.execute(
            """
            SELECT CASE WHEN json_valid(routing_json)
                        THEN json_extract(routing_json, '$.route_source') END AS source,
                   COUNT(*)
            FROM calls
            WHERE ts >= datetime('now', ?) AND observation_role = 'production'
            GROUP BY source
            """,
            (f"-{int(since_days)} days",),
        ).fetchall()
    except sqlite3.Error as exc:
        return {
            "measurable": False,
            "window_days": since_days,
            "reason": f"calls on this ledger's schema cannot say ({exc})",
        }
    by_source = {s: n for s, n in rows if s in ROUTE_SOURCES}
    marked = sum(by_source.values())
    unmarked = sum(n for s, n in rows if s not in ROUTE_SOURCES)
    out = {
        "measurable": marked > 0,
        "window_days": since_days,
        "evidence": sum(n for s, n in by_source.items() if ROUTE_SOURCES[s]),
        "marked": marked,
        "unmarked": unmarked,
        "by_source": by_source,
    }
    if not marked:
        out["reason"] = (
            f"none of {unmarked} production calls in the window records why its "
            "route was chosen (routing_json.route_source)"
        )
    return out


def _recommendations_applied(conn: sqlite3.Connection) -> dict:
    days = RECOMMENDATION_WINDOW_DAYS
    try:
        acted, eligible, pending, applied = conn.execute(
            """
            SELECT COALESCE(SUM(closed IS NOT NULL
                                AND julianday(closed) - julianday(created_at) <= ?), 0),
                   COALESCE(SUM(closed IS NOT NULL
                                OR julianday('now') - julianday(created_at) >= ?), 0),
                   COALESCE(SUM(closed IS NULL
                                AND julianday('now') - julianday(created_at) < ?), 0),
                   COALESCE(SUM(applied_at IS NOT NULL), 0)
            FROM (SELECT created_at, applied_at,
                         COALESCE(applied_at, dismissed_at) AS closed
                  FROM recommendations)
            """,
            (days, days, days),
        ).fetchone()
    except sqlite3.Error as exc:
        return {
            "measurable": False,
            "window_days": days,
            "reason": f"recommendations cannot be read here ({exc})",
        }
    out = {
        "measurable": eligible > 0,
        "window_days": days,
        "acted": acted,
        "eligible": eligible,
        "pending": pending,
        "applied": applied,
    }
    if not eligible:
        out["reason"] = f"no recommendation is closed or {days} days old ({pending} younger)"
    return out


# ---------------------------------------------------------------------------
# Schema guard — call first; reads degrade rather than crash on an older ledger.

# Columns every read needs. All date from the first schema; a ledger without
# them is not a somm ledger this API can read.
_BASE_CALL_COLUMNS = frozenset(
    {"id", "ts", "project", "workload_id", "provider", "model", "tokens_in",
     "tokens_out", "latency_ms", "cost_usd", "outcome"}
)

# Table -> (has a project column, time column, order). The tables ``table_rows``
# exports; every column is returned (SELECT *).
_TABLES: dict[str, tuple[bool, str, tuple[str, ...]]] = {
    "calls": (True, "ts", ("ts", "id")),
    "workloads": (True, "created_at", ("created_at", "id")),
    "eval_results": (False, "ts", ("ts", "id")),
    "eval_receipts": (False, "created_at", ("created_at", "id")),
    "campaigns": (True, "created_at", ("created_at", "id")),
    "campaign_events": (False, "created_at", ("created_at", "campaign_id", "sequence")),
    "decisions": (True, "ts", ("ts", "id")),
    "recommendations": (False, "created_at", ("created_at", "id")),
    "call_updates": (False, "ts", ("ts", "id")),
}


def schema_status(conn_or_path) -> dict:
    """Whether this version of the read API understands a ledger's schema.

    Call it before anything else. Returns::

        {"schema_version": int,            # what the ledger says (0 = none)
         "library_schema_version": int,    # what this somm_core writes
         "understood": bool,
         "reason": str | None,             # why not, when not
         "missing_columns": {"calls": [...]},   # newer columns this ledger lacks
         "missing_tables": [...]}          # export tables this ledger lacks

    ``understood`` is true when ``calls`` and ``workloads`` have the columns
    every read needs and the ledger is not newer than this library. Missing
    columns and tables do not make a ledger unreadable: reads return ``None``
    for a missing ``calls`` column and no rows for a missing table.
    """
    from somm_core.schema import current_schema_version

    with _ledger(conn_or_path) as conn:
        version = current_schema_version(conn)
        cols = _columns(conn, "calls")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    reason = None
    if not cols:
        reason = "no calls table: not a somm ledger"
    elif _BASE_CALL_COLUMNS - cols:
        reason = f"calls lacks {', '.join(sorted(_BASE_CALL_COLUMNS - cols))}"
    elif "workloads" not in tables:
        reason = "no workloads table"
    elif version > SCHEMA_VERSION:
        reason = f"ledger schema {version} is newer than this library's {SCHEMA_VERSION}; upgrade somm"
    return {
        "schema_version": version,
        "library_schema_version": SCHEMA_VERSION,
        "understood": reason is None,
        "reason": reason,
        "missing_columns": {
            "calls": sorted(c for c in _CALL_COLUMNS if c != "workload" and c not in cols)
        },
        "missing_tables": sorted(t for t in _TABLES if t not in tables),
    }


# ---------------------------------------------------------------------------
# Windowed rollups: provider/model health and per-dimension totals.


def _where(
    cols: set[str], *, project, workload, provider, model, outcome=None,
    since_days, since, until, role,
) -> tuple[list[str], list]:
    """WHERE clauses over ``calls c`` (joined to ``workloads w`` when filtering by workload)."""
    clauses: list[str] = []
    params: list = []
    if project is not None:
        clauses.append("c.project = ?")
        params.append(project)
    if since is not None:
        clauses.append("c.ts >= ?")
        params.append(since)
    elif since_days is not None:
        clauses.append("c.ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', ?)")
        params.append(f"-{int(since_days)} days")
    if until is not None:
        clauses.append("c.ts < ?")
        params.append(until)
    if workload:
        clauses.append("(w.name = ? OR c.workload_id = ?)")
        params.extend([workload, workload])
    for name, value in (("provider", provider), ("model", model), ("outcome", outcome)):
        if value:
            clauses.append(f"c.{name} = ?")
            params.append(value)
    if role and "observation_role" in cols:
        clauses.append("c.observation_role = ?")
        params.append(role)
    return clauses, params


def _percentile(sorted_values: list, pct: float):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1))))]


def provider_health(
    conn_or_path,
    project: str | None = None,
    *,
    since_days: int | None = 7,
    since: str | None = None,
    until: str | None = None,
    workload: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    role: str | None = "production",
) -> list[dict]:
    """How each (provider, model) has been doing over a window, busiest first.

    ``since`` (ISO, inclusive) overrides ``since_days``; ``until`` is exclusive;
    ``project=None`` spans every project in the ledger; ``role`` is the
    observation role (``None`` for all). Each row::

        provider, model, calls, ok, errors, ok_rate, tokens_in, tokens_out,
        p50_ms, p95_ms          # latency of ok calls (nearest rank); None if no ok call
        last_ok_ts, last_error_ts,
        last_error_kind, last_error_outcome   # of the most recent non-ok call
    """
    with _ledger(conn_or_path) as conn:
        cols = _columns(conn, "calls")
        clauses, params = _where(
            cols, project=project, workload=workload, provider=provider, model=model,
            since_days=since_days, since=since, until=until, role=role,
        )
        join = " LEFT JOIN workloads w ON w.id = c.workload_id" if workload else ""
        base = f"FROM calls c{join} WHERE {' AND '.join(clauses) or '1'}"
        agg = conn.execute(
            f"""SELECT c.provider, c.model, COUNT(*), SUM(c.outcome = 'ok'),
                       COALESCE(SUM(c.tokens_in), 0), COALESCE(SUM(c.tokens_out), 0),
                       MAX(CASE WHEN c.outcome = 'ok' THEN c.ts END),
                       MAX(CASE WHEN c.outcome != 'ok' THEN c.ts END)
                {base} GROUP BY c.provider, c.model ORDER BY COUNT(*) DESC""",
            params,
        ).fetchall()
        # SQLite takes the bare columns from the row that holds the single MAX().
        last_err = {
            (r[0], r[1]): (r[2], r[3])
            for r in conn.execute(
                f"""SELECT c.provider, c.model, {_col('error_kind', cols)}, c.outcome, MAX(c.ts)
                    {base} AND c.outcome != 'ok' GROUP BY c.provider, c.model""",
                params,
            )
        }
        latencies: dict[tuple, list] = {}
        for p, m, ms in conn.execute(
            f"""SELECT c.provider, c.model, c.latency_ms {base}
                AND c.outcome = 'ok' AND c.latency_ms IS NOT NULL
                ORDER BY c.provider, c.model, c.latency_ms""",
            params,
        ):
            latencies.setdefault((p, m), []).append(ms)
    out = []
    for p, m, n, ok, t_in, t_out, last_ok, last_error in agg:
        ok = ok or 0
        lat = latencies.get((p, m), [])
        kind, outcome = last_err.get((p, m), (None, None))
        out.append({
            "provider": p, "model": m, "calls": n, "ok": ok, "errors": n - ok,
            "ok_rate": ok / n if n else None, "tokens_in": t_in, "tokens_out": t_out,
            "p50_ms": _percentile(lat, 0.50), "p95_ms": _percentile(lat, 0.95),
            "last_ok_ts": last_ok, "last_error_ts": last_error,
            "last_error_kind": kind, "last_error_outcome": outcome,
        })
    return out


def _dimension(name: str, cols: set[str]) -> str:
    sql = {
        "day": "date(c.ts)",
        "project": "c.project",
        "workload": f"COALESCE(w.name, '{UNREGISTERED}')",
        "provider": "c.provider",
        "model": "c.model",
        "outcome": "c.outcome",
        "data_tier": _col("data_tier", cols),
    }.get(name)
    if sql is None:
        raise ValueError(
            f"unknown rollup dimension {name!r}; use any of "
            "day, project, workload, provider, model, outcome, data_tier"
        )
    return sql


def rollup(
    conn_or_path,
    project: str | None = None,
    *,
    by: tuple[str, ...] | list[str] = ("day",),
    since_days: int | None = 30,
    since: str | None = None,
    until: str | None = None,
    workload: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    outcome: str | None = None,
    role: str | None = "production",
) -> list[dict]:
    """Totals grouped by any of ``day`` (UTC date), ``project``, ``workload``,
    ``provider``, ``model``, ``outcome``, ``data_tier``; sorted by the grouping.

    Filters as in :func:`provider_health`. Each row has the grouping keys plus
    ``calls``, ``ok``, ``tokens_in``, ``tokens_out``, ``cache_tokens_in``,
    ``avg_latency_ms``, ``cost_usd`` and ``unknown_cost_calls``. ``cost_usd``
    sums only calls whose price somm knew; a call with no known price is
    recorded as 0.0 and counted in ``unknown_cost_calls`` instead, so unknown
    never reads as free. ``by=()`` gives one total row.
    """
    with _ledger(conn_or_path) as conn:
        cols = _columns(conn, "calls")
        dims = [(name, _dimension(name, cols)) for name in by]
        clauses, params = _where(
            cols, project=project, workload=workload, provider=provider, model=model,
            outcome=outcome, since_days=since_days, since=since, until=until, role=role,
        )
        join = (
            " LEFT JOIN workloads w ON w.id = c.workload_id"
            if workload or "workload" in by
            else ""
        )
        known = known_cost_sql("c.") if "cost_source" in cols else "1"
        select_dims = "".join(f"{sql}, " for _, sql in dims)
        group = ""
        if dims:
            exprs = ", ".join(sql for _, sql in dims)
            group = f" GROUP BY {exprs} ORDER BY {exprs}"
        rows = conn.execute(
            f"""SELECT {select_dims}COUNT(*), COALESCE(SUM(c.outcome = 'ok'), 0),
                       COALESCE(SUM(c.tokens_in), 0), COALESCE(SUM(c.tokens_out), 0),
                       COALESCE(SUM({_col('cache_tokens_in', cols)}), 0),
                       AVG(c.latency_ms),
                       COALESCE(SUM(CASE WHEN {known} THEN c.cost_usd ELSE 0 END), 0),
                       COALESCE(SUM(NOT ({known})), 0)
                FROM calls c{join}
                WHERE {' AND '.join(clauses) or '1'}{group}""",
            params,
        ).fetchall()
    n = len(dims)
    keys = ("calls", "ok", "tokens_in", "tokens_out", "cache_tokens_in",
            "avg_latency_ms", "cost_usd", "unknown_cost_calls")
    out = []
    for r in rows:
        row = {name: r[i] for i, (name, _) in enumerate(dims)}
        row.update(zip(keys, r[n:], strict=True))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Whole-table rows — eval results, recommendations, decisions, and the rest.


def table_rows(
    conn_or_path,
    table: str,
    *,
    since: str | None = None,
    until: str | None = None,
    project: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Every column of ``table`` as dicts, oldest first.

    ``table`` is one of ``calls``, ``workloads``, ``eval_results``,
    ``eval_receipts``, ``campaigns``, ``campaign_events``, ``decisions``,
    ``recommendations``, ``call_updates``. ``since`` (inclusive) and ``until``
    (exclusive) bound the table's own time column (``ts`` or ``created_at``) by
    ISO string; ``project`` filters the tables that have one and raises
    ``ValueError`` on the others. A table the ledger predates returns ``[]``
    (see :func:`schema_status`). Keys are the column names, so they follow the
    additive contract. For a large table, page with ``since`` and ``limit``.
    """
    if table not in _TABLES:
        raise ValueError(f"unknown table {table!r}; use one of {', '.join(_TABLES)}")
    has_project, time_col, order = _TABLES[table]
    if project is not None and not has_project:
        raise ValueError(f"{table} has no project column")
    clauses: list[str] = []
    params: list = []
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if since is not None:
        clauses.append(f"{time_col} >= ?")
        params.append(since)
    if until is not None:
        clauses.append(f"{time_col} < ?")
        params.append(until)
    sql = f"SELECT * FROM {table}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY " + ", ".join(order)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _ledger(conn_or_path) as conn:
        if not _columns(conn, table):
            return []
        cur = conn.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Every ledger in the registry.


def across_ledgers(
    read_fn: Callable,
    *,
    ledgers: list | None = None,
    pass_project: bool = False,
    **kwargs,
) -> dict:
    """Run one read over every registered project ledger.

    ``read_fn(path, **kwargs)`` is any function above that takes a ledger as
    its first argument (``pass_project=True`` calls ``read_fn(path, project,
    **kwargs)``, for ``search_calls`` and ``spend_today``). Returns
    ``{"rows": [...], "skipped": [...]}``: each row is a dict from the read
    with ``ledger_project`` set to the registry's name for the ledger (a
    function that returns one dict contributes one row); each skipped entry is
    ``{"project", "db_path", "reason"}`` for a ledger that is missing or that
    raised ``sqlite3.Error``. ``ledgers`` is a list of ``(project, path)``
    pairs and defaults to ``~/.somm/registry.json``. The shared global mirror
    is not a project ledger; read it directly.
    """
    if ledgers is None:
        from somm_core.registry import _load, registry_path

        entries = _load(registry_path())["projects"]
        ledgers = [(name, e.get("db_path")) for name, e in entries.items()]
    rows: list[dict] = []
    skipped: list[dict] = []
    for project, raw in ledgers:
        path = Path(raw) if raw else None
        if path is None or not path.exists():
            skipped.append(
                {"project": project, "db_path": str(raw or ""), "reason": "database file is missing"}
            )
            continue
        try:
            got = read_fn(path, project, **kwargs) if pass_project else read_fn(path, **kwargs)
        except sqlite3.Error as exc:
            skipped.append(
                {"project": project, "db_path": str(path), "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for row in [got] if isinstance(got, dict) else got:
            rows.append({**row, "ledger_project": project})
    return {"rows": rows, "skipped": skipped}

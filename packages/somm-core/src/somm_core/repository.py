"""Repository — the single query/write API used by library, service, and MCP.

Owns the SQLite connection. Applies schema on init. Exposes typed methods.
Same surface is used by MCP stdio (direct) and MCP HTTP (service-proxied).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from somm_core.models import Call, Decision, Outcome, PrivacyClass, Prompt, Workload
from somm_core.parse import prompt_id as _prompt_id
from somm_core.parse import stable_hash
from somm_core.parse import workload_id as _workload_id
from somm_core.schema import ensure_schema


class Repository:
    """SQLite-backed repository. Thread-safe via per-call connection creation.

    For high-write paths, use `somm.telemetry.WriterQueue` (wraps a single
    long-lived connection). For reads and low-volume writes, Repository
    opens short-lived connections per call.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # chmod 0700 on dir, 0600 on DB — shared-machine safety.
        self.db_path.parent.chmod(0o700)
        with self._open() as conn:
            ensure_schema(conn)
        self.db_path.chmod(0o600)

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # autocommit; we manage transactions
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # Workloads ---------------------------------------------------------------

    def register_workload(
        self,
        name: str,
        project: str,
        description: str = "",
        input_schema: dict | None = None,
        output_schema: dict | None = None,
        quality_criteria: list[str] | None = None,
        budget_cap_usd_daily: float | None = None,
        privacy_class: PrivacyClass = PrivacyClass.INTERNAL,
        capabilities_required: list[str] | None = None,
    ) -> Workload:
        wid = _workload_id(name, input_schema, output_schema)
        with self._open() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO workloads (
                    id, name, project, description,
                    input_schema_json, output_schema_json, quality_criteria_json,
                    budget_cap_usd_daily, privacy_class, capabilities_required_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wid,
                    name,
                    project,
                    description,
                    json.dumps(input_schema) if input_schema else None,
                    json.dumps(output_schema) if output_schema else None,
                    json.dumps(quality_criteria or []),
                    budget_cap_usd_daily,
                    privacy_class.value,
                    json.dumps(capabilities_required) if capabilities_required else None,
                ),
            )
        return Workload(
            id=wid,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            quality_criteria=quality_criteria or [],
            budget_cap_usd_daily=budget_cap_usd_daily,
            privacy_class=privacy_class,
            capabilities_required=list(capabilities_required or []),
        )

    def workload_by_name(self, name: str, project: str) -> Workload | None:
        with self._open() as conn:
            row = conn.execute(
                "SELECT id, name, description, input_schema_json, output_schema_json, "
                "quality_criteria_json, budget_cap_usd_daily, privacy_class, "
                "capabilities_required_json "
                "FROM workloads WHERE project = ? AND name = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (project, name),
            ).fetchone()
        if not row:
            return None
        return Workload(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            input_schema=json.loads(row[3]) if row[3] else None,
            output_schema=json.loads(row[4]) if row[4] else None,
            quality_criteria=json.loads(row[5]) if row[5] else [],
            budget_cap_usd_daily=row[6],
            privacy_class=PrivacyClass(row[7]),
            capabilities_required=json.loads(row[8]) if row[8] else [],
        )

    # Shadow-eval config ------------------------------------------------------

    def set_shadow_config(self, workload_id: str, config: dict | None) -> None:
        """Attach (or clear) shadow-eval config for a workload.

        config = None → shadow disabled.
        config = {"gold_provider": ..., "gold_model": ..., "sample_rate": ...,
                  "budget_usd_daily": ...} → enabled.
        """
        with self._open() as conn:
            conn.execute(
                "UPDATE workloads SET shadow_config_json = ? WHERE id = ?",
                (json.dumps(config) if config else None, workload_id),
            )

    def get_shadow_config(self, workload_id: str) -> dict | None:
        with self._open() as conn:
            row = conn.execute(
                "SELECT shadow_config_json FROM workloads WHERE id = ?",
                (workload_id,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    # Prompts -----------------------------------------------------------------

    def register_prompt(self, workload_id: str, body: str, version: str = "v1") -> Prompt:
        pid = _prompt_id(body)
        with self._open() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO prompts (id, workload_id, version, hash, body) "
                "VALUES (?, ?, ?, ?, ?)",
                (pid, workload_id, version, pid, body),
            )
        return Prompt(
            id=pid,
            workload_id=workload_id,
            version=version,
            hash=pid,
            body=body,
        )

    # Calls -------------------------------------------------------------------

    def write_call(self, call: Call) -> None:
        """Single-call insert. For high-volume use somm.telemetry.WriterQueue."""
        with self._open() as conn:
            conn.execute(
                """
                INSERT INTO calls (
                    id, ts, project, workload_id, prompt_id,
                    provider, model,
                    tokens_in, tokens_out, latency_ms, cost_usd,
                    outcome, error_kind, error_detail, prompt_hash, response_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.id,
                    call.ts.isoformat(),
                    call.project,
                    call.workload_id,
                    call.prompt_id,
                    call.provider,
                    call.model,
                    call.tokens_in,
                    call.tokens_out,
                    call.latency_ms,
                    call.cost_usd,
                    call.outcome.value,
                    call.error_kind,
                    call.error_detail,
                    call.prompt_hash,
                    call.response_hash,
                ),
            )

    def write_calls_batch(self, calls: list[Call]) -> None:
        """Batch insert used by WriterQueue."""
        if not calls:
            return
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    """
                    INSERT INTO calls (
                        id, ts, project, workload_id, prompt_id,
                        provider, model,
                        tokens_in, tokens_out, latency_ms, cost_usd,
                        outcome, error_kind, error_detail, prompt_hash, response_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            c.id,
                            c.ts.isoformat(),
                            c.project,
                            c.workload_id,
                            c.prompt_id,
                            c.provider,
                            c.model,
                            c.tokens_in,
                            c.tokens_out,
                            c.latency_ms,
                            c.cost_usd,
                            c.outcome.value,
                            c.error_kind,
                            c.error_detail,
                            c.prompt_hash,
                            c.response_hash,
                        )
                        for c in calls
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get_call(self, call_id: str) -> Call | None:
        with self._open() as conn:
            row = conn.execute(
                "SELECT id, ts, project, workload_id, prompt_id, provider, model, "
                "tokens_in, tokens_out, latency_ms, cost_usd, outcome, error_kind, "
                "prompt_hash, response_hash, error_detail FROM calls WHERE id = ?",
                (call_id,),
            ).fetchone()
        if not row:
            return None
        return Call(
            id=row[0],
            ts=datetime.fromisoformat(row[1]),
            project=row[2],
            workload_id=row[3],
            prompt_id=row[4],
            provider=row[5],
            model=row[6],
            tokens_in=row[7],
            tokens_out=row[8],
            latency_ms=row[9],
            cost_usd=row[10],
            outcome=Outcome(row[11]),
            error_kind=row[12],
            prompt_hash=row[13],
            response_hash=row[14],
            error_detail=row[15],
        )

    def record_outcome_update(self, call_id: str, outcome: Outcome) -> None:
        """Late-arriving outcome mark. Goes into call_updates, not calls."""
        with self._open() as conn:
            conn.execute(
                "INSERT INTO call_updates (call_id, field, value) VALUES (?, 'outcome', ?)",
                (call_id, outcome.value),
            )

    # Rollups -----------------------------------------------------------------

    def stats_by_workload(self, project: str, since_days: int = 7) -> list[dict]:
        with self._open() as conn:
            rows = conn.execute(
                """
                SELECT
                    COALESCE(w.name, '(unregistered)') AS workload,
                    c.provider,
                    c.model,
                    COUNT(*) AS n_calls,
                    SUM(c.tokens_in) AS tokens_in,
                    SUM(c.tokens_out) AS tokens_out,
                    SUM(c.cost_usd) AS cost_usd,
                    AVG(c.latency_ms) AS latency_ms_avg,
                    SUM(CASE WHEN c.outcome != 'ok' THEN 1 ELSE 0 END) AS n_failed
                FROM calls c
                LEFT JOIN workloads w ON w.id = c.workload_id
                WHERE c.project = ?
                  AND c.ts >= datetime('now', ?)
                GROUP BY workload, c.provider, c.model
                ORDER BY cost_usd DESC NULLS LAST
                """,
                (project, f"-{since_days} days"),
            ).fetchall()
        return [
            {
                "workload": r[0],
                "provider": r[1],
                "model": r[2],
                "n_calls": r[3],
                "tokens_in": r[4],
                "tokens_out": r[5],
                "cost_usd": r[6],
                "latency_ms_avg": r[7],
                "n_failed": r[8],
            }
            for r in rows
        ]

    # Decisions (sommelier) --------------------------------------------------

    def record_decision(self, decision: Decision) -> None:
        """Persist a decision row. Idempotent on (id)."""
        with self._open() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decisions (
                    id, ts, project, workload_id, workload_name,
                    question, question_hash, constraints_json, candidates_json,
                    chosen_provider, chosen_model, rationale, agent,
                    superseded_by, outcome_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.ts.isoformat(),
                    decision.project,
                    decision.workload_id,
                    decision.workload_name,
                    decision.question,
                    decision.question_hash,
                    json.dumps(decision.constraints) if decision.constraints else None,
                    json.dumps(decision.candidates),
                    decision.chosen_provider,
                    decision.chosen_model,
                    decision.rationale,
                    decision.agent,
                    decision.superseded_by,
                    decision.outcome_note,
                ),
            )

    def search_decisions(
        self,
        question: str | None = None,
        project: str | None = None,
        workload: str | None = None,
        chosen_provider: str | None = None,
        limit: int = 20,
    ) -> list[Decision]:
        """Search decisions. If `question` is provided, matches by question_hash
        first (exact); falls back to LIKE on the natural-language text.

        Results are newest-first. Supersession is surfaced via the
        `superseded_by` field — callers decide whether to show or hide.
        """
        clauses: list[str] = []
        params: list = []
        if question:
            clauses.append("(question_hash = ? OR question LIKE ?)")
            params.append(stable_hash(_normalise_question(question)))
            params.append(f"%{question}%")
        if project:
            clauses.append("project = ?")
            params.append(project)
        if workload:
            clauses.append("(workload_name = ? OR workload_id = ?)")
            params.append(workload)
            params.append(workload)
        if chosen_provider:
            clauses.append("chosen_provider = ?")
            params.append(chosen_provider)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, ts, project, workload_id, workload_name, question, "
            "       question_hash, constraints_json, candidates_json, "
            "       chosen_provider, chosen_model, rationale, agent, "
            "       superseded_by, outcome_note "
            f"FROM decisions {where} ORDER BY ts DESC LIMIT ?"
        )
        params.append(limit)
        with self._open() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_decision(r) for r in rows]

    def get_decision(self, decision_id: str) -> Decision | None:
        with self._open() as conn:
            row = conn.execute(
                "SELECT id, ts, project, workload_id, workload_name, question, "
                "       question_hash, constraints_json, candidates_json, "
                "       chosen_provider, chosen_model, rationale, agent, "
                "       superseded_by, outcome_note FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
        return _row_to_decision(row) if row else None

    def mark_decision_outcome(self, decision_id: str, note: str) -> None:
        """Attach a retrospective note. Idempotent — overwrites."""
        with self._open() as conn:
            conn.execute(
                "UPDATE decisions SET outcome_note = ? WHERE id = ?",
                (note, decision_id),
            )


def _normalise_question(q: str) -> str:
    return " ".join(q.strip().lower().split())


def _row_to_decision(row) -> Decision:
    return Decision(
        id=row[0],
        ts=datetime.fromisoformat(row[1]),
        project=row[2],
        workload_id=row[3],
        workload_name=row[4],
        question=row[5],
        question_hash=row[6],
        constraints=json.loads(row[7]) if row[7] else None,
        candidates=json.loads(row[8]) if row[8] else [],
        chosen_provider=row[9],
        chosen_model=row[10],
        rationale=row[11],
        agent=row[12],
        superseded_by=row[13],
        outcome_note=row[14],
    )

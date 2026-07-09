"""Repository — the single query/write API used by library, service, and MCP.

Owns the SQLite connection. Applies schema on init. Exposes typed methods.
Same surface is used by MCP stdio (direct) and MCP HTTP (service-proxied).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from somm_core.models import Call, Decision, Outcome, PrivacyClass, Prompt, Workload
from somm_core.parse import prompt_id as _prompt_id
from somm_core.parse import stable_hash
from somm_core.parse import workload_id as _workload_id
from somm_core.schema import ensure_schema

_POLICY_KEYS = {"fallback", "retry", "timeout_s"}
_RETRY_KEYS = {"max", "backoff_s", "deadline_s"}


def _percentiles(csv: str | None) -> tuple[int | None, int | None]:
    """Return (p50, p95) of a comma-separated latency_ms list using nearest-rank.

    Returns (None, None) when there are no usable values. Nearest-rank handles
    small samples sensibly: with n=2, p50 returns the smaller value and p95
    returns the larger; with n=1, both return the only value.
    """
    if not csv:
        return None, None
    try:
        values = sorted(int(v) for v in csv.split(",") if v)
    except ValueError:
        return None, None
    if not values:
        return None, None
    n = len(values)
    p50_idx = max(0, min(n - 1, math.ceil(0.50 * n) - 1))
    p95_idx = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
    return values[p50_idx], values[p95_idx]


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_workload_policy(policy: dict | None) -> dict | None:
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise ValueError("workload policy must be a JSON object")
    unknown = set(policy) - _POLICY_KEYS
    if unknown:
        raise ValueError(f"workload policy has unknown key(s): {', '.join(sorted(unknown))}")

    out: dict = {}
    if "fallback" in policy:
        fallback = policy["fallback"]
        if not isinstance(fallback, list):
            raise ValueError("workload policy fallback must be a list")
        normalized_fallback: list[dict] = []
        for idx, item in enumerate(fallback):
            if not isinstance(item, dict):
                raise ValueError(f"workload policy fallback[{idx}] must be an object")
            unknown_item = set(item) - {"provider", "model"}
            if unknown_item:
                raise ValueError(
                    "workload policy fallback["
                    f"{idx}] has unknown key(s): {', '.join(sorted(unknown_item))}"
                )
            provider = item.get("provider")
            if not isinstance(provider, str) or not provider.strip():
                raise ValueError(f"workload policy fallback[{idx}].provider must be a string")
            model = item.get("model")
            if model is not None and not isinstance(model, str):
                raise ValueError(f"workload policy fallback[{idx}].model must be a string or null")
            normalized_fallback.append({"provider": provider, "model": model})
        out["fallback"] = normalized_fallback

    if "retry" in policy:
        retry = policy["retry"]
        if retry is None:
            out["retry"] = None
        else:
            if not isinstance(retry, dict):
                raise ValueError("workload policy retry must be an object")
            unknown_retry = set(retry) - _RETRY_KEYS
            if unknown_retry:
                raise ValueError(
                    f"workload policy retry has unknown key(s): {', '.join(sorted(unknown_retry))}"
                )
            normalized_retry: dict = {}
            if "max" in retry:
                retry_max = retry["max"]
                if (
                    not isinstance(retry_max, int)
                    or isinstance(retry_max, bool)
                    or not math.isfinite(float(retry_max))
                    or retry_max < 0
                ):
                    raise ValueError(
                        "workload policy retry.max must be a finite non-negative integer"
                    )
                normalized_retry["max"] = retry_max
            for key in ("backoff_s", "deadline_s"):
                if key in retry:
                    value = retry[key]
                    if not _is_number(value) or float(value) < 0:
                        raise ValueError(
                            f"workload policy retry.{key} must be a finite non-negative number"
                        )
                    if key == "deadline_s" and float(value) <= 0:
                        raise ValueError(
                            "workload policy retry.deadline_s must be a finite number "
                            "greater than zero"
                        )
                    normalized_retry[key] = float(value)
            out["retry"] = normalized_retry

    if "timeout_s" in policy:
        timeout_s = policy["timeout_s"]
        if not _is_number(timeout_s) or float(timeout_s) <= 0:
            raise ValueError("workload policy timeout_s must be a finite positive number")
        out["timeout_s"] = float(timeout_s)
    return out


class Repository:
    """SQLite-backed repository. Thread-safe via per-thread connection reuse.

    For high-write paths, use `somm.telemetry.WriterQueue` (wraps a single
    long-lived connection). For reads and low-volume writes, Repository
    reuses one connection per Repository instance per thread.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # chmod 0700 on dir, 0600 on DB — shared-machine safety.
        self.db_path.parent.chmod(0o700)
        with self._open() as conn:
            ensure_schema(conn)
        self.db_path.chmod(0o600)

    def _open(self) -> sqlite3.Connection:
        pid = os.getpid()
        cached = getattr(self._local, "conn", None)
        cached_pid = getattr(self._local, "pid", None)
        if cached is not None and cached_pid == pid:
            return cached
        # A pid mismatch means the connection was inherited across fork().
        # Abandon it without close(): closing an inherited handle can
        # finalize statements / checkpoint WAL against the parent's state.
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # autocommit; we manage transactions
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        self._local.conn = conn
        self._local.pid = pid
        return conn

    def close(self) -> None:
        """Close this thread's cached SQLite connection, if one exists."""
        cached = getattr(self._local, "conn", None)
        if cached is not None:
            try:
                cached.close()
            finally:
                self._local.conn = None
                self._local.pid = None

    # Workloads ---------------------------------------------------------------

    @staticmethod
    def _workload_revision_row(row) -> dict:
        return {
            "id": row[0],
            "workload_id": row[1],
            "revision": row[2],
            "config": json.loads(row[3]),
            "created_at": row[4],
            "created_by": row[5],
        }

    def _workload_config_snapshot(
        self,
        conn: sqlite3.Connection,
        workload_id: str,
    ) -> dict | None:
        row = conn.execute(
            "SELECT max_p95_latency_ms, max_capability_failure_rate, "
            "max_cost_per_call_usd, budget_cap_usd_daily, shadow_config_json, "
            "policy_json "
            "FROM workloads WHERE id = ?",
            (workload_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "max_p95_latency_ms": row[0],
            "max_capability_failure_rate": row[1],
            "max_cost_per_call_usd": row[2],
            "budget_cap_usd_daily": row[3],
            "shadow_config": json.loads(row[4]) if row[4] else None,
            "policy": json.loads(row[5]) if row[5] else None,
        }

    def _record_workload_revision_in_tx(
        self,
        conn: sqlite3.Connection,
        workload_id: str,
        config: dict,
        created_by: str | None,
    ) -> int:
        revision = conn.execute(
            "SELECT COALESCE(MAX(revision), 0) + 1 "
            "FROM workload_revisions WHERE workload_id = ?",
            (workload_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO workload_revisions "
            "(workload_id, revision, config_json, created_by) "
            "VALUES (?, ?, ?, ?)",
            (
                workload_id,
                revision,
                json.dumps(config, sort_keys=True),
                created_by,
            ),
        )
        return int(revision)

    def _ensure_initial_workload_revision_in_tx(
        self,
        conn: sqlite3.Connection,
        workload_id: str,
    ) -> None:
        has_revision = conn.execute(
            "SELECT 1 FROM workload_revisions WHERE workload_id = ? LIMIT 1",
            (workload_id,),
        ).fetchone()
        if has_revision is not None:
            return
        snapshot = self._workload_config_snapshot(conn, workload_id)
        if snapshot is not None:
            self._record_workload_revision_in_tx(conn, workload_id, snapshot, None)

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
        max_p95_latency_ms: int | None = None,
        max_capability_failure_rate: float | None = None,
        max_cost_per_call_usd: float | None = None,
        policy: dict | None = None,
    ) -> Workload:
        wid = _workload_id(name, input_schema, output_schema)
        policy = _validate_workload_policy(policy)
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workloads (
                        id, name, project, description,
                        input_schema_json, output_schema_json, quality_criteria_json,
                        budget_cap_usd_daily, privacy_class, capabilities_required_json,
                        max_p95_latency_ms, max_capability_failure_rate,
                        max_cost_per_call_usd, policy_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        max_p95_latency_ms,
                        max_capability_failure_rate,
                        max_cost_per_call_usd,
                        json.dumps(policy, sort_keys=True) if policy else None,
                    ),
                )
                self._ensure_initial_workload_revision_in_tx(conn, wid)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
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
            max_p95_latency_ms=max_p95_latency_ms,
            max_capability_failure_rate=max_capability_failure_rate,
            max_cost_per_call_usd=max_cost_per_call_usd,
            policy=policy,
        )

    def workload_by_name(self, name: str, project: str) -> Workload | None:
        with self._open() as conn:
            row = conn.execute(
                "SELECT id, name, description, input_schema_json, output_schema_json, "
                "quality_criteria_json, budget_cap_usd_daily, privacy_class, "
                "capabilities_required_json, "
                "max_p95_latency_ms, max_capability_failure_rate, "
                "max_cost_per_call_usd, policy_json "
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
            max_p95_latency_ms=row[9],
            max_capability_failure_rate=row[10],
            max_cost_per_call_usd=row[11],
            policy=json.loads(row[12]) if row[12] else None,
        )

    def set_workload_constraints(
        self,
        workload_id: str,
        *,
        max_p95_latency_ms: int | None = None,
        max_capability_failure_rate: float | None = None,
        max_cost_per_call_usd: float | None = None,
        clear: bool = False,
    ) -> None:
        """Update adequacy thresholds on the live workload row.

        Pass ``clear=True`` to set all three back to NULL. Otherwise,
        only fields with a non-None value here are written; existing
        values are preserved (use ``clear`` then re-set if you need to
        null one specifically).

        Dual-write note: the workloads row remains the current source for
        hot-path routing reads. After each row update, workload_revisions gets
        an append-only snapshot for audit, diff, and forward-only rollback.
        """
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if clear:
                    self._ensure_initial_workload_revision_in_tx(conn, workload_id)
                    cursor = conn.execute(
                        "UPDATE workloads SET "
                        "max_p95_latency_ms = NULL, "
                        "max_capability_failure_rate = NULL, "
                        "max_cost_per_call_usd = NULL "
                        "WHERE id = ?",
                        (workload_id,),
                    )
                    if cursor.rowcount:
                        snapshot = self._workload_config_snapshot(conn, workload_id)
                        if snapshot is not None:
                            self._record_workload_revision_in_tx(
                                conn, workload_id, snapshot, None
                            )
                    conn.execute("COMMIT")
                    return
                sets: list[str] = []
                values: list[object] = []
                if max_p95_latency_ms is not None:
                    sets.append("max_p95_latency_ms = ?")
                    values.append(max_p95_latency_ms)
                if max_capability_failure_rate is not None:
                    sets.append("max_capability_failure_rate = ?")
                    values.append(max_capability_failure_rate)
                if max_cost_per_call_usd is not None:
                    sets.append("max_cost_per_call_usd = ?")
                    values.append(max_cost_per_call_usd)
                if not sets:
                    conn.execute("COMMIT")
                    return
                values.append(workload_id)
                self._ensure_initial_workload_revision_in_tx(conn, workload_id)
                cursor = conn.execute(
                    f"UPDATE workloads SET {', '.join(sets)} WHERE id = ?",
                    values,
                )
                if cursor.rowcount:
                    snapshot = self._workload_config_snapshot(conn, workload_id)
                    if snapshot is not None:
                        self._record_workload_revision_in_tx(
                            conn, workload_id, snapshot, None
                        )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def record_workload_revision(
        self,
        workload_id: str,
        config: dict,
        created_by: str | None = None,
    ) -> int:
        """Append a workload config snapshot and return its revision number.

        Low-level primitive: this records HISTORY only and does NOT update the
        live workloads row the router reads. Calling it directly with a config
        that differs from the live row makes current_workload_revision() and the
        router disagree. Prefer set_workload_constraints / set_shadow_config /
        set_workload_policy, which dual-write the live row and a revision.
        """
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                revision = self._record_workload_revision_in_tx(
                    conn, workload_id, config, created_by
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return revision

    def current_workload_revision(self, workload_id: str) -> dict | None:
        """Return the latest recorded workload config snapshot."""
        with self._open() as conn:
            row = conn.execute(
                "SELECT config_json FROM workload_revisions "
                "WHERE workload_id = ? ORDER BY revision DESC LIMIT 1",
                (workload_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set_workload_policy(
        self,
        workload_id: str,
        policy: dict | None,
        created_by: str | None = None,
    ) -> None:
        """Update the live routing policy and append a workload revision."""
        policy = _validate_workload_policy(policy)
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_initial_workload_revision_in_tx(conn, workload_id)
                cursor = conn.execute(
                    "UPDATE workloads SET policy_json = ? WHERE id = ?",
                    (
                        json.dumps(policy, sort_keys=True) if policy else None,
                        workload_id,
                    ),
                )
                if cursor.rowcount:
                    snapshot = self._workload_config_snapshot(conn, workload_id)
                    if snapshot is not None:
                        self._record_workload_revision_in_tx(
                            conn, workload_id, snapshot, created_by
                        )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def workload_revisions(self, workload_id: str) -> list[dict]:
        """Return workload config revision history, oldest first."""
        with self._open() as conn:
            rows = conn.execute(
                "SELECT id, workload_id, revision, config_json, created_at, created_by "
                "FROM workload_revisions WHERE workload_id = ? ORDER BY revision",
                (workload_id,),
            ).fetchall()
        return [self._workload_revision_row(row) for row in rows]

    def workload_revision_diff(
        self,
        workload_id: str,
        rev_a: int,
        rev_b: int,
    ) -> dict:
        """Return a simple per-key old/new diff between two config revisions."""
        with self._open() as conn:
            rows = conn.execute(
                "SELECT revision, config_json FROM workload_revisions "
                "WHERE workload_id = ? AND revision IN (?, ?)",
                (workload_id, rev_a, rev_b),
            ).fetchall()
        configs = {int(row[0]): json.loads(row[1]) for row in rows}
        if rev_a not in configs or rev_b not in configs:
            raise ValueError(
                f"workload {workload_id!r} does not have revisions "
                f"{rev_a!r} and {rev_b!r}"
            )
        old = configs[rev_a]
        new = configs[rev_b]
        return {
            key: {"old": old.get(key), "new": new.get(key)}
            for key in sorted(set(old) | set(new))
            if old.get(key) != new.get(key)
        }

    def rollback_workload(
        self,
        workload_id: str,
        revision: int,
        created_by: str | None = None,
    ) -> int:
        """Re-apply an old config snapshot and append a new revision.

        Rollback is forward-only: the selected snapshot is copied back onto
        the live workloads row, then recorded as a new workload_revisions row,
        like a git revert. Router reads still hit workloads directly.
        """
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT config_json FROM workload_revisions "
                    "WHERE workload_id = ? AND revision = ?",
                    (workload_id, revision),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"workload {workload_id!r} does not have revision {revision!r}"
                    )
                config = json.loads(row[0])
                shadow_config = config.get("shadow_config")
                cursor = conn.execute(
                    "UPDATE workloads SET "
                    "max_p95_latency_ms = ?, "
                    "max_capability_failure_rate = ?, "
                    "max_cost_per_call_usd = ?, "
                    "budget_cap_usd_daily = ?, "
                    "shadow_config_json = ?, "
                    "policy_json = ? "
                    "WHERE id = ?",
                    (
                        config.get("max_p95_latency_ms"),
                        config.get("max_capability_failure_rate"),
                        config.get("max_cost_per_call_usd"),
                        config.get("budget_cap_usd_daily"),
                        json.dumps(shadow_config) if shadow_config is not None else None,
                        json.dumps(config.get("policy"), sort_keys=True)
                        if config.get("policy") is not None
                        else None,
                        workload_id,
                    ),
                )
                if not cursor.rowcount:
                    raise ValueError(f"unknown workload {workload_id!r}")
                snapshot = self._workload_config_snapshot(conn, workload_id)
                if snapshot is None:
                    raise ValueError(f"unknown workload {workload_id!r}")
                new_revision = self._record_workload_revision_in_tx(
                    conn, workload_id, snapshot, created_by
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return new_revision

    # Shadow-eval config ------------------------------------------------------

    def write_sample(self, call_id: str, prompt_body: str, response_body: str) -> None:
        """Store one call's bodies for online-eval grading.

        Only ever written for workloads that opted into shadow eval (the
        documented consent for body storage) — see SommLLM's capture
        hook. INSERT OR IGNORE: a call is sampled at most once.

        FK enforcement is disabled for this one connection: the calls row
        is written by the ASYNC WriterQueue and usually hasn't landed yet
        when capture fires. A momentarily-orphaned sample is harmless —
        every reader joins through calls — and heals when the batch (or a
        spool drain) lands."""
        with self._open() as conn:
            foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO samples (call_id, prompt_body, response_body) "
                    "VALUES (?, ?, ?)",
                    (call_id, prompt_body, response_body),
                )
            finally:
                if foreign_keys:
                    conn.execute("PRAGMA foreign_keys = ON")

    def set_shadow_config(self, workload_id: str, config: dict | None) -> None:
        """Attach (or clear) shadow-eval config on the live workload row.

        config = None → shadow disabled.
        config = {"gold_provider": ..., "gold_model": ..., "sample_rate": ...,
                  "budget_usd_daily": ...} → enabled.

        Dual-write note: workloads.shadow_config_json is the hot read path.
        workload_revisions receives an append-only post-update snapshot.
        """
        with self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_initial_workload_revision_in_tx(conn, workload_id)
                cursor = conn.execute(
                    "UPDATE workloads SET shadow_config_json = ? WHERE id = ?",
                    (json.dumps(config) if config else None, workload_id),
                )
                if cursor.rowcount:
                    snapshot = self._workload_config_snapshot(conn, workload_id)
                    if snapshot is not None:
                        self._record_workload_revision_in_tx(
                            conn, workload_id, snapshot, None
                        )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

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
            parent_prompt_id=None,
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
                    outcome, error_kind, error_detail, prompt_hash, response_hash,
                    correlation_id, temperature, max_tokens, top_p, stop_sequences_json,
                    ttft_ms, session_id, parent_call_id, cache_tokens_in,
                    cache_tokens_out, citations_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
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
                    call.correlation_id,
                    call.temperature,
                    call.max_tokens,
                    call.top_p,
                    call.stop_sequences_json,
                    call.ttft_ms,
                    call.session_id,
                    call.parent_call_id,
                    call.cache_tokens_in,
                    call.cache_tokens_out,
                    call.citations_json,
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
                        outcome, error_kind, error_detail, prompt_hash, response_hash,
                        correlation_id, temperature, max_tokens, top_p, stop_sequences_json,
                        ttft_ms, session_id, parent_call_id, cache_tokens_in,
                        cache_tokens_out, citations_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?
                    )
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
                            c.correlation_id,
                            c.temperature,
                            c.max_tokens,
                            c.top_p,
                            c.stop_sequences_json,
                            c.ttft_ms,
                            c.session_id,
                            c.parent_call_id,
                            c.cache_tokens_in,
                            c.cache_tokens_out,
                            c.citations_json,
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
                "prompt_hash, response_hash, error_detail, correlation_id, "
                "temperature, max_tokens, top_p, stop_sequences_json, ttft_ms, "
                "session_id, parent_call_id, cache_tokens_in, cache_tokens_out, "
                "citations_json FROM calls WHERE id = ?",
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
            correlation_id=row[16],
            temperature=row[17],
            max_tokens=row[18],
            top_p=row[19],
            stop_sequences_json=row[20],
            ttft_ms=row[21],
            session_id=row[22],
            parent_call_id=row[23],
            cache_tokens_in=row[24],
            cache_tokens_out=row[25],
            citations_json=row[26],
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

    def workload_frontier(
        self,
        workload_id: str,
        since_days: int = 30,
    ) -> list[dict]:
        """Per-(provider, model) adequacy rollup for a workload.

        Returns one row per (provider, model) the workload has touched in the
        window, with capability vs. detractor failures kept separate so a
        flaky free-tier provider doesn't get scored as a model that can't do
        the work. Used by ``somm frontier`` and the admin UI.

        Each row has:

        * ``n_calls``, ``n_ok`` — call counts
        * ``n_capability_failures`` (Tier 2/3: model's fault) and
          ``capability_failure_rate`` (failures / n_calls)
        * ``n_detractors`` (Tier 2 detractor: provider/network) and
          ``detractor_rate``
        * ``p50_latency_ms``, ``p95_latency_ms`` — across ok calls only
        * ``mean_cost_per_ok_call``, ``total_cost_usd``
        * ``fitness`` — dict of bool|None flags vs. workload constraints
          (None = constraint unset; True = constraint *exceeded*)

        Sorted by capability_failure_rate ascending, then mean_cost_per_ok_call,
        so the candidate that's cheapest among the model-fit options floats up.
        """
        with self._open() as conn:
            wl_row = conn.execute(
                "SELECT max_p95_latency_ms, max_capability_failure_rate, "
                "max_cost_per_call_usd FROM workloads WHERE id = ?",
                (workload_id,),
            ).fetchone()
            if wl_row is None:
                return []
            constraints = {
                "max_p95_latency_ms": wl_row[0],
                "max_capability_failure_rate": wl_row[1],
                "max_cost_per_call_usd": wl_row[2],
            }
            rows = conn.execute(
                """
                SELECT
                    provider, model,
                    COUNT(*) AS n_calls,
                    SUM(CASE WHEN outcome = 'ok' THEN 1 ELSE 0 END) AS n_ok,
                    SUM(is_capability_signal) AS n_capability_failures,
                    SUM(is_detractor) AS n_detractors,
                    AVG(CASE WHEN outcome = 'ok' THEN cost_usd END) AS mean_cost_per_ok_call,
                    SUM(cost_usd) AS total_cost_usd,
                    GROUP_CONCAT(
                        CASE WHEN outcome = 'ok' THEN latency_ms END
                    ) AS ok_latencies_csv
                FROM v_calls_classified
                WHERE workload_id = ?
                  AND ts >= datetime('now', ?)
                GROUP BY provider, model
                """,
                (workload_id, f"-{since_days} days"),
            ).fetchall()

        out: list[dict] = []
        for r in rows:
            n_calls = r[2]
            n_ok = r[3] or 0
            n_cap = r[4] or 0
            n_det = r[5] or 0
            cap_rate = (n_cap / n_calls) if n_calls else 0.0
            det_rate = (n_det / n_calls) if n_calls else 0.0
            p50, p95 = _percentiles(r[8])
            mean_cost = r[6]
            fitness = {
                "exceeds_max_p95_latency_ms": (
                    None
                    if constraints["max_p95_latency_ms"] is None or p95 is None
                    else p95 > constraints["max_p95_latency_ms"]
                ),
                "exceeds_max_capability_failure_rate": (
                    None
                    if constraints["max_capability_failure_rate"] is None
                    else cap_rate > constraints["max_capability_failure_rate"]
                ),
                "exceeds_max_cost_per_call_usd": (
                    None
                    if constraints["max_cost_per_call_usd"] is None or mean_cost is None
                    else mean_cost > constraints["max_cost_per_call_usd"]
                ),
            }
            out.append(
                {
                    "provider": r[0],
                    "model": r[1],
                    "n_calls": n_calls,
                    "n_ok": n_ok,
                    "n_capability_failures": n_cap,
                    "n_detractors": n_det,
                    "capability_failure_rate": cap_rate,
                    "detractor_rate": det_rate,
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "mean_cost_per_ok_call": mean_cost,
                    "total_cost_usd": r[7],
                    "fitness": fitness,
                }
            )
        out.sort(
            key=lambda x: (
                x["capability_failure_rate"],
                x["mean_cost_per_ok_call"] if x["mean_cost_per_ok_call"] is not None else float("inf"),
            )
        )
        return out

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

    # Learned parameter overrides (self-healing) ------------------------------

    def lookup_learned_override(
        self, workload_id: str, model: str, provider: str | None = None
    ) -> dict | None:
        """Hot-path read: the learned param override for a (workload, model).

        Keyed by (workload, provider, model) but matched on (workload, model)
        with the requested provider preferred — so it applies whether or not
        the caller pinned a provider. Returns None when there's nothing learned.
        Callers MUST treat this as best-effort (wrap in try/except): a lookup
        failure must never break a live call.
        """
        with self._open() as conn:
            row = conn.execute(
                "SELECT max_tokens_floor, failure_signature, confidence, provider, model "
                "FROM learned_param_overrides "
                "WHERE workload_id = ? AND model = ? "
                "ORDER BY (CASE WHEN provider = ? THEN 0 ELSE 1 END), confidence DESC "
                "LIMIT 1",
                (workload_id, model, provider or ""),
            ).fetchone()
        if row is None:
            return None
        return {
            "max_tokens_floor": row[0],
            "failure_signature": row[1],
            "confidence": row[2],
            "provider": row[3],
            "model": row[4],
        }

    def upsert_learned_override(
        self,
        *,
        workload_id: str,
        provider: str,
        model: str,
        max_tokens_floor: int | None,
        failure_signature: str,
        evidence: dict | None = None,
        confidence: float = 0.0,
    ) -> None:
        """Write (or refresh) a learned override for a (workload, provider, model)."""
        with self._open() as conn:
            conn.execute(
                "INSERT INTO learned_param_overrides "
                "(workload_id, provider, model, max_tokens_floor, failure_signature, "
                " evidence_json, confidence, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(workload_id, provider, model) DO UPDATE SET "
                "  max_tokens_floor = excluded.max_tokens_floor, "
                "  failure_signature = excluded.failure_signature, "
                "  evidence_json = excluded.evidence_json, "
                "  confidence = excluded.confidence, "
                "  updated_at = CURRENT_TIMESTAMP",
                (
                    workload_id,
                    provider,
                    model,
                    max_tokens_floor,
                    failure_signature,
                    json.dumps(evidence or {}),
                    confidence,
                ),
            )

    def detect_capability_overflow(
        self,
        project: str | None = None,
        *,
        window_days: int = 7,
        min_calls: int = 8,
        empty_rate_threshold: float = 0.4,
        min_stripped: int = 3,
    ) -> list[dict]:
        """Find (workload, provider, model) triples that keep returning empty
        because the model exhausts its output budget on reasoning tokens before
        answering — the ``stripped_empty`` signature. These are candidates for an
        automatic max_tokens bump.

        Returns one dict per candidate with a recommended ``max_tokens_floor``.
        """
        with self._open() as conn:
            rows = conn.execute(
                """
                SELECT c.workload_id,
                       COALESCE(w.name, '(unregistered)') AS workload_name,
                       c.provider, c.model,
                       COUNT(*) AS n_calls,
                       SUM(CASE WHEN c.outcome = 'empty' THEN 1 ELSE 0 END) AS n_empty,
                       SUM(CASE WHEN c.outcome = 'empty'
                                 AND c.error_detail LIKE '%stripped_empty%'
                                THEN 1 ELSE 0 END) AS n_stripped,
                       MAX(c.max_tokens) AS max_tokens_req
                FROM calls c
                LEFT JOIN workloads w ON w.id = c.workload_id
                WHERE (? IS NULL OR c.project = ?)
                  AND c.ts >= datetime('now', ?)
                  AND c.workload_id IS NOT NULL
                GROUP BY c.workload_id, c.provider, c.model
                HAVING n_calls >= ? AND n_stripped >= ?
                """,
                (project, project, f"-{window_days} days", min_calls, min_stripped),
            ).fetchall()

        out: list[dict] = []
        for wl_id, name, provider, model, n_calls, n_empty, n_stripped, max_req in rows:
            empty_rate = (n_empty or 0) / n_calls if n_calls else 0.0
            if empty_rate < empty_rate_threshold:
                continue
            base = int(max_req) if max_req else 4096
            # Give the model materially more room; clamp to a sane ceiling so a
            # runaway pattern can't request an absurd budget.
            recommended_floor = min(max(base * 2, 8192), 32768)
            # confidence scales with how dominant + how well-sampled the pattern is
            confidence = round(min(0.95, 0.5 + empty_rate * 0.3 + min(n_calls, 50) / 200), 2)
            out.append(
                {
                    "workload_id": wl_id,
                    "workload_name": name,
                    "provider": provider,
                    "model": model,
                    "n_calls": n_calls,
                    "n_empty": n_empty,
                    "n_stripped": n_stripped,
                    "empty_rate": round(empty_rate, 3),
                    "max_tokens_req": base,
                    "recommended_max_tokens_floor": recommended_floor,
                    "confidence": confidence,
                }
            )
        return out


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

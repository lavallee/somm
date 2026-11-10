"""Schema migration runner.

Keeps `schema_version` up-to-date. Idempotent: safe to call on every start.
Migrations are versioned SQL files under `migrations/NNNN_<name>.sql`.
"""

from __future__ import annotations

import re
import sqlite3
import time
from importlib import resources
from pathlib import Path

from somm_core.version import SCHEMA_VERSION


def _migrations_root() -> Path:
    # Works for installed wheels and editable installs.
    return Path(str(resources.files("somm_core"))) / "migrations"


def _list_migrations() -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    root = _migrations_root()
    if not root.exists():
        return out
    for p in sorted(root.glob("*.sql")):
        stem = p.stem
        num = int(stem.split("_", 1)[0])
        out.append((num, p))
    return out


def _split_statements(sql: str) -> list[str]:
    """Split a migration file into individual statements.

    Migrations are plain additive DDL (CREATE/ALTER/INDEX) with no triggers or
    string literals — so stripping ``--`` line comments (which may themselves
    contain ';') and then splitting on ';' yields correct statements, letting us
    run each under our own transaction control. A migration that ever needs
    richer SQL (a trigger body, a string literal with ';') must revisit this.
    """
    # Strip `--` comments to end-of-line FIRST: a comment like
    # "OFF by default; per-workload opt-in." contains a ';' that would
    # otherwise mis-split the following statement.
    no_comments = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
    return [stmt.strip() for stmt in no_comments.split(";") if stmt.strip()]


def current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied schema_version, or 0 if the table doesn't exist."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    if cur.fetchone() is None:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def _already_applied(version: int, conn: sqlite3.Connection) -> bool:
    """Recognize a migration whose DDL landed but whose version stamp did not.

    The pre-0.8 migration runner committed DDL and the schema_version row
    separately, so a crash between the two left databases that look older than
    they are. Re-running additive DDL fails loudly on a duplicate column, which
    is the right outcome; a DROP has no such guard, so the one destructive
    migration gets an explicit check (issue #44).
    """
    if version == 10:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
        return "commission_id" not in columns
    return False


def _relax_cost_usd_not_null(conn: sqlite3.Connection) -> None:
    """Migration 0029: drop NOT NULL from ``calls.cost_usd`` without a rebuild.

    SQLite's documented procedure for removing a NOT NULL constraint: edit
    the stored CREATE TABLE text and bump ``schema_version`` so every open
    connection re-reads it. Stored rows are untouched. Runs inside the
    migration's transaction, so a failure rolls back with it.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'calls'"
    ).fetchone()
    if row is None:
        return
    relaxed = re.sub(
        r"(\bcost_usd\s+REAL)\s+NOT\s+NULL", r"\1", row[0], count=1, flags=re.IGNORECASE
    )
    if relaxed == row[0]:
        return
    defensive = conn.getconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE)
    if defensive:
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, False)
    cookie = conn.execute("PRAGMA schema_version").fetchone()[0]
    conn.execute("PRAGMA writable_schema = ON")
    try:
        conn.execute(
            "UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = 'calls'",
            (relaxed,),
        )
        conn.execute(f"PRAGMA schema_version = {int(cookie) + 1}")
    finally:
        conn.execute("PRAGMA writable_schema = OFF")
        if defensive:
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)


# Migrations whose step SQL cannot express; run after the file's statements.
_PYTHON_STEPS = {29: _relax_cost_usd_not_null}


def _execute_pragma(conn: sqlite3.Connection, stmt: str, *, wait_s: float = 10.0) -> None:
    """Run a migration PRAGMA, waiting out a peer that holds the database.

    Switching ``journal_mode`` needs an exclusive lock and SQLite does not
    consult the busy handler for it, so with a second process mid-migration it
    fails at once with "database is locked" whatever ``busy_timeout`` says.
    The pragmas only set durable database properties, so retrying is safe.
    """
    deadline = time.monotonic() + wait_s
    delay = 0.01
    while True:
        try:
            conn.execute(stmt)
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations. Returns the schema version after running.

    Applied inside a transaction per-file. Called automatically on first library
    use so zero-config `somm.llm()` works; re-running any somm command also
    opens the repository and applies pending migrations.
    """
    current = current_schema_version(conn)
    applied = current
    pending = [(v, p) for v, p in _list_migrations() if v > current]

    # Apply each migration's DDL AND its schema_version stamp in ONE
    # transaction. Otherwise (the connection is autocommit, isolation_level=
    # None) a separate version INSERT could commit after the DDL — a crash in
    # that window would leave a DB with the new columns but the old version,
    # wedging the next run on a duplicate ADD COLUMN. We drive the transaction
    # with explicit BEGIN/COMMIT, which needs autocommit semantics; force them
    # for the duration so this works whether the caller's connection is
    # autocommit (production) or default-isolation (some tests). Migration
    # files therefore carry no transaction control of their own.
    if pending:
        # Drive the transaction ourselves via conn.execute (NOT executescript,
        # whose implicit commit-first behavior varies with isolation mode and
        # can't wrap the version stamp atomically). Force autocommit for the
        # duration so our explicit BEGIN/COMMIT are the only transaction
        # control, whatever mode the caller's connection is in.
        prev_isolation = conn.isolation_level
        if conn.in_transaction:
            conn.commit()
        conn.isolation_level = None
        try:
            for version, path in pending:
                statements = _split_statements(path.read_text())
                if _already_applied(version, conn):
                    # Stamp only; the DDL is already in place.
                    statements = []
                # PRAGMA statements (e.g. journal_mode = WAL) cannot run inside
                # a transaction; run them first, outside BEGIN. They set durable
                # DB properties, so re-running a rolled-back migration that only
                # got as far as its pragmas is harmless.
                pragmas = [s for s in statements if s.lstrip().upper().startswith("PRAGMA")]
                ddl = [s for s in statements if not s.lstrip().upper().startswith("PRAGMA")]
                # Another process may be migrating the same fresh ledger (several
                # clients constructed at once). BEGIN IMMEDIATE takes the write
                # lock up front, where busy_timeout applies; a deferred BEGIN
                # would fail at once with "database is locked" when two readers
                # both try to upgrade. Once we hold the lock, re-read the version:
                # if a peer applied this migration while we waited, skip it rather
                # than re-run its DDL or collide on the schema_version stamp.
                if current_schema_version(conn) >= version:
                    applied = version
                    continue
                for stmt in pragmas:
                    _execute_pragma(conn, stmt)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    if current_schema_version(conn) >= version:
                        conn.execute("ROLLBACK")
                        applied = version
                        continue
                    for stmt in ddl:
                        conn.execute(stmt)
                    if version in _PYTHON_STEPS:
                        _PYTHON_STEPS[version](conn)
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (int(version),),
                    )
                    conn.execute("COMMIT")
                    applied = version
                except Exception:
                    conn.rollback()
                    raise
        finally:
            conn.isolation_level = prev_isolation
    if applied < SCHEMA_VERSION:
        # Library compiled against a newer schema than we have migrations for.
        raise RuntimeError(
            f"Internal: SCHEMA_VERSION={SCHEMA_VERSION} but "
            f"highest migration on disk is {applied}. Missing migration file."
        )
    return applied

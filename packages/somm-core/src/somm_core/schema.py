"""Schema migration runner.

Keeps `schema_version` up-to-date. Idempotent: safe to call on every start.
Migrations are versioned SQL files under `migrations/NNNN_<name>.sql`.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from somm_core.version import SCHEMA_VERSION


class SchemaStale(RuntimeError):
    """Raised when the DB is on an older schema than this library expects."""

    def __init__(self, db_version: int, expected: int, path: str) -> None:
        self.db_version = db_version
        self.expected = expected
        self.path = path
        super().__init__(
            f"SOMM_SCHEMA_STALE\n\n"
            f"Problem: {path} is schema v{db_version}, but this somm version requires v{expected}.\n"
            f"Cause: an older somm process is still using the database, or this install is missing migrations.\n"
            f"Fix:\n"
            f"  1. Upgrade all somm packages used by this project.\n"
            f"  2. Stop any long-running processes or daemons that still import the old version.\n"
            f"  3. Re-run any somm command, for example: somm doctor --project <project>\n"
            f"  4. Restart your own daemons after that command succeeds.\n"
            f"Docs: docs/errors/SOMM_SCHEMA_STALE.md"
        )


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
                # PRAGMA statements (e.g. journal_mode = WAL) cannot run inside
                # a transaction; run them first, outside BEGIN. They set durable
                # DB properties, so re-running a rolled-back migration that only
                # got as far as its pragmas is harmless.
                pragmas = [s for s in statements if s.lstrip().upper().startswith("PRAGMA")]
                ddl = [s for s in statements if not s.lstrip().upper().startswith("PRAGMA")]
                for stmt in pragmas:
                    conn.execute(stmt)
                try:
                    conn.execute("BEGIN")
                    for stmt in ddl:
                        conn.execute(stmt)
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

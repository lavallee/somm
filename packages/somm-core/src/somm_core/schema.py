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
            f"Cause: somm was upgraded and migrations have not been applied.\n"
            f"Fix:\n"
            f"  somm service stop\n"
            f"  somm migrate --check\n"
            f"  somm migrate\n"
            f"  somm service start\n"
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
    use so zero-config `somm.llm()` works. For explicit control, run `somm migrate`.
    """
    current = current_schema_version(conn)
    applied = current
    for version, path in _list_migrations():
        if version <= current:
            continue
        sql = path.read_text()
        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (version,),
            )
            conn.commit()
            applied = version
        except Exception:
            conn.rollback()
            raise
    if applied < SCHEMA_VERSION:
        # Library compiled against a newer schema than we have migrations for.
        raise RuntimeError(
            f"Internal: SCHEMA_VERSION={SCHEMA_VERSION} but "
            f"highest migration on disk is {applied}. Missing migration file."
        )
    return applied

"""Best-effort worker heartbeat writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository


def beat(repo: Repository, worker_name: str, ok: bool | None) -> None:
    """Record that a worker ran.

    Heartbeats are observability only: failures here must never break the
    scheduler or manual admin commands.
    """
    ok_true = 1 if ok is True else 0
    ok_false = 1 if ok is False else 0
    try:
        with repo._open() as conn:
            conn.execute(
                "INSERT INTO worker_heartbeat "
                "(worker_name, last_run_at, last_success_at, consecutive_failures) "
                "VALUES ("
                "  ?, "
                "  CURRENT_TIMESTAMP, "
                "  CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END, "
                "  CASE WHEN ? THEN 0 WHEN ? THEN 1 ELSE 0 END"
                ") "
                "ON CONFLICT(worker_name) DO UPDATE SET "
                "  last_run_at = CURRENT_TIMESTAMP, "
                "  last_success_at = CASE "
                "    WHEN ? THEN CURRENT_TIMESTAMP "
                "    ELSE worker_heartbeat.last_success_at "
                "  END, "
                "  consecutive_failures = CASE "
                "    WHEN ? THEN 0 "
                "    WHEN ? THEN worker_heartbeat.consecutive_failures + 1 "
                "    ELSE worker_heartbeat.consecutive_failures "
                "  END",
                (
                    worker_name,
                    ok_true,
                    ok_true,
                    ok_false,
                    ok_true,
                    ok_true,
                    ok_false,
                ),
            )
    except Exception:
        pass

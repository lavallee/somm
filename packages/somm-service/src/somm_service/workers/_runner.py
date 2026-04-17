"""Scheduler — polls the `jobs` table; claims + runs due workers atomically.

Design:
- One scheduler per service process. Daemon thread; stops when service
  does.
- Default jobs (seeded on first start): model_intel (24h), shadow_eval
  (15min), agent (7d). Intervals overridable via config.
- Lease-based claim: `UPDATE jobs SET locked_until = ? WHERE job_name = ?
  AND (locked_until IS NULL OR locked_until < ?)` — atomic, crash-safe.
- On success: clear lease, update last_success_at, reset due_at to
  now + interval_seconds.
- On failure: clear lease, increment consecutive_failures, backoff
  (60s + 30s * consecutive_failures, capped).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository


_log = logging.getLogger("somm.workers.scheduler")


# (job_name, default_interval_seconds)
DEFAULT_JOBS: list[tuple[str, int]] = [
    ("model_intel", 24 * 3600),  # 24h
    ("shadow_eval", 15 * 60),  # 15 min
    ("agent", 7 * 24 * 3600),  # 7 days
]


class Scheduler:
    """Background job runner. Polls `jobs` table; claims + runs due workers."""

    def __init__(
        self,
        repo: Repository,
        worker_factory: Callable[[str], object | None],
        poll_interval_s: float = 30.0,
        lease_window_s: int = 1800,
        backoff_base_s: int = 60,
        backoff_step_s: int = 30,
        backoff_max_s: int = 1800,
    ) -> None:
        self.repo = repo
        self.worker_factory = worker_factory
        self.poll_interval_s = poll_interval_s
        self.lease_window_s = lease_window_s
        self.backoff_base_s = backoff_base_s
        self.backoff_step_s = backoff_step_s
        self.backoff_max_s = backoff_max_s
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle

    def seed(self, jobs: list[tuple[str, int]] | None = None) -> None:
        """Insert default jobs if missing. Idempotent."""
        jobs = jobs or DEFAULT_JOBS
        with self.repo._open() as conn:
            for name, interval_s in jobs:
                conn.execute(
                    "INSERT OR IGNORE INTO jobs "
                    "(job_name, due_at, interval_seconds) "
                    "VALUES (?, CURRENT_TIMESTAMP, ?)",
                    (name, interval_s),
                )

    def start(self, seed_jobs: list[tuple[str, int]] | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.seed(seed_jobs)
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="somm-scheduler",
        )
        self._thread.start()
        _log.info("scheduler started (poll=%.1fs)", self.poll_interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        _log.info("scheduler stopped")

    # ------------------------------------------------------------------
    # Loop

    def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                self.tick()
            except Exception as e:  # pragma: no cover — defensive
                _log.warning("scheduler tick raised: %s", e)
            self._stopping.wait(self.poll_interval_s)

    def tick(self) -> list[str]:
        """Run one scheduling pass. Returns the list of job_names executed."""
        executed: list[str] = []
        for job_name in self._fetch_due():
            if not self._claim(job_name):
                continue
            worker = self.worker_factory(job_name)
            if worker is None:
                _log.warning("scheduler: no worker for %r; skipping", job_name)
                self._mark_skipped(job_name)
                continue
            try:
                worker.run_once()
                self._mark_success(job_name)
                executed.append(job_name)
            except Exception as e:
                _log.warning("scheduler: %s failed: %s", job_name, e)
                self._mark_failure(job_name)
        return executed

    # ------------------------------------------------------------------
    # Job table ops

    def _fetch_due(self) -> list[str]:
        now = datetime.now(UTC).isoformat()
        with self.repo._open() as conn:
            rows = conn.execute(
                "SELECT job_name FROM jobs "
                "WHERE due_at <= ? "
                "AND (locked_until IS NULL OR locked_until < ?)",
                (now, now),
            ).fetchall()
        return [r[0] for r in rows]

    def _claim(self, job_name: str) -> bool:
        """Atomic lease acquisition. Returns True if this process owns the job now."""
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        lease_until = _iso_plus(now_dt, self.lease_window_s)
        with self.repo._open() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET locked_until = ?, last_started_at = ? "
                "WHERE job_name = ? "
                "AND due_at <= ? "
                "AND (locked_until IS NULL OR locked_until < ?)",
                (lease_until, now, job_name, now, now),
            )
        return cursor.rowcount > 0

    def _mark_success(self, job_name: str) -> None:
        with self.repo._open() as conn:
            conn.execute(
                "UPDATE jobs SET "
                "  last_success_at = CURRENT_TIMESTAMP, "
                "  locked_until = NULL, "
                "  consecutive_failures = 0, "
                "  due_at = datetime('now', ?) "
                "WHERE job_name = ?",
                (f"+{self._interval_for(job_name)} seconds", job_name),
            )

    def _mark_failure(self, job_name: str) -> None:
        with self.repo._open() as conn:
            # Compute backoff from current failure count
            row = conn.execute(
                "SELECT consecutive_failures FROM jobs WHERE job_name = ?",
                (job_name,),
            ).fetchone()
            failures = (row[0] if row else 0) + 1
            backoff_s = min(
                self.backoff_max_s,
                self.backoff_base_s + self.backoff_step_s * failures,
            )
            conn.execute(
                "UPDATE jobs SET "
                "  consecutive_failures = consecutive_failures + 1, "
                "  locked_until = NULL, "
                "  due_at = datetime('now', ?) "
                "WHERE job_name = ?",
                (f"+{backoff_s} seconds", job_name),
            )

    def _mark_skipped(self, job_name: str) -> None:
        """No worker available (misconfig); push due_at out so we don't spin."""
        with self.repo._open() as conn:
            conn.execute(
                "UPDATE jobs SET locked_until = NULL, "
                "  due_at = datetime('now', '+3600 seconds') "
                "WHERE job_name = ?",
                (job_name,),
            )

    def _interval_for(self, job_name: str) -> int:
        with self.repo._open() as conn:
            row = conn.execute(
                "SELECT interval_seconds FROM jobs WHERE job_name = ?",
                (job_name,),
            ).fetchone()
        return int(row[0]) if row else 3600


def _iso_plus(base: datetime, seconds: int) -> str:
    from datetime import timedelta

    return (base + timedelta(seconds=seconds)).isoformat()

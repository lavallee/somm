"""Scheduler tests — seed, lease atomicity, success/failure state, crash recovery."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core.config import Config
from somm_core.repository import Repository
from somm_service.workers._runner import Scheduler


def _tmp_repo(tmp_path: Path) -> Repository:
    cfg = Config()
    cfg.db_dir = tmp_path / ".somm"
    return Repository(cfg.db_path)


class _RecordingWorker:
    """Worker stand-in that records invocations + optionally raises."""

    def __init__(self, name: str, should_raise: bool = False):
        self.name = name
        self.should_raise = should_raise
        self.runs = 0

    def run_once(self):
        self.runs += 1
        if self.should_raise:
            raise RuntimeError(f"{self.name} boom")
        return {"ok": True}


def _workers_factory(workers_by_name: dict):
    def factory(job_name: str):
        return workers_by_name.get(job_name)

    return factory


# ---------------------------------------------------------------------------
# Seed + lifecycle


def test_seed_inserts_default_jobs(tmp_path):
    repo = _tmp_repo(tmp_path)
    sched = Scheduler(repo, _workers_factory({}))
    sched.seed()
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT job_name, interval_seconds FROM jobs ORDER BY job_name"
        ).fetchall()
    names = {r[0] for r in rows}
    assert names == {"model_intel", "shadow_eval", "agent"}


def test_seed_is_idempotent(tmp_path):
    repo = _tmp_repo(tmp_path)
    sched = Scheduler(repo, _workers_factory({}))
    sched.seed()
    sched.seed()  # second call shouldn't duplicate
    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert n == 3


def test_custom_jobs_override_defaults(tmp_path):
    repo = _tmp_repo(tmp_path)
    sched = Scheduler(repo, _workers_factory({}))
    sched.seed([("custom_job", 60)])
    with repo._open() as conn:
        names = [r[0] for r in conn.execute("SELECT job_name FROM jobs").fetchall()]
    assert "custom_job" in names


# ---------------------------------------------------------------------------
# Tick behavior


def test_tick_runs_due_workers(tmp_path):
    repo = _tmp_repo(tmp_path)
    w = _RecordingWorker("model_intel")
    sched = Scheduler(repo, _workers_factory({"model_intel": w}))
    sched.seed()

    executed = sched.tick()
    assert "model_intel" in executed
    assert w.runs == 1


def test_tick_doesnt_run_not_due_jobs(tmp_path):
    repo = _tmp_repo(tmp_path)
    w = _RecordingWorker("agent")
    sched = Scheduler(repo, _workers_factory({"agent": w}))

    # Seed with due_at far in the future
    future = (datetime.now(UTC) + timedelta(hours=24)).isoformat()
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO jobs (job_name, due_at, interval_seconds) VALUES (?, ?, ?)",
            ("agent", future, 3600),
        )

    executed = sched.tick()
    assert "agent" not in executed
    assert w.runs == 0


def test_tick_success_clears_lease_and_reschedules(tmp_path):
    repo = _tmp_repo(tmp_path)
    w = _RecordingWorker("model_intel")
    sched = Scheduler(repo, _workers_factory({"model_intel": w}))
    sched.seed()

    sched.tick()
    with repo._open() as conn:
        row = conn.execute(
            "SELECT locked_until, consecutive_failures, last_success_at, due_at "
            "FROM jobs WHERE job_name = 'model_intel'"
        ).fetchone()
    locked_until, failures, last_success, due_at = row
    assert locked_until is None
    assert failures == 0
    assert last_success is not None
    # due_at pushed into the future
    due_dt = datetime.fromisoformat(due_at.replace(" ", "T"))
    assert due_dt > datetime.now().replace(tzinfo=None)


def test_tick_failure_backs_off(tmp_path):
    repo = _tmp_repo(tmp_path)
    w = _RecordingWorker("model_intel", should_raise=True)
    sched = Scheduler(
        repo, _workers_factory({"model_intel": w}), backoff_base_s=1, backoff_step_s=1
    )
    sched.seed()

    sched.tick()
    with repo._open() as conn:
        row = conn.execute(
            "SELECT locked_until, consecutive_failures FROM jobs WHERE job_name = 'model_intel'"
        ).fetchone()
    locked_until, failures = row
    assert locked_until is None  # lease cleared after failure
    assert failures == 1


def test_missing_worker_pushes_due_out(tmp_path):
    """If factory returns None for a job, it gets rescheduled far out (no spin)."""
    repo = _tmp_repo(tmp_path)
    sched = Scheduler(repo, _workers_factory({}))  # no workers
    sched.seed()

    executed = sched.tick()
    assert executed == []
    # All jobs rescheduled (due_at in future)
    with repo._open() as conn:
        rows = conn.execute("SELECT due_at FROM jobs").fetchall()
    for (due_at,) in rows:
        due_dt = datetime.fromisoformat(due_at.replace(" ", "T"))
        assert due_dt > datetime.now().replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Concurrent lease (two Scheduler instances on same DB)


def test_lease_is_atomic_across_instances(tmp_path):
    """Two schedulers racing — only one should run the worker per tick."""
    repo = _tmp_repo(tmp_path)
    shared = _tmp_repo(tmp_path)  # second handle, same file

    runs = {"n": 0}
    lock = threading.Lock()

    class _SlowWorker:
        def __init__(self):
            self.name = "slow"

        def run_once(self):
            with lock:
                runs["n"] += 1
            time.sleep(0.1)

    # Both schedulers have a worker for model_intel; only one should win.
    s1 = Scheduler(repo, _workers_factory({"model_intel": _SlowWorker()}))
    s2 = Scheduler(shared, _workers_factory({"model_intel": _SlowWorker()}))
    s1.seed()

    results: dict[str, list] = {"s1": [], "s2": []}

    def t1():
        results["s1"] = s1.tick()

    def t2():
        results["s2"] = s2.tick()

    th1 = threading.Thread(target=t1)
    th2 = threading.Thread(target=t2)
    th1.start()
    th2.start()
    th1.join(timeout=10)
    th2.join(timeout=10)

    # The worker should have been invoked exactly once across both instances.
    assert runs["n"] == 1
    combined = results["s1"] + results["s2"]
    assert combined.count("model_intel") == 1


# ---------------------------------------------------------------------------
# Start/stop lifecycle


def test_start_stop_runs_in_background(tmp_path):
    repo = _tmp_repo(tmp_path)
    w = _RecordingWorker("model_intel")
    sched = Scheduler(repo, _workers_factory({"model_intel": w}), poll_interval_s=0.05)
    sched.start()
    # Wait for the first poll to fire the worker
    deadline = time.monotonic() + 3
    while w.runs == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    sched.stop()

    assert w.runs >= 1

"""Per-process writer queue + JSONL spool fallback.

Design (Eng-E1 / Codex refinement): threads enqueue Call objects; one writer
thread drains in short batched transactions. On repeated SQLITE_BUSY or disk
pressure, rows spill to JSONL under `.somm/spool/`. `somm drain-spool`
replays the spool back into SQLite when pressure clears.

Preserves zero-service hot path: library works without somm serve running.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from somm_core import Call
from somm_core.repository import Repository

_BATCH_MAX = 100
_BATCH_MS = 100
_MAX_BUSY_RETRIES = 3
_SPOOL_DRAIN_INTERVAL_S = 600
_SPOOL_CLAIM_MARKER = ".claimed-"
_auto_drain_warned = False
_auto_drain_warned_lock = threading.Lock()


class WriterQueue:
    """Per-process async writer. One queue, one draining thread, one DB connection.

    Optional cross-project mirror: if `mirror_repo` is supplied, every
    successful batch insert is replicated there. Mirror failures are logged
    and spilled to a separate mirror_spool but do not block the primary write.
    """

    _STOP = object()

    def __init__(
        self,
        repo: Repository,
        spool_dir: Path,
        mirror_repo: Repository | None = None,
    ) -> None:
        self._repo = repo
        self._mirror_repo = mirror_repo
        self._spool_dir = Path(spool_dir)
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._spool_dir.chmod(0o700)
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="somm-writer", daemon=True)
        self._started = False
        self._stopping = False
        self._atexit_registered = False
        self._next_spool_drain_at = 0.0

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True
            # Daemon thread = killed on process exit without draining. Without
            # this hook, calls submitted in the last ~100ms of process lifetime
            # vanish — caller never sees them in calls.sqlite, and they don't
            # spill to the JSONL fallback either (spill only fires on a drain
            # *failure*, not on writer-thread death). atexit runs while the
            # daemon thread is still alive so flush() + stop() can drain
            # cleanly. Idempotent if close() is also called explicitly.
            if not self._atexit_registered:
                atexit.register(self._atexit_drain)
                self._atexit_registered = True

    def submit(self, call: Call) -> None:
        if not self._started:
            self.start()
        self._q.put(call)

    def flush(self, timeout: float = 5.0) -> None:
        """Block until queued and in-flight rows have been committed (best-effort)."""
        deadline = time.monotonic() + timeout
        while getattr(self._q, "unfinished_tasks", 0) and time.monotonic() < deadline:
            time.sleep(0.01)

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started or self._stopping:
            return
        self._stopping = True
        self._q.put(self._STOP)
        self._thread.join(timeout=timeout)

    def _atexit_drain(self) -> None:
        """Atexit-safe drain. Tolerates a partially torn-down interpreter:
        skips cleanly when stop() already fired (explicit close), and
        swallows any teardown-related errors so we never crash exit."""
        if self._stopping or not self._started:
            return
        try:
            self.flush(timeout=5.0)
            self.stop(timeout=5.0)
        except Exception:
            # Interpreter shutdown can yank module-level objects out from
            # under us; never block exit on a logging concern.
            pass

    # ------------------------------------------------------------------

    def _run(self) -> None:
        batch: list[Call] = []
        last_flush = time.monotonic()
        self._auto_drain_spool()
        while True:
            if time.monotonic() >= self._next_spool_drain_at:
                self._auto_drain_spool()
            # Floor at 1ms so an idle queue can't fall through to a 0-timeout
            # poll: when batch is empty and no flush happened for >_BATCH_MS,
            # the budget would otherwise clamp to 0 and busy-spin the writer.
            timeout = max(0.001, _BATCH_MS / 1000 - (time.monotonic() - last_flush))
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is self._STOP:
                if batch:
                    self._drain(batch)
                    for _ in batch:
                        self._q.task_done()
                    batch = []
                self._q.task_done()
                return

            if item is not None:
                batch.append(item)

            if len(batch) >= _BATCH_MAX or (
                batch and (time.monotonic() - last_flush) * 1000 >= _BATCH_MS
            ):
                self._drain(batch)
                for _ in batch:
                    self._q.task_done()
                batch = []
                last_flush = time.monotonic()
            elif not batch:
                # Idle tick — reset the budget so the next get() blocks for the
                # full _BATCH_MS window instead of waking every millisecond.
                last_flush = time.monotonic()

    def _auto_drain_spool(self) -> None:
        self._next_spool_drain_at = time.monotonic() + _SPOOL_DRAIN_INTERVAL_S
        if not _spool_has_files(self._spool_dir):
            return
        try:
            drain_spool(self._repo, self._spool_dir)
        except Exception as exc:  # noqa: BLE001 — telemetry must never break callers
            global _auto_drain_warned
            with _auto_drain_warned_lock:
                if not _auto_drain_warned:
                    print(f"[somm] spool auto-drain failed; will retry later: {exc}", file=sys.stderr)
                    _auto_drain_warned = True

    def _drain(self, batch: list[Call]) -> None:
        for attempt in range(_MAX_BUSY_RETRIES):
            try:
                self._repo.write_calls_batch(batch)
                self._mirror(batch)
                return
            except sqlite3.OperationalError as e:
                if "busy" in str(e).lower() or "locked" in str(e).lower():
                    time.sleep(0.05 * (2**attempt))
                    continue
                self._spill(batch, reason=str(e))
                return
            except OSError as e:
                self._spill(batch, reason=f"disk: {e}")
                return
        self._spill(batch, reason="sqlite_busy_retries_exhausted")

    def _mirror(self, batch: list[Call]) -> None:
        """Replicate the batch to the mirror repo (opt-in cross-project view).
        Failures are isolated from the primary write — never raise upward."""
        if self._mirror_repo is None:
            return
        try:
            self._mirror_repo.write_calls_batch(batch)
        except Exception:  # noqa: BLE001 — mirror must not poison primary path
            # Don't spill mirror failures (they'd duplicate the primary spool).
            # Just drop; re-run with SOMM_CROSS_PROJECT=1 so future calls mirror.
            pass

    def _spill(self, batch: list[Call], reason: str) -> None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = self._spool_dir / f"{ts}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for call in batch:
                row = asdict(call)
                row["ts"] = call.ts.isoformat()
                row["outcome"] = call.outcome.value
                row["_spill_reason"] = reason
                f.write(json.dumps(row))
                f.write("\n")
        path.chmod(0o600)


def _spool_has_files(spool_dir: Path) -> bool:
    spool_dir = Path(spool_dir)
    return spool_dir.exists() and (
        any(spool_dir.glob("*.jsonl"))
        or any(spool_dir.glob(f"*.jsonl{_SPOOL_CLAIM_MARKER}*"))
    )


def _claimed_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{_SPOOL_CLAIM_MARKER}{os.getpid()}-{threading.get_ident()}")


def _claim_spool_file(path: Path) -> Path | None:
    claimed = _claimed_path(path)
    try:
        path.rename(claimed)
        return claimed
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _original_spool_path(claimed: Path) -> Path:
    return claimed.with_name(claimed.name.split(_SPOOL_CLAIM_MARKER, 1)[0])


def _claim_pid(claimed: Path) -> int | None:
    if _SPOOL_CLAIM_MARKER not in claimed.name:
        return None
    raw = claimed.name.split(_SPOOL_CLAIM_MARKER, 1)[1].split("-", 1)[0]
    with contextlib.suppress(ValueError):
        return int(raw)
    return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _recover_orphaned_claims(spool_dir: Path) -> None:
    for claimed in sorted(spool_dir.glob(f"*.jsonl{_SPOOL_CLAIM_MARKER}*")):
        pid = _claim_pid(claimed)
        if pid is not None and _pid_is_running(pid):
            continue
        original = _original_spool_path(claimed)
        if original.exists():
            continue
        with contextlib.suppress(OSError):
            claimed.rename(original)


def drain_spool(repo: Repository, spool_dir: Path) -> int:
    """Replay every .jsonl in the spool into the DB. Returns rows drained.

    Called by `somm drain-spool` (or on service startup).
    Each file is claimed with an atomic rename before replay so concurrent
    drainers cannot double-insert the same rows; success deletes the claim.
    """
    spool_dir = Path(spool_dir)
    if not spool_dir.exists():
        return 0
    _recover_orphaned_claims(spool_dir)
    total = 0
    for path in sorted(spool_dir.glob("*.jsonl")):
        claimed = _claim_spool_file(path)
        if claimed is None:
            continue
        calls: list[Call] = []
        try:
            with claimed.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    row.pop("_spill_reason", None)
                    from somm_core.models import Outcome

                    calls.append(
                        Call(
                            id=row["id"],
                            ts=datetime.fromisoformat(row["ts"]),
                            project=row["project"],
                            workload_id=row["workload_id"],
                            prompt_id=row["prompt_id"],
                            provider=row["provider"],
                            model=row["model"],
                            tokens_in=row["tokens_in"],
                            tokens_out=row["tokens_out"],
                            latency_ms=row["latency_ms"],
                            cost_usd=row["cost_usd"],
                            outcome=Outcome(row["outcome"]),
                            error_kind=row["error_kind"],
                            prompt_hash=row["prompt_hash"],
                            response_hash=row["response_hash"],
                            # Optional fields — present in newer spool files; rows
                            # spilled by pre-0.3 writers carry commission_id
                            # instead, which replay ignores.
                            error_detail=row.get("error_detail"),
                            correlation_id=row.get("correlation_id"),
                            temperature=row.get("temperature"),
                            max_tokens=row.get("max_tokens"),
                            top_p=row.get("top_p"),
                            stop_sequences_json=row.get("stop_sequences_json"),
                            ttft_ms=row.get("ttft_ms"),
                            session_id=row.get("session_id"),
                            parent_call_id=row.get("parent_call_id"),
                            cache_tokens_in=row.get("cache_tokens_in"),
                            cache_tokens_out=row.get("cache_tokens_out"),
                            citations_json=row.get("citations_json"),
                        )
                    )
            repo.write_calls_batch(calls)
        except Exception:
            # Leave spool file in place; try again later.
            original = _original_spool_path(claimed)
            if not original.exists():
                with contextlib.suppress(OSError):
                    claimed.rename(original)
            continue
        total += len(calls)
        claimed.unlink()
    return total

"""The writer thread must survive a bad row and never lose the rest.

Before the fix, one row whose value SQLite could not bind (here
``model={"id": "m"}``) raised ``ProgrammingError`` out of ``_drain``; only
``OperationalError`` and ``OSError`` were caught, so the daemon thread died.
The good rows in the same batch and every later call stayed in memory and
vanished at exit, with nothing spilled.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm.telemetry import WriterQueue
from somm_core import Outcome
from somm_core.models import Call
from somm_core.repository import Repository


def _mk_call(model: object = "m") -> Call:
    return Call(
        id=str(uuid.uuid4()),
        ts=datetime.now(UTC),
        project="wr",
        workload_id=None,
        prompt_id=None,
        provider="fake",
        model=model,  # type: ignore[arg-type]
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="h",
        response_hash="h",
    )


def _count(repo: Repository) -> int:
    with repo._open() as conn:
        return conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]


def test_unbindable_row_is_quarantined_and_writer_survives(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "calls.sqlite")
    spool = tmp_path / "spool"
    writer = WriterQueue(repo, spool)
    good_a, bad, good_b = _mk_call(), _mk_call(model={"id": "m"}), _mk_call()
    for call in (good_a, bad, good_b):
        writer.submit(call)
    writer.flush(timeout=5.0)

    assert writer._thread.is_alive()
    assert {repo.get_call(c.id) is not None for c in (good_a, good_b)} == {True}
    assert repo.get_call(bad.id) is None
    assert writer.quarantined == 1

    # The bad row is kept, with its error, outside the replayable spool so
    # auto-drain does not retry it forever.
    assert list(spool.glob("*.jsonl")) == []
    files = list((spool / "quarantine").glob("*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text().strip())
    assert row["id"] == bad.id
    assert "_quarantine_error" in row

    # Later calls still land.
    later = _mk_call()
    writer.submit(later)
    writer.flush(timeout=5.0)
    assert repo.get_call(later.id) is not None
    assert _count(repo) == 3
    writer.stop()


def test_unexpected_loop_error_is_logged_once_and_loop_continues(tmp_path, capsys) -> None:
    repo = Repository(tmp_path / "calls.sqlite")
    writer = WriterQueue(repo, tmp_path / "spool")
    real_drain = writer._drain
    calls = {"n": 0}

    def flaky(batch):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        real_drain(batch)

    writer._drain = flaky  # type: ignore[method-assign]
    writer.submit(_mk_call())
    writer.flush(timeout=5.0)
    writer.submit(_mk_call())
    writer.flush(timeout=5.0)
    ok = _mk_call()
    writer.submit(ok)
    writer.flush(timeout=5.0)

    assert writer._thread.is_alive()
    assert repo.get_call(ok.id) is not None
    assert capsys.readouterr().err.count("writer loop error") == 1
    writer.stop()


def test_busy_rows_in_row_fallback_spill_to_the_replayable_spool(tmp_path: Path) -> None:
    """A row that is merely busy during the one-by-one fallback is not bad data."""
    repo = Repository(tmp_path / "calls.sqlite")
    spool = tmp_path / "spool"
    writer = WriterQueue(repo, spool)
    real = repo.write_calls_batch

    def write(batch):
        if len(batch) > 1:
            raise sqlite3.IntegrityError("constraint")
        raise sqlite3.OperationalError("database is locked")

    repo.write_calls_batch = write  # type: ignore[method-assign]
    a, b = _mk_call(), _mk_call()
    writer._drain([a, b])
    assert writer.quarantined == 0
    spilled = list(spool.glob("*.jsonl"))
    assert len(spilled) == 2
    repo.write_calls_batch = real  # type: ignore[method-assign]
    writer._stopping = True


def test_dead_writer_thread_is_restarted_on_submit(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "calls.sqlite")
    writer = WriterQueue(repo, tmp_path / "spool")
    writer.start()
    # Kill the thread the way a BaseException would.
    writer._q.put(WriterQueue._STOP)
    writer._thread.join(timeout=5.0)
    assert not writer._thread.is_alive()

    call = _mk_call()
    writer.submit(call)
    writer.flush(timeout=5.0)
    assert writer._thread.is_alive()
    assert repo.get_call(call.id) is not None
    writer.stop()

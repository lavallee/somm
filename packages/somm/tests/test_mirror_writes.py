"""The mirror write must not drop a batch silently.

Every project on the machine writes to one mirror, so it is the most
contended file there is. Before the fix a busy mirror lost the batch on the
first ``database is locked``, with no retry and no trace: the mirror for a
project could stop while its ledger kept growing, and nobody could tell.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm.telemetry import WriterQueue
from somm_core import Outcome
from somm_core.models import Call
from somm_core.repository import Repository


def _call() -> Call:
    return Call(
        id=str(uuid.uuid4()), ts=datetime.now(UTC), project="mw", workload_id=None,
        prompt_id=None, provider="fake", model="m", tokens_in=1, tokens_out=1, latency_ms=1,
        cost_usd=0.0, outcome=Outcome.OK, error_kind=None, prompt_hash="h", response_hash="h",
    )


class _FlakyMirror:
    """A mirror that is busy ``busy`` times, then writes for real."""

    def __init__(self, repo: Repository, busy: int, error: str = "database is locked"):
        self._repo, self._busy, self._error = repo, busy, error

    def write_calls_batch(self, calls):
        if self._busy:
            self._busy -= 1
            raise sqlite3.OperationalError(self._error)
        self._repo.write_calls_batch(calls)


def _ids(repo: Repository) -> set[str]:
    with repo._open() as conn:
        return {r[0] for r in conn.execute("SELECT id FROM calls")}


def test_a_briefly_busy_mirror_still_gets_the_batch(tmp_path: Path) -> None:
    primary, mirror = Repository(tmp_path / "p.sqlite"), Repository(tmp_path / "m.sqlite")
    writer = WriterQueue(primary, tmp_path / "spool", mirror_repo=_FlakyMirror(mirror, busy=2))
    call = _call()
    writer.submit(call)
    writer.flush(timeout=5.0)
    writer.stop()

    assert _ids(primary) == {call.id}
    assert _ids(mirror) == {call.id}
    assert writer.mirror_failures == 0


def test_a_failing_mirror_is_counted_and_said_once(tmp_path: Path, capsys) -> None:
    primary, mirror = Repository(tmp_path / "p.sqlite"), Repository(tmp_path / "m.sqlite")
    flaky = _FlakyMirror(mirror, busy=10**6, error="database or disk is full")
    writer = WriterQueue(primary, tmp_path / "spool", mirror_repo=flaky)
    calls = [_call(), _call()]
    for c in calls:
        writer.submit(c)
        writer.flush(timeout=5.0)
    writer.stop()

    assert _ids(primary) == {c.id for c in calls}  # the primary write is untouched
    assert _ids(mirror) == set()
    assert writer.mirror_failures == 2
    err = capsys.readouterr().err
    assert err.count("mirror write failed") == 1
    assert "somm mirror sync" in err

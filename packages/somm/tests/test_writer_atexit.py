"""WriterQueue must drain pending calls on normal process exit.

The writer thread is a daemon — Python kills it during interpreter shutdown
without giving it a chance to finish its batch. Before this fix, calls
submitted in the last ~100ms of a script's life vanished entirely (not
even spilled to the JSONL fallback, since spill only fires on a *drain
failure*, not on writer-thread death).

The two tests below cover both paths: the in-process atexit hook fires
correctly when invoked directly, and a real subprocess that exits without
explicit close() lands its calls.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm.telemetry import WriterQueue
from somm_core import Outcome
from somm_core.config import Config
from somm_core.models import Call
from somm_core.repository import Repository


def _mk_call(project: str = "atx", model: str = "m") -> Call:
    return Call(
        id=str(uuid.uuid4()),
        ts=datetime.now(UTC),
        project=project,
        workload_id=None,
        prompt_id=None,
        provider="fake",
        model=model,
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="h",
        response_hash="h",
    )


def test_atexit_drain_lands_pending_calls(tmp_path: Path) -> None:
    """Direct test: build a WriterQueue, submit, invoke the atexit hook,
    verify the row landed in the DB."""
    cfg = Config()
    cfg.project = "atx"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    repo = Repository(cfg.db_path)

    writer = WriterQueue(repo, cfg.spool_dir)
    call = _mk_call()
    writer.submit(call)

    # Simulate the atexit hook firing without ever calling close().
    writer._atexit_drain()

    landed = repo.get_call(call.id)
    assert landed is not None, "submitted call must be visible after atexit drain"
    assert landed.id == call.id


def test_atexit_drain_idempotent_after_close(tmp_path: Path) -> None:
    """If close() ran first, the atexit hook should be a no-op (no double-stop)."""
    cfg = Config()
    cfg.project = "atx"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    repo = Repository(cfg.db_path)

    writer = WriterQueue(repo, cfg.spool_dir)
    writer.submit(_mk_call())
    writer.flush(timeout=2.0)
    writer.stop(timeout=2.0)

    # Should return cleanly without re-stopping.
    writer._atexit_drain()


def test_subprocess_exit_without_close_lands_calls(tmp_path: Path) -> None:
    """End-to-end: a subprocess constructs SommLLM, submits a call via
    generate(), and exits without explicit close(). The atexit-registered
    drain should land the row before the process actually exits."""
    db_dir = tmp_path / ".somm"
    db_dir.mkdir(parents=True, exist_ok=True)
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from somm.client import SommLLM
        from somm.providers.base import ProviderHealth, SommResponse
        from somm_core.config import Config

        class FakeProvider:
            name = "fake"
            def generate(self, request):
                return SommResponse(text="hi", model="m", tokens_in=1,
                                    tokens_out=1, latency_ms=1, raw=None)
            def stream(self, request):
                yield
            def health(self):
                return ProviderHealth(available=True)
            def models(self):
                return []
            def estimate_tokens(self, text, model):
                return 1

        cfg = Config()
        cfg.project = "atx_subproc"
        cfg.mode = "observe"
        cfg.db_dir = Path("{db_dir.as_posix()}")
        cfg.spool_dir = cfg.db_dir / "spool"

        llm = SommLLM(config=cfg, providers=[FakeProvider()])
        result = llm.generate("p", workload="test")
        # Print the call_id so the test can assert it landed; then exit
        # WITHOUT calling llm.close() — that's the case we're testing.
        print(result.call_id)
        """
    )
    out = subprocess.check_output([sys.executable, "-c", script], timeout=30).decode()
    call_id = out.strip().splitlines()[-1]
    assert call_id, f"subprocess did not print a call_id; output={out!r}"

    repo = Repository(db_dir / "calls.sqlite")
    landed = repo.get_call(call_id)
    assert landed is not None, (
        f"call {call_id!r} should have landed via atexit drain after subprocess exit"
    )
    assert landed.project == "atx_subproc"
    assert landed.outcome == Outcome.OK

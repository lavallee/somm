from __future__ import annotations

import threading
from datetime import UTC, datetime

from somm_core import Call, Outcome
from somm_core.repository import Repository


def _call(call_id: str, workload_id: str) -> Call:
    return Call(
        id=call_id,
        ts=datetime.now(UTC),
        project="repo-test",
        workload_id=workload_id,
        prompt_id=None,
        provider="fake",
        model="m",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="p",
        response_hash="r",
    )


def test_open_reuses_connection_in_same_thread(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")

    first = repo._open()
    second = repo._open()

    assert first is second
    repo.close()


def test_open_uses_different_connections_in_different_threads(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    barrier = threading.Barrier(3)
    connections = []

    def worker() -> None:
        conn = repo._open()
        connections.append(conn)
        barrier.wait()
        barrier.wait()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()

    assert len({id(conn) for conn in connections}) == 2

    barrier.wait()
    for thread in threads:
        thread.join()
    repo.close()


def test_open_reopens_after_fork_pid_change(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "calls.sqlite")
    first = repo._open()

    monkeypatch.setattr("somm_core.repository.os.getpid", lambda: 999_999)
    second = repo._open()

    assert second is not first
    repo.close()


def test_repository_concurrent_threads_share_instance_without_errors(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for n in range(10):
                wl = repo.register_workload(
                    name=f"workload-{idx}-{n}",
                    project="repo-test",
                )
                assert repo.workload_by_name(wl.name, "repo-test") is not None
                repo.write_call(_call(f"call-{idx}-{n}", wl.id))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with repo._open() as conn:
        workload_count = conn.execute("SELECT COUNT(*) FROM workloads").fetchone()[0]
        call_count = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    assert workload_count == 80
    assert call_count == 80
    repo.close()

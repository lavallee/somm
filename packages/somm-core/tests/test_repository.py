from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest
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


def test_stats_by_workload_serving_profile_rolls_up(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload(
        name="serving-profile",
        project="repo-test",
        max_p95_latency_ms=200,
        max_p95_ttft_ms=60,
        max_tpot_ms=8.0,
    )
    ok_calls = [
        (100, 20, 11, 5, 2, 1),
        (200, 50, 21, 5, 3, 2),
        (400, 100, 41, 5, 4, 3),
    ]
    for idx, (latency_ms, ttft_ms, tokens_out, tokens_in, cache_in, cache_out) in enumerate(
        ok_calls
    ):
        repo.write_call(
            Call(
                id=f"ok-{idx}",
                ts=datetime.now(UTC),
                project="repo-test",
                workload_id=wl.id,
                prompt_id=None,
                provider="fake",
                model="m",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=0.0,
                outcome=Outcome.OK,
                error_kind=None,
                prompt_hash=f"p-ok-{idx}",
                response_hash=f"r-ok-{idx}",
                ttft_ms=ttft_ms,
                cache_tokens_in=cache_in,
                cache_tokens_out=cache_out,
            )
        )
    repo.write_call(
        Call(
            id="failed-fast",
            ts=datetime.now(UTC),
            project="repo-test",
            workload_id=wl.id,
            prompt_id=None,
            provider="fake",
            model="m",
            tokens_in=1,
            tokens_out=1,
            latency_ms=50,
            cost_usd=0.0,
            outcome=Outcome.TIMEOUT,
            error_kind="timeout",
            prompt_hash="p-failed",
            response_hash="r-failed",
            ttft_ms=10,
            cache_tokens_in=1,
            cache_tokens_out=1,
        )
    )

    row = repo.stats_by_workload("repo-test", since_days=1)[0]

    assert row["workload"] == "serving-profile"
    assert row["n_calls"] == 4
    assert row["n_ok"] == 3
    assert row["n_failed"] == 1
    assert row["tokens_in"] == 16
    assert row["cache_tokens_in"] == 10
    assert row["cache_tokens_out"] == 7
    assert row["cache_read_ratio"] == pytest.approx(10 / 16)
    assert row["input_tokens_per_second"] == pytest.approx(15_000 / 700)
    assert row["p50_latency_ms"] == 200
    assert row["p95_latency_ms"] == 400
    assert row["p99_latency_ms"] == 400
    assert row["p50_ttft_ms"] == 50
    assert row["p95_ttft_ms"] == 100
    assert row["p99_ttft_ms"] == 100
    assert row["tpot_ms"] == pytest.approx((8.0 + 7.5 + 7.5) / 3)
    assert row["output_tokens_per_second"] == pytest.approx(73_000 / 700)
    assert row["total_tokens_per_second"] == pytest.approx(88_000 / 700)
    assert row["requests_per_second"] == pytest.approx(3_000 / 700)
    assert row["goodput_slo_latency_ms"] == 200
    assert row["goodput_slo_ttft_ms"] == 60
    assert row["goodput_slo_tpot_ms"] == 8.0
    assert row["goodput_calls"] == 2
    assert row["goodput_under_slo"] == pytest.approx(2 / 3)
    assert row["goodput_requests_per_second"] == pytest.approx(2_000 / 300)
    assert row["goodput_output_tokens_per_second"] == pytest.approx(32_000 / 300)
    assert row["goodput_total_tokens_per_second"] == pytest.approx(42_000 / 300)
    assert row["goodput_tokens_in"] == 10
    assert row["goodput_tokens_out"] == 32
    assert row["goodput_tokens_total"] == 42

    global_row = repo.stats_global_by_workload(since_days=1)[0]
    assert global_row["project"] == "repo-test"
    assert global_row["p95_latency_ms"] == 400

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


def test_call_cost_provenance_round_trips(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    call = _call("provenance", "workload")
    call.cost_basis = "reported"
    call.cost_kind = "marginal"
    call.cost_accuracy = "actual"
    call.cost_source = "provider:invoice"
    call.pricing_version = "2026-07"
    call.observation_role = "shadow_gold"
    call.source_call_id = "production-1"
    call.eval_result_id = 42
    call.provider_request_id = "provider-request-1"
    call.billing_id = "billing-1"
    call.origin = "native"
    call.budget_eligible = False

    repo.write_call(call)
    recovered = repo.get_call(call.id)

    assert recovered is not None
    assert recovered.cost_basis == "reported"
    assert recovered.cost_kind == "marginal"
    assert recovered.cost_accuracy == "actual"
    assert recovered.cost_source == "provider:invoice"
    assert recovered.pricing_version == "2026-07"
    assert recovered.observation_role == "shadow_gold"
    assert recovered.source_call_id == "production-1"
    assert recovered.eval_result_id == 42
    assert recovered.provider_request_id == "provider-request-1"
    assert recovered.billing_id == "billing-1"
    assert recovered.origin == "native"
    assert recovered.budget_eligible is False
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


# ---- call_updates recording API ---------------------------------------------


def _correlated_call(call_id: str, correlation_id: str | None) -> Call:
    call = _call(call_id, "callupdate-test")
    call.correlation_id = correlation_id
    return call


def test_record_call_update_appends_generic_field(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_call("c-1", "callupdate-test"))

    repo.record_call_update("c-1", field="fab_job_outcome", value='{"status":"done"}')

    with repo._open() as conn:
        rows = conn.execute(
            "SELECT field, value FROM call_updates WHERE call_id = ?", ("c-1",)
        ).fetchall()
    assert rows == [("fab_job_outcome", '{"status":"done"}')]
    repo.close()


def test_record_call_update_rejects_bad_field_and_value(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    with pytest.raises(ValueError):
        repo.record_call_update("c-1", field="", value="x")
    with pytest.raises(ValueError):
        repo.record_call_update("c-1", field="outcome", value=None)  # type: ignore[arg-type]
    repo.close()


def test_record_outcome_update_still_persists_outcome_row(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_call("c-oc", "callupdate-test"))

    repo.record_outcome_update("c-oc", Outcome.BAD_JSON)

    with repo._open() as conn:
        rows = conn.execute(
            "SELECT field, value FROM call_updates WHERE call_id = ?", ("c-oc",)
        ).fetchall()
    assert rows == [("outcome", Outcome.BAD_JSON.value)]
    repo.close()


def test_record_call_updates_for_correlation_links_exact_and_children(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_correlated_call("c-exact", "job-42"))
    repo.write_call(_correlated_call("c-child", "job-42:attempt:1"))
    repo.write_call(_correlated_call("c-other", "job-43"))
    repo.write_call(_correlated_call("c-none", None))

    linked = repo.record_call_updates_for_correlation(
        "job-42", field="fab_job_outcome", value='{"status":"needs_review"}'
    )

    assert sorted(linked) == ["c-child", "c-exact"]
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT call_id, field, value FROM call_updates ORDER BY call_id"
        ).fetchall()
    assert rows == [
        ("c-child", "fab_job_outcome", '{"status":"needs_review"}'),
        ("c-exact", "fab_job_outcome", '{"status":"needs_review"}'),
    ]
    repo.close()


def test_record_call_updates_for_correlation_exact_only(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_correlated_call("c-exact", "job-42"))
    repo.write_call(_correlated_call("c-child", "job-42:attempt:1"))

    linked = repo.record_call_updates_for_correlation(
        "job-42", field="grade", value="A", include_children=False
    )

    assert linked == ["c-exact"]
    repo.close()


def test_record_call_updates_for_correlation_escapes_like_wildcards(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_correlated_call("c-underscore", "job_1"))
    repo.write_call(_correlated_call("c-child", "job_1:attempt:1"))
    # Without ESCAPE, the `_` in the prefix would wildcard-match this one.
    repo.write_call(_correlated_call("c-lookalike", "jobX1:attempt:1"))

    linked = repo.record_call_updates_for_correlation(
        "job_1", field="grade", value="A"
    )

    assert sorted(linked) == ["c-child", "c-underscore"]
    repo.close()


def test_record_call_updates_for_correlation_no_matches_is_empty(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")

    linked = repo.record_call_updates_for_correlation(
        "job-nothing", field="grade", value="A"
    )

    assert linked == []
    with repo._open() as conn:
        count = conn.execute("SELECT COUNT(*) FROM call_updates").fetchone()[0]
    assert count == 0
    repo.close()


def test_record_call_updates_for_correlation_rejects_empty_correlation(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    with pytest.raises(ValueError):
        repo.record_call_updates_for_correlation("", field="grade", value="A")
    repo.close()

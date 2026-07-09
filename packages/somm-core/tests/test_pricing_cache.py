from __future__ import annotations

from somm_core.pricing import _clear_price_cache, cost_for_call, write_intel
from somm_core.repository import Repository


def test_cost_for_call_caches_price_lookup_and_write_intel_invalidates(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    _clear_price_cache()
    write_intel(
        repo,
        provider="fake",
        model="m",
        price_in_per_1m=1.0,
        price_out_per_1m=2.0,
        context_window=None,
        capabilities=None,
        source="test",
    )

    traces: list[str] = []
    conn = repo._open()
    conn.set_trace_callback(
        lambda stmt: traces.append(stmt) if "FROM model_intel" in stmt else None
    )

    assert cost_for_call(repo, "fake", "m", 100, 50) == 0.0002
    assert len(traces) == 1

    assert cost_for_call(repo, "fake", "m", 200, 100) == 0.0004
    assert len(traces) == 1

    write_intel(
        repo,
        provider="fake",
        model="m",
        price_in_per_1m=3.0,
        price_out_per_1m=4.0,
        context_window=None,
        capabilities=None,
        source="test",
    )
    assert cost_for_call(repo, "fake", "m", 100, 50) == 0.0005
    assert len(traces) == 2

    conn.set_trace_callback(None)
    _clear_price_cache()
    repo.close()

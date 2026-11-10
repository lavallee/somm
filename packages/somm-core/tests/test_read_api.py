"""The supported read API beyond search_calls: guard, health, rollup, tables, fleet, vocabulary."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from somm_core.models import Call, Outcome
from somm_core.read import (
    across_ledgers,
    get_calls,
    provider_health,
    rollup,
    schema_status,
    search_calls,
    table_rows,
)
from somm_core.repository import Repository
from somm_core.schema import _list_migrations
from somm_core.version import SCHEMA_VERSION
from somm_core.vocabulary import vocabulary


def _call(project="p", provider="a", model="m1", *, ago_h=1, **kw) -> Call:
    base = dict(
        id=str(uuid.uuid4()),
        ts=datetime.now(UTC) - timedelta(hours=ago_h),
        project=project,
        workload_id=None,
        prompt_id=None,
        provider=provider,
        model=model,
        tokens_in=10,
        tokens_out=5,
        latency_ms=100,
        cost_usd=0.5,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="p",
        response_hash="r",
    )
    base.update(kw)
    return Call(**base)


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / "calls.sqlite")
    wl = r.register_workload(name="extract", project="p")
    for ms in (100, 200, 300, 400):
        r.write_call(_call(latency_ms=ms, workload_id=wl.id))
    r.write_call(_call(ago_h=2, outcome=Outcome.RATE_LIMIT, error_kind="RateLimit", latency_ms=5))
    r.write_call(_call(provider="b", model="m2", cost_usd=0.0, cost_source="unknown:no-price"))
    r.write_call(_call(project="q", provider="b", model="m2"))
    r.write_call(_call(ago_h=24 * 40))  # outside default windows
    return r


def test_schema_status_current(repo):
    s = schema_status(repo.db_path)
    assert s["understood"] and s["reason"] is None
    assert s["schema_version"] == s["library_schema_version"] == SCHEMA_VERSION
    assert s["missing_columns"] == {"calls": []} and s["missing_tables"] == []


def test_schema_status_rejects_non_ledger_and_newer(tmp_path, repo):
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).execute("CREATE TABLE t (x)").connection.close()
    assert not schema_status(other)["understood"]
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION + 1,))
    s = schema_status(repo.db_path)
    assert not s["understood"] and "newer" in s["reason"]


def _old_ledger(tmp_path, upto: int):
    path = tmp_path / f"v{upto}.sqlite"
    with sqlite3.connect(path) as conn:
        for version, p in _list_migrations():
            if version <= upto:
                conn.executescript(p.read_text())
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        now = datetime.now(UTC)
        for i, outcome in enumerate(["ok", "ok", "timeout"]):
            conn.execute(
                "INSERT INTO calls (id, ts, project, provider, model, latency_ms, cost_usd, outcome,"
                " prompt_hash, response_hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"c{i}", (now - timedelta(hours=i)).isoformat(), "p", "a", "m", 100 * (i + 1),
                 0.25, outcome, "h", "h"),
            )
    return path


def test_reads_degrade_on_an_old_ledger(tmp_path):
    path = _old_ledger(tmp_path, 10)
    s = schema_status(path)
    assert s["understood"] and s["schema_version"] == 10
    assert {"routing_json", "data_tier", "cost_source"} <= set(s["missing_columns"]["calls"])
    assert "recommendations" not in s["missing_tables"]  # exists from the start
    assert "campaigns" in s["missing_tables"]

    rows = search_calls(path, "p", since_days=1, origin="native")
    assert rows == []  # a column the ledger lacks matches nothing, it does not crash
    rows = search_calls(path, "p", since_days=1, q="c0")
    assert rows[0]["routing_json"] is None and rows[0]["data_tier"] is None
    assert rows[0]["route_source"] is None and rows[0]["cost_known"] is True

    assert get_calls(path, ["c1", "nope"])[0]["id"] == "c1"
    health = provider_health(path, "p", since_days=1)
    assert health[0]["calls"] == 3 and health[0]["last_error_outcome"] == "timeout"
    day = rollup(path, "p", by=["day", "data_tier"], since_days=1)
    assert sum(r["calls"] for r in day) == 3 and day[0]["data_tier"] is None
    assert table_rows(path, "campaigns") == []
    assert len(table_rows(path, "calls", project="p")) == 3


def test_provider_health(repo):
    rows = {(r["provider"], r["model"]): r for r in provider_health(repo.db_path, "p")}
    a = rows[("a", "m1")]
    assert (a["calls"], a["ok"], a["errors"]) == (5, 4, 1)
    assert a["ok_rate"] == pytest.approx(0.8)
    assert (a["p50_ms"], a["p95_ms"]) == (300, 400)  # ok calls only; the 5 ms failure is excluded
    assert a["last_error_kind"] == "RateLimit" and a["last_error_outcome"] == "rate_limit"
    assert a["last_ok_ts"] > a["last_error_ts"]
    assert rows[("b", "m2")]["last_error_ts"] is None
    # busiest first, project-less spans every project, 40-day-old call is outside the window
    assert provider_health(repo.db_path)[0]["calls"] == 5
    assert sum(r["calls"] for r in provider_health(repo.db_path)) == 7
    assert [r["calls"] for r in provider_health(repo.db_path, "p", workload="extract")] == [4]


def test_rollup_respects_unknown_cost(repo):
    (total,) = rollup(repo.db_path, "p", by=(), since_days=7)
    assert total["calls"] == 6 and total["unknown_cost_calls"] == 1
    assert total["cost_usd"] == pytest.approx(2.5)  # five known calls at 0.5; the unknown 0.0 is not "free"
    by_wl = {r["workload"]: r["calls"] for r in rollup(repo.db_path, "p", by=["workload"])}
    assert by_wl == {"extract": 4, "(unregistered)": 2}
    assert [r["calls"] for r in rollup(repo.db_path, by=["provider", "outcome"], provider="b")] == [2]
    assert len(rollup(repo.db_path, "p", by=["day"], since_days=60)) == 2
    with pytest.raises(ValueError):
        rollup(repo.db_path, by=["colour"])


def test_search_calls_new_keys(repo):
    repo.write_call(
        _call(routing_json=json.dumps({"route_source": "pin", "route": {"depth": 0}}),
              data_tier="local", cost_source="unknown:no-price")
    )
    row = search_calls(repo.db_path, "p", since_days=1, limit=1)[0]
    assert row["route_source"] == "pin" and row["route"] == {"depth": 0}
    assert row["cost_known"] is False and row["data_tier"] == "local"


def test_table_rows(repo):
    assert len(table_rows(repo.db_path, "calls", project="p")) == 7
    assert len(table_rows(repo.db_path, "calls", since=(datetime.now(UTC) - timedelta(days=1)).isoformat())) == 7
    assert [r["name"] for r in table_rows(repo.db_path, "workloads")] == ["extract"]
    assert table_rows(repo.db_path, "recommendations") == []
    with pytest.raises(ValueError):
        table_rows(repo.db_path, "sqlite_master")
    with pytest.raises(ValueError):
        table_rows(repo.db_path, "recommendations", project="p")


def test_across_ledgers(tmp_path, repo):
    gone = tmp_path / "gone.sqlite"
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"not a database at all, just text" * 10)
    out = across_ledgers(
        provider_health,
        ledgers=[("p", repo.db_path), ("gone", gone), ("bad", bad)],
        since_days=7,
        project="p",
    )
    assert {r["ledger_project"] for r in out["rows"]} == {"p"}
    assert {s["project"] for s in out["skipped"]} == {"gone", "bad"}
    assert "missing" in next(s for s in out["skipped"] if s["project"] == "gone")["reason"]
    one = across_ledgers(schema_status, ledgers=[("p", repo.db_path)])
    assert one["rows"][0]["understood"] and one["rows"][0]["ledger_project"] == "p"
    calls = across_ledgers(search_calls, ledgers=[("p", repo.db_path)], pass_project=True, limit=1)
    assert len(calls["rows"]) == 1


def test_across_ledgers_reads_the_registry(tmp_path, repo, monkeypatch):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"projects": {
        "p": {"db_path": str(repo.db_path)}, "ghost": {"db_path": str(tmp_path / "x.sqlite")}}}))
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(reg))
    out = across_ledgers(rollup, by=(), since_days=7)
    assert [r["ledger_project"] for r in out["rows"]] == ["p"]
    assert out["skipped"][0]["project"] == "ghost"


def test_get_calls_matches_search(repo):
    row = search_calls(repo.db_path, "p", since_days=1, limit=1)[0]
    assert get_calls(repo.db_path, [row["id"]]) == [row]


def test_vocabulary_covers_the_enums():
    v = vocabulary()
    assert set(v["outcomes"]) == {o.value for o in Outcome}
    assert v["outcomes"]["rate_limit"]["detractor"] is True
    assert v["schema_version"] == SCHEMA_VERSION and v["vocabulary_version"] >= 1
    assert v["data_tiers"][0] == "local" and v["data_tiers"][-1] == "unvetted"
    assert v["route_sources"]["pin"] is False and v["route_sources"]["policy"] is True
    assert "usd_equiv" in v["meter_units"]["plan_limit"]
    assert v["cost_sources"]["unknown_prefix"] == "unknown:"
    assert v["error_kinds"]["open"] is True
    json.dumps(v)  # plain data

"""somm_core.read — one query behind every surface; somm_core.aliases — rules."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from somm_core.aliases import canonicalize, needs_alias, seed_model_aliases
from somm_core.models import Call, Decision, Outcome
from somm_core.read import fleet_summary, north_star, search_calls, spend_today
from somm_core.repository import Repository


def _call(project: str, provider: str, model: str, *, ts=None, workload_id=None, **kw) -> Call:
    base = dict(
        id=str(uuid.uuid4()),
        ts=ts or datetime.now(UTC),
        project=project,
        workload_id=workload_id,
        prompt_id=None,
        provider=provider,
        model=model,
        tokens_in=10,
        tokens_out=5,
        latency_ms=100,
        cost_usd=0.001,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="p",
        response_hash="r",
    )
    base.update(kw)
    return Call(**base)


def test_search_calls_filters_and_labels(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload(name="extract", project="p")
    repo.write_call(_call("p", "ollama", "qwen3:8b", workload_id=wl.id, session_id="s1"))
    repo.write_call(_call("p", "minimax", "MiniMax-M3", correlation_id="job-7"))
    repo.write_call(_call("other", "ollama", "qwen3:8b"))

    rows = search_calls(repo.db_path, "p", since_days=1, limit=10)
    assert len(rows) == 2
    assert {r["workload"] for r in rows} == {"extract", "(unregistered)"}
    assert set(rows[0]) >= {"id", "ts", "provider", "model", "cost_usd", "origin", "client_version"}

    assert [r["provider"] for r in search_calls(repo.db_path, "p", workload="extract")] == ["ollama"]
    assert [r["provider"] for r in search_calls(repo.db_path, "p", workload=wl.id)] == ["ollama"]
    assert [r["model"] for r in search_calls(repo.db_path, "p", q="job-7")] == ["MiniMax-M3"]
    with repo._open() as conn:
        assert len(search_calls(conn, "p", provider="minimax")) == 1

    # `since` (exclusive ISO cutoff) overrides the day window, oldest-first for a tail.
    old = _call("p", "ollama", "x", ts=datetime.now(UTC) - timedelta(days=3))
    repo.write_call(old)
    tail = search_calls(repo.db_path, "p", since=(datetime.now(UTC) - timedelta(days=4)).isoformat(),
                        newest_first=False, limit=10)
    assert tail[0]["id"] == old.id


def test_spend_today_uses_the_same_label(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_call("p", "minimax", "MiniMax-M3", cost_usd=0.5))
    rows = spend_today(repo.db_path, "p", None)
    assert rows == [{"workload": "(unregistered)", "spent_usd": 0.5, "cap_usd": None}]


def test_fleet_summary_reports_clients_loop_and_sentinels(tmp_path):
    repo = Repository(tmp_path / "global.sqlite")
    repo.write_call(_call("a", "openrouter", "*all_cooled", client_version=None))
    repo.write_call(_call("a", "", "", client_version=None))
    repo.write_call(_call("b", "claude-cli", "claude-opus-5[1m]", client_version="0.17.0"))
    repo.write_call(_call("b", "claude_cli", "claude-opus-4-7", client_version="0.17.0"))

    fleet = fleet_summary(repo.db_path, since_days=7)
    assert {(c["project"], c["client_version"]) for c in fleet["clients"]} == {
        ("a", "(pre-0.17)"), ("b", "0.17.0")
    }
    assert fleet["sentinel_rows"] == 2
    assert fleet["loop"]["eval_results"] == 0 and fleet["loop"]["recommendations"] == 0
    assert fleet["provider_spellings"] == {"claude-cli": ["claude-cli", "claude_cli"]}
    assert ("claude-cli", "claude-opus-5[1m]") in fleet["unaliased"]


def test_canonicalize_rules():
    assert canonicalize("claude-cli", "claude-opus-5[1m]") == "anthropic/claude-opus-5"
    assert canonicalize("claude-cli", "claude-opus-5") == "anthropic/claude-opus-5"
    assert canonicalize("claude_cli", "claude-opus-4-7") == "anthropic/claude-opus-4-7"
    assert canonicalize("openrouter", "google/gemma-3-27b-it:free") == "openrouter/google/gemma-3-27b-it"
    assert canonicalize("openrouter", "anthropic/claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"
    # Already canonical, or a moving nickname: leave alone rather than guess.
    assert canonicalize("ollama", "qwen3:8b") is None
    assert canonicalize("claude-cli", "sonnet") is None
    assert canonicalize("openrouter", "*all_cooled") is None
    assert canonicalize("minimax", "") is None


def test_needs_alias_skips_routes_that_are_already_canonical():
    # minimax/MiniMax-M3 has no alias row and `somm alias seed` never writes
    # one, so the doctor must not nag about it on a fresh install.
    assert needs_alias("minimax", "MiniMax-M3") is False
    assert needs_alias("minimax", "MiniMax-M2") is False
    assert needs_alias("ollama", "qwen3:8b") is False
    assert needs_alias("openrouter", "*all_cooled") is False
    # A rule can fix these, or they are moving nicknames: still worth a line.
    assert needs_alias("claude-cli", "claude-opus-5[1m]") is True
    assert needs_alias("claude_cli", "claude-opus-4-7") is True
    assert needs_alias("claude-cli", "sonnet") is True


def test_seed_model_aliases_writes_rule_rows_and_respects_manual(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_call("p", "claude-cli", "claude-opus-5[1m]"))
    repo.write_call(_call("p", "claude-cli", "claude-opus-5"))
    repo.write_call(_call("p", "ollama", "qwen3:8b"))
    repo.set_model_alias("custom/x", "claude-cli", "claude-opus-5", source="manual")

    n = seed_model_aliases(repo)

    assert n == 1
    aliases = repo.model_alias_map()
    assert aliases[("claude-cli", "claude-opus-5[1m]")] == "anthropic/claude-opus-5"
    assert aliases[("claude-cli", "claude-opus-5")] == "custom/x"  # manual wins
    assert ("ollama", "qwen3:8b") not in aliases
    assert seed_model_aliases(repo) == 0  # idempotent


def test_prune_calls_is_coarse_and_keeps_memory(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload(name="w", project="p")
    old = _call("p", "ollama", "m", ts=datetime.now(UTC) - timedelta(days=120), workload_id=wl.id)
    new = _call("p", "ollama", "m", workload_id=wl.id)
    repo.write_call(old)
    repo.write_call(new)
    repo.record_call_update(old.id, field="outcome", value="ok")

    preview = repo.prune_calls(older_than_days=90, dry_run=True)
    assert preview["calls"] == 1 and preview["call_updates"] == 1
    assert repo.get_call(old.id) is not None

    counts = repo.prune_calls(older_than_days=90)
    assert counts["calls"] == 1
    assert repo.get_call(old.id) is None and repo.get_call(new.id) is not None
    assert repo.workload_by_name("w", "p") is not None
    repo.vacuum()


def test_fleet_summary_reports_loop_state_per_ledger(tmp_path):
    mirror = Repository(tmp_path / "global.sqlite")
    mirror.write_call(_call("a", "ollama", "m"))
    project = Repository(tmp_path / "a" / ".somm" / "calls.sqlite")
    wl = project.register_workload(name="graded", project="a")
    project.set_shadow_config(wl.id, {"gold_provider": "ollama", "gold_model": "m"})
    with project._open() as conn:
        conn.execute(
            "INSERT INTO worker_heartbeat (worker_name, last_run_at, last_success_at, "
            "consecutive_failures) VALUES ('shadow_eval', '2026-09-19 10:00:00', "
            "'2026-09-19 10:00:00', 0)"
        )

    fleet = fleet_summary(mirror.db_path, since_days=7, ledgers=[project.db_path])

    assert fleet["loop"]["eval_results"] == 0  # the mirror never has these
    assert len(fleet["ledgers"]) == 1
    state = fleet["ledgers"][0]
    assert state["shadow_configured_workloads"] == 1
    assert state["last_worker_success"] == "2026-09-19 10:00:00"
    assert state["recommendations_open"] == 0


def _decision(*, workload_id=None, recalled_from=None) -> Decision:
    return Decision(
        id=str(uuid.uuid4()), ts=datetime.now(UTC), project="p", question="q",
        question_hash="h", candidates=[], rationale="r", chosen_provider="ollama",
        chosen_model="m", workload_id=workload_id, recalled_from=recalled_from,
    )


def test_north_star_on_an_empty_ledger_says_why_nothing_is_measurable(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    repo.write_call(_call("p", "ollama", "m"))  # no routing_json at all

    ns = north_star(repo.db_path)

    assert not any(v["measurable"] for v in ns.values())
    assert ns["decisions_recalled"]["reason"] == "no decisions recorded"
    assert "none of 1 production calls" in ns["evidence_routed"]["reason"]
    assert ns["evidence_routed"]["unmarked"] == 1


def test_north_star_counts_recalls_routes_and_recommendations(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    live = repo.register_workload(name="live", project="p")
    residue = repo.register_workload(name="fixture", project="p")
    route = lambda source: json.dumps({"route_source": source})  # noqa: E731
    repo.write_call(_call("p", "ollama", "m", workload_id=live.id, routing_json=route("policy")))
    repo.write_call(_call("p", "ollama", "m", routing_json=route("learned_override")))
    repo.write_call(_call("p", "ollama", "m", routing_json=route("pin")))
    repo.write_call(_call("p", "ollama", "m", routing_json=route("default")))
    repo.write_call(_call("p", "ollama", "m"))  # predates the marker: unknown, not a pin
    repo.write_call(_call("p", "ollama", "m", routing_json=route("policy"),
                          observation_role="shadow_gold"))  # grading, not production

    first = _decision()
    repo.record_decision(first)
    repo.record_decision(_decision(workload_id=live.id, recalled_from=first.id))
    for _ in range(3):  # residue: bound to a workload that never made a call here
        repo.record_decision(_decision(workload_id=residue.id, recalled_from=first.id))

    with repo._open() as conn:
        for created, applied, dismissed in [
            ("-30 days", "-25 days", None),  # applied in 5 days
            ("-30 days", None, "-1 days"),   # dismissed after 29 days: too late
            ("-20 days", None, None),        # open past the window
            ("-2 days", None, "-1 days"),    # young but decided: counts
            ("-2 days", None, None),         # young and open: pending
        ]:
            conn.execute(
                "INSERT INTO recommendations (workload_id, action, evidence_json, created_at, "
                "applied_at, dismissed_at) VALUES (?, 'switch_model', '{}', datetime('now', ?), "
                "datetime('now', ?), datetime('now', ?))",
                (live.id, created, applied, dismissed),
            )

    ns = north_star(repo.db_path)

    assert ns["decisions_recalled"] == {
        "measurable": True, "recalled": 1, "decisions": 2, "excluded": 3
    }
    routed = ns["evidence_routed"]
    assert (routed["evidence"], routed["marked"], routed["unmarked"]) == (2, 4, 1)
    assert routed["by_source"] == {"policy": 1, "learned_override": 1, "pin": 1, "default": 1}
    recs = ns["recommendations_applied"]
    assert (recs["acted"], recs["eligible"], recs["pending"], recs["applied"]) == (2, 4, 1, 1)


def test_north_star_degrades_on_a_ledger_without_the_new_columns(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE decisions (id TEXT, workload_id TEXT)")
        conn.execute("CREATE TABLE calls (id TEXT, ts TEXT, workload_id TEXT)")
        conn.execute("CREATE TABLE recommendations (id INTEGER, created_at TEXT, "
                     "applied_at TEXT, dismissed_at TEXT)")

    ns = north_star(db)

    assert "predates recall tracking" in ns["decisions_recalled"]["reason"]
    assert ns["evidence_routed"]["measurable"] is False
    assert ns["recommendations_applied"]["eligible"] == 0

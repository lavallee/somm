"""somm_core.read — one query behind every surface; somm_core.aliases — rules."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from somm_core.aliases import canonicalize, seed_model_aliases
from somm_core.models import Call, Outcome
from somm_core.read import fleet_summary, search_calls, spend_today
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

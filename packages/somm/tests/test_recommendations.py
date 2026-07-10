from __future__ import annotations

import json

from somm.cli import main
from somm.recommendations import apply_recommendation, dismiss_recommendation
from somm_core.repository import Repository


def _seed_switch_recommendation(repo: Repository, project: str = "recs"):
    wl = repo.register_workload(name="claims", project=project)
    evidence = {
        "workload": "claims",
        "current": {"provider": "ollama", "model": "slow"},
        "candidate": {"provider": "openrouter", "model": "fast"},
        "score_delta": 0.4,
        "n_evals": 8,
    }
    with repo._open() as conn:
        rec_id = conn.execute(
            "INSERT INTO recommendations "
            "(workload_id, action, evidence_json, expected_impact, confidence) "
            "VALUES (?, 'switch_model', ?, '+40% quality', 0.88)",
            (wl.id, json.dumps(evidence)),
        ).lastrowid
    return wl, int(rec_id)


def _seed_adaptive_recommendation(repo: Repository, project: str = "recs"):
    wl = repo.register_workload(name="claims", project=project)
    evidence = {
        "workload": "claims",
        "provider": "minimax",
        "model": "MiniMax-M2.7",
        "failure_signature": "capability_empty:stripped_empty",
        "recommended_max_tokens_floor": 8192,
        "empty_rate": 0.8,
    }
    with repo._open() as conn:
        rec_id = conn.execute(
            "INSERT INTO recommendations "
            "(workload_id, action, evidence_json, expected_impact, confidence) "
            "VALUES (?, 'adaptive_param_bump', ?, 'raise max token floor', 0.75)",
            (wl.id, json.dumps(evidence)),
        ).lastrowid
    return wl, int(rec_id)


def test_apply_recommendation_updates_policy_records_decision_and_applies(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    mirror = Repository(tmp_path / "global.sqlite")
    wl, rec_id = _seed_switch_recommendation(repo)

    result = apply_recommendation(
        repo,
        rec_id,
        actor="test",
        mirror_repo=mirror,
    )

    refreshed = repo.workload_by_name("claims", "recs")
    assert refreshed is not None
    assert refreshed.policy == {
        "fallback": [
            {"provider": "openrouter", "model": "fast"},
            {"provider": "ollama", "model": "slow"},
        ]
    }
    assert result.revision == 2
    assert result.policy == refreshed.policy
    assert result.decision.chosen_provider == "openrouter"
    assert result.decision.chosen_model == "fast"

    with repo._open() as conn:
        applied_at = conn.execute(
            "SELECT applied_at FROM recommendations WHERE id = ?",
            (rec_id,),
        ).fetchone()[0]
    assert applied_at is not None
    assert repo.search_decisions(workload="claims")[0].id == result.decision.id
    assert mirror.search_decisions(workload="claims")[0].id == result.decision.id


def test_dismiss_recommendation_closes_without_policy_change(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, rec_id = _seed_switch_recommendation(repo)

    rec = dismiss_recommendation(repo, rec_id)

    assert rec.dismissed_at is not None
    assert repo.workload_by_name("claims", "recs").policy is None


def test_apply_adaptive_recommendation_writes_learned_override(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, rec_id = _seed_adaptive_recommendation(repo)

    result = apply_recommendation(repo, rec_id, actor="test")

    override = repo.lookup_learned_override(wl.id, "MiniMax-M2.7", "minimax")
    assert override is not None
    assert override["max_tokens_floor"] == 8192
    assert result.recommendation.applied_at is not None
    assert result.decision.chosen_provider == "minimax"
    assert result.decision.chosen_model == "MiniMax-M2.7"


def test_inbox_cli_lists_and_applies(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    _wl, rec_id = _seed_switch_recommendation(repo, project="cli-recs")

    list_rc = main(["inbox", "list", "--project", "cli-recs"])
    list_out = capsys.readouterr().out
    assert list_rc == 0
    assert "claims" in list_out
    assert "switch_model" in list_out

    apply_rc = main(["inbox", "apply", str(rec_id), "--project", "cli-recs"])
    apply_out = capsys.readouterr().out
    assert apply_rc == 0
    assert f"applied recommendation {rec_id}" in apply_out
    assert "decision_id:" in apply_out

    refreshed = repo.workload_by_name("claims", "cli-recs")
    assert refreshed.policy["fallback"][0] == {"provider": "openrouter", "model": "fast"}

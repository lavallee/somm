from __future__ import annotations

import uuid
from datetime import UTC, datetime

from somm.cli import main
from somm.optimize import propose_prompt_optimization
from somm.prompts import get_label, register_prompt, set_label
from somm_core.models import Call, Outcome, SommResult
from somm_core.repository import Repository


def _seed_failing_prompt_case(repo: Repository, project: str = "opt"):
    wl = repo.register_workload(name="claims", project=project)
    source = register_prompt(repo, wl.id, "Extract claims.")
    set_label(repo, wl.id, "production", source.id)
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=project,
            workload_id=wl.id,
            prompt_id=source.id,
            provider="fake",
            model="old",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )
    repo.write_sample(call_id, "Extract claims.\nInput: text", "not json")
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO eval_results "
            "(call_id, gold_model, structural_score, embedding_score, judge_score) "
            "VALUES (?, 'dataset:golden', 0.2, 0.2, 0.2)",
            (call_id,),
        )
    return wl, source


def test_propose_prompt_optimization_forks_and_labels_proposed(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, source = _seed_failing_prompt_case(repo)
    seen_prompt = {}

    def proposer(prompt: str) -> str:
        seen_prompt["body"] = prompt
        return (
            '{"proposed_prompt": "Extract claims as strict JSON.", '
            '"rationale": "The failure was not valid JSON."}'
        )

    result = propose_prompt_optimization(
        repo,
        workload_id=wl.id,
        from_ref="production",
        proposer=proposer,
        threshold=0.8,
    )

    assert "Failing cases" in seen_prompt["body"]
    assert result.source_prompt.id == source.id
    assert result.proposed_prompt.parent_prompt_id == source.id
    assert result.proposed_prompt.body == "Extract claims as strict JSON."
    assert result.rationale == "The failure was not valid JSON."
    assert get_label(repo, wl.id, "proposed").id == result.proposed_prompt.id
    assert get_label(repo, wl.id, "production").id == source.id


def test_propose_prompt_optimization_requires_failing_cases(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl = repo.register_workload(name="claims", project="opt")
    source = register_prompt(repo, wl.id, "Extract claims.")
    set_label(repo, wl.id, "production", source.id)

    try:
        propose_prompt_optimization(
            repo,
            workload_id=wl.id,
            from_ref="production",
            proposer=lambda _prompt: '{"proposed_prompt": "new"}',
        )
    except ValueError as exc:
        assert "no sampled graded calls" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_optimize_cli_creates_proposed_label(tmp_path, capsys, monkeypatch):
    import somm as somm_pkg

    monkeypatch.chdir(tmp_path)
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    wl, source = _seed_failing_prompt_case(repo, project="opt-cli")

    class _FakeOptimizeLLM:
        def __init__(self, config):
            self.config = config

        def generate(self, **_kwargs):
            return SommResult(
                text=(
                    '{"proposed_prompt": "Extract claims as JSON with citations.", '
                    '"rationale": "Add explicit JSON and citation requirements."}'
                ),
                provider="fake",
                model="optimizer",
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                cost_usd=0.0,
                call_id=str(uuid.uuid4()),
            )

        def close(self):
            pass

    monkeypatch.setattr(somm_pkg, "SommLLM", _FakeOptimizeLLM)

    rc = main(["optimize", "--workload", "claims", "--project", "opt-cli"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "proposed ->" in out
    assert "rationale:" in out
    proposed = get_label(repo, wl.id, "proposed")
    assert proposed.body == "Extract claims as JSON with citations."
    assert proposed.parent_prompt_id == source.id
    assert get_label(repo, wl.id, "production").id == source.id

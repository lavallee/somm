from __future__ import annotations

import uuid
from datetime import UTC, datetime

from somm.cli import main
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


def _prompt_id(repo: Repository, workload_id: str, version: str) -> str:
    with repo._open() as conn:
        row = conn.execute(
            "SELECT id FROM prompts WHERE workload_id = ? AND version = ?",
            (workload_id, version),
        ).fetchone()
    assert row is not None
    return row[0]


def test_prompt_cli_lifecycle_score_and_promote(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = "prompt-cli"
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    workload = repo.register_workload(name="claims", project=project)

    body1 = tmp_path / "prompt-v1.txt"
    body2 = tmp_path / "prompt-v2.txt"
    fork_body = tmp_path / "prompt-fork.txt"
    body1.write_text("extract claim facts\n")
    body2.write_text("extract claim facts as json\n")
    fork_body.write_text("extract claim facts carefully\n")

    assert (
        main(
            [
                "prompt",
                "register",
                "--workload",
                "claims",
                "--project",
                project,
                "--body-file",
                str(body1),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "registered v1" in out

    assert (
        main(
            [
                "prompt",
                "register",
                "--workload",
                "claims",
                "--project",
                project,
                "--body-file",
                str(body2),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["prompt", "list", "--workload", "claims", "--project", project]) == 0
    out = capsys.readouterr().out
    assert "Workload: claims" in out
    assert "v1" in out
    assert "v1.1" in out
    assert "latest" in out

    assert (
        main(
            [
                "prompt",
                "show",
                "--workload",
                "claims",
                "--project",
                project,
                "--version",
                "v1",
                "--full",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "version: v1" in out
    assert "extract claim facts" in out

    assert (
        main(
            [
                "prompt",
                "diff",
                "--workload",
                "claims",
                "--project",
                project,
                "v1",
                "v1.1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "--- v1 (v1)" in out
    assert "+++ v1.1 (v1.1)" in out
    assert "-extract claim facts" in out
    assert "+extract claim facts as json" in out

    assert (
        main(
            [
                "prompt",
                "label",
                "--workload",
                "claims",
                "--project",
                project,
                "--label",
                "production",
                "--weights",
                "v1=90,v1.1=10",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "production ->" in out
    assert "90.0%" in out
    assert "10.0%" in out

    assert (
        main(
            [
                "prompt",
                "show",
                "--workload",
                "claims",
                "--project",
                project,
                "--label",
                "production",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "version: v1" in out

    assert (
        main(
            [
                "prompt",
                "fork",
                "--workload",
                "claims",
                "--project",
                project,
                "--from",
                "production",
                "--body-file",
                str(fork_body),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "forked" in out
    assert "parent_prompt_id:" in out

    p1_id = _prompt_id(repo, workload.id, "v1")
    for structural, embedding, judge in ((0.8, 0.7, 0.6), (1.0, 0.9, 0.8)):
        call_id = str(uuid.uuid4())
        repo.write_call(
            Call(
                id=call_id,
                ts=datetime.now(UTC),
                project=project,
                workload_id=workload.id,
                prompt_id=p1_id,
                provider="fake",
                model="fake-m",
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                cost_usd=0.0,
                outcome=Outcome.OK,
                error_kind=None,
                prompt_hash="a",
                response_hash="b",
            )
        )
        with repo._open() as conn:
            conn.execute(
                "INSERT INTO eval_results "
                "(call_id, gold_model, structural_score, embedding_score, judge_score) "
                "VALUES (?, 'gold', ?, ?, ?)",
                (call_id, structural, embedding, judge),
            )

    assert (
        main(
            [
                "prompt",
                "score",
                "--workload",
                "claims",
                "--project",
                project,
                "--version",
                "v1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "v1" in out
    assert "       2" in out
    assert "0.900" in out
    assert "0.800" in out
    assert "0.700" in out

    rc = main(
        [
            "prompt",
            "promote",
            "--workload",
            "claims",
            "--project",
            project,
            "--version",
            "v1",
            "--to",
            "staging",
            "--min-graded",
            "3",
            "--min-score",
            "0.7",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "promotion gate failed" in err
    assert "graded=2" in err

    assert (
        main(
            [
                "prompt",
                "promote",
                "--workload",
                "claims",
                "--project",
                project,
                "--version",
                "v1",
                "--to",
                "staging",
                "--min-graded",
                "3",
                "--min-score",
                "0.7",
                "--force",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "staging -> v1" in out
    assert "forced: yes" in out

"""`--json` on the commands that print a list: workload list, prompt list, frontier, compare."""

from __future__ import annotations

import json

from somm.cli import main
from somm_core.repository import Repository


def test_workload_list_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    repo.register_workload(name="claims", project="jp")
    assert main(["workload", "list", "--project", "jp", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in rows] == ["claims"]
    assert main(["workload", "list", "--project", "empty", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_prompt_list_json_carries_labels_not_bodies(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Repository(tmp_path / ".somm" / "calls.sqlite").register_workload(name="claims", project="jp")
    body = tmp_path / "p.txt"
    body.write_text("secret body")
    base = ["prompt", "--project", "jp"]
    assert main([*base, "register", "--workload", "claims", "--body-file", str(body)]) == 0
    assert main([*base, "promote", "--workload", "claims", "--version", "v1", "--to", "production",
                 "--force"]) == 0
    capsys.readouterr()
    assert main([*base, "list", "--workload", "claims", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["version"] == "v1" and "production" in rows[0]["labels"]
    assert "body" not in rows[0]


def test_frontier_json_empty_window(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Repository(tmp_path / ".somm" / "calls.sqlite").register_workload(name="claims", project="jp")
    assert main(["frontier", "--workload", "claims", "--project", "jp", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_compare_json_reports_unconfigured_provider(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["compare", "hi", "--models", "nonesuch/m", "--project", "jp", "--json"])
    assert rc == 1  # no model answered
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["provider"] == "nonesuch" and "not configured" in rows[0]["error"]

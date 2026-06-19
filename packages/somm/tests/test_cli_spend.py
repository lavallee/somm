"""Tests for `somm spend` subcommand."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm.cli import _cmd_spend, main, spend_today
from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


def _cfg(tmp_path: Path, project: str = "spend-test") -> Config:
    cfg = Config()
    cfg.project = project
    cfg.db_dir = tmp_path / ".somm"
    return cfg


def _call(project: str, workload_id: str | None, cost_usd: float) -> Call:
    return Call(
        id=str(uuid.uuid4()),
        ts=datetime.now(UTC),
        project=project,
        workload_id=workload_id,
        prompt_id=None,
        provider="fake",
        model="m",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=cost_usd,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="a",
        response_hash="b",
    )


# ---------------------------------------------------------------------------
# spend_today unit tests


def test_spend_today_sums_per_workload(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)

    wl1 = repo.register_workload(
        name="summarize", project=cfg.project, budget_cap_usd_daily=1.00
    )
    wl2 = repo.register_workload(
        name="classify", project=cfg.project, budget_cap_usd_daily=0.50
    )

    repo.write_call(_call(cfg.project, wl1.id, 0.10))
    repo.write_call(_call(cfg.project, wl1.id, 0.05))
    repo.write_call(_call(cfg.project, wl2.id, 0.20))

    rows = spend_today(cfg.db_path, cfg.project, default_cap=None)

    assert len(rows) == 2
    by_name = {r["workload"]: r for r in rows}

    assert abs(by_name["summarize"]["spent_usd"] - 0.15) < 1e-9
    assert abs(by_name["summarize"]["cap_usd"] - 1.00) < 1e-9

    assert abs(by_name["classify"]["spent_usd"] - 0.20) < 1e-9
    assert abs(by_name["classify"]["cap_usd"] - 0.50) < 1e-9

    # sorted by spent_usd desc → classify ($0.20) before summarize ($0.15)
    assert rows[0]["workload"] == "classify"


def test_spend_today_default_cap_fallback(tmp_path):
    cfg = _cfg(tmp_path, "spend-default")
    repo = Repository(cfg.db_path)

    wl = repo.register_workload(name="no-cap", project=cfg.project)
    repo.write_call(_call(cfg.project, wl.id, 0.30))

    rows = spend_today(cfg.db_path, cfg.project, default_cap=2.00)

    assert len(rows) == 1
    assert rows[0]["workload"] == "no-cap"
    assert abs(rows[0]["cap_usd"] - 2.00) < 1e-9


def test_spend_today_no_cap_returns_none(tmp_path):
    cfg = _cfg(tmp_path, "spend-nocap")
    repo = Repository(cfg.db_path)

    wl = repo.register_workload(name="uncapped", project=cfg.project)
    repo.write_call(_call(cfg.project, wl.id, 0.05))

    rows = spend_today(cfg.db_path, cfg.project, default_cap=None)

    assert len(rows) == 1
    assert rows[0]["cap_usd"] is None


def test_spend_today_empty_db_returns_empty(tmp_path):
    cfg = _cfg(tmp_path, "spend-empty")
    Repository(cfg.db_path)  # create schema, no rows

    rows = spend_today(cfg.db_path, cfg.project, default_cap=None)

    assert rows == []


# ---------------------------------------------------------------------------
# _cmd_spend / main integration tests


def test_cmd_spend_no_db_prints_friendly_message(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "spend-nodb")

    rc = main(["spend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no spend recorded today" in out


def test_cmd_spend_empty_rows_prints_friendly_message(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path, "spend-norows")
    Repository(cfg.db_path)  # schema only, no calls

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    rc = main(["spend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no spend recorded today" in out


def test_cmd_spend_prints_table(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path, "spend-table")
    repo = Repository(cfg.db_path)

    wl = repo.register_workload(
        name="my-task", project=cfg.project, budget_cap_usd_daily=1.00
    )
    repo.write_call(_call(cfg.project, wl.id, 0.25))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    rc = main(["spend"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "my-task" in out
    assert "0.25" in out
    assert "1.00" in out
    assert "25.0%" in out


def test_cmd_spend_no_cap_shows_dash(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path, "spend-nocap-cmd")
    repo = Repository(cfg.db_path)

    wl = repo.register_workload(name="free-wl", project=cfg.project)
    repo.write_call(_call(cfg.project, wl.id, 0.01))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    rc = main(["spend"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "free-wl" in out
    assert "—" in out

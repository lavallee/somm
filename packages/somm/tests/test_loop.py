"""`somm loop` — the intelligence loop as a one-shot command and a timer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm import loop
from somm.cli import main
from somm_core.config import Config
from somm_core.repository import Repository


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "loop-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


class _Worker:
    def __init__(self, name: str, log: list[str], *, fail: bool = False) -> None:
        self.name, self.log, self.fail = name, log, fail

    def run_once(self) -> dict:
        self.log.append(self.name)
        if self.fail:
            raise RuntimeError("boom")
        return {"ok": True, "errors": []}


def test_run_once_runs_due_workers_and_records_heartbeats(tmp_path):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path)
    log: list[str] = []

    report = loop.run_once(cfg, factory=lambda n: _Worker(n, log))

    assert log == ["model_intel", "shadow_eval", "agent"]
    assert set(report.ran) == {"model_intel", "shadow_eval", "agent"}
    repo = Repository(cfg.db_path)
    assert loop.last_success(repo, "shadow_eval") is not None

    # A second pass an hour later: intel and agent are fresh, grading runs again.
    log.clear()
    later = datetime.now(UTC) + timedelta(hours=1)
    report = loop.run_once(cfg, factory=lambda n: _Worker(n, log), now=later)
    assert log == ["shadow_eval"]
    assert set(report.skipped) == {"model_intel", "agent"}
    assert "ago" in report.skipped["model_intel"]

    # --force overrides staleness; one failing worker does not stop the rest.
    log.clear()
    report = loop.run_once(
        cfg, force=True, factory=lambda n: _Worker(n, log, fail=(n == "model_intel")), now=later
    )
    assert log == ["model_intel", "shadow_eval", "agent"]
    assert "RuntimeError" in report.errors["model_intel"]
    assert set(report.ran) == {"shadow_eval", "agent"}


def test_run_once_only_restricts(tmp_path):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path)
    log: list[str] = []
    loop.run_once(cfg, only=("agent",), factory=lambda n: _Worker(n, log))
    assert log == ["agent"]


def test_install_writes_units_and_enables_timer(tmp_path):
    cfg = _cfg(tmp_path)
    calls: list[list[str]] = []

    written = loop.install(cfg, every="30min", target_dir=tmp_path / "units", systemctl=calls.append)

    names = sorted(p.name for p in written)
    assert names == ["10-somm.conf", "somm-loop@.service", "somm-loop@.timer"]
    service = (tmp_path / "units" / "somm-loop@.service").read_text()
    assert "loop run --project %i" in service
    # The template is shared across projects; the ledger location is per project.
    assert "SOMM_DB_DIR" not in service
    dropin = (tmp_path / "units" / "somm-loop@loop-test.service.d" / "10-somm.conf").read_text()
    assert f"SOMM_DB_DIR={cfg.db_dir.resolve()}" in dropin
    timer = (tmp_path / "units" / "somm-loop@.timer").read_text()
    assert "OnUnitActiveSec=30min" in timer
    assert calls == [["daemon-reload"], ["enable", "--now", "somm-loop@loop-test.timer"]]


def test_loop_cli_run_and_status(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path)
    Repository(cfg.db_path)
    monkeypatch.setattr("somm.cli.load_config", lambda project=None: cfg)
    log: list[str] = []
    monkeypatch.setattr(loop, "build_workers_factory", lambda c, r: (lambda n: _Worker(n, log)))
    monkeypatch.setattr(loop, "_systemctl", lambda args, capture=False: "inactive\n")

    assert main(["loop", "run"]) == 0
    out = capsys.readouterr().out
    assert "ran      shadow_eval" in out and log == ["model_intel", "shadow_eval", "agent"]

    assert main(["loop", "status"]) == 0
    out = capsys.readouterr().out
    assert "somm-loop@loop-test.timer" in out and "last success" in out
    assert main(["loop"]) == 0  # default is status


def test_loop_cli_accepts_project_after_the_subcommand(tmp_path, capsys, monkeypatch):
    seen = {}

    def fake_load(project=None):
        seen["project"] = project
        cfg = _cfg(tmp_path)
        cfg.project = project or cfg.project
        return cfg

    monkeypatch.setattr("somm.cli.load_config", fake_load)
    monkeypatch.setattr(loop, "_systemctl", lambda args, capture=False: "inactive\n")
    Repository(_cfg(tmp_path).db_path)

    assert main(["loop", "status", "--project", "after"]) == 0
    assert seen["project"] == "after"
    assert main(["loop", "--project", "before", "status"]) == 0
    assert seen["project"] == "before"


def test_install_prefers_the_somm_beside_the_interpreter(tmp_path, monkeypatch):
    import sys

    fake_bin = tmp_path / "venv" / "bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "somm").write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    assert loop.somm_executable() == str(fake_bin / "somm")
    cfg = _cfg(tmp_path)
    loop.install(cfg, target_dir=tmp_path / "units", systemctl=lambda a: None)
    service = (tmp_path / "units" / "somm-loop@.service").read_text()
    assert f"ExecStart={fake_bin / 'somm'} loop run --project %i" in service

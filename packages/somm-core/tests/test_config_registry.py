"""`--project X` from outside X's checkout reads X's registered ledger."""

from __future__ import annotations

from somm_core import registry
from somm_core.config import load


def _registered_ledger(tmp_path, monkeypatch, project="research-app"):
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("SOMM_REGISTRY_ALLOW_TMP", "1")
    monkeypatch.setattr(registry, "_registered_projects", set())
    monkeypatch.delenv("SOMM_DB_DIR", raising=False)
    monkeypatch.delenv("SOMM_PROJECT", raising=False)
    ledger = tmp_path / "elsewhere" / ".somm" / "calls.sqlite"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("")
    registry.register_project(project, ledger)
    cwd = tmp_path / "other-checkout"
    cwd.mkdir()
    return ledger, cwd


def test_cli_project_resolves_the_registered_ledger(tmp_path, monkeypatch):
    ledger, cwd = _registered_ledger(tmp_path, monkeypatch)
    cfg = load(project="research-app", cwd=cwd, prefer_registry=True)
    assert cfg.project == "research-app"
    assert cfg.db_path == ledger.resolve()


def test_library_callers_keep_the_local_ledger(tmp_path, monkeypatch):
    """`somm.llm(project=...)` from a checkout that has a ledger writes that
    ledger; only the CLI prefers the registry. (A checkout with no ledger at
    all has always fallen back to the registered one — see `load`.)"""
    ledger, cwd = _registered_ledger(tmp_path, monkeypatch)
    (cwd / ".somm").mkdir()
    assert load(project="research-app", cwd=cwd).db_path == (cwd / ".somm" / "calls.sqlite").resolve()
    assert load(project="research-app", cwd=cwd, prefer_registry=True).db_path == ledger.resolve()


def test_an_explicit_ledger_location_wins_over_the_registry(tmp_path, monkeypatch):
    ledger, cwd = _registered_ledger(tmp_path, monkeypatch)
    pinned = tmp_path / "pinned"
    monkeypatch.setenv("SOMM_DB_DIR", str(pinned))
    cfg = load(project="research-app", cwd=cwd, prefer_registry=True)
    assert cfg.db_path == (pinned / "calls.sqlite").resolve()


def test_an_unregistered_project_falls_back_to_the_local_ledger(tmp_path, monkeypatch):
    _ledger, cwd = _registered_ledger(tmp_path, monkeypatch)
    cfg = load(project="nobody-registered-this", cwd=cwd, prefer_registry=True)
    assert cfg.db_path == (cwd / ".somm" / "calls.sqlite").resolve()


def test_direct_config_honours_global_path_override(tmp_path, monkeypatch):
    """A hand-built Config must not fall through to ~/.somm/global.sqlite."""
    from somm_core.config import Config

    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(tmp_path / "g.sqlite"))
    assert Config().global_db_path == tmp_path / "g.sqlite"

    explicit = Config()
    explicit.cross_project_path = tmp_path / "explicit.sqlite"
    assert explicit.global_db_path == tmp_path / "explicit.sqlite"


def test_suite_never_resolves_the_real_global_ledger():
    """The repo conftest points SOMM_GLOBAL_PATH at a temp dir for every test."""
    from pathlib import Path

    from somm_core.config import Config

    assert Path.home() / ".somm" not in Config().global_db_path.parents

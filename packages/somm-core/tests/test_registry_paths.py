"""The registry keeps every ledger a project writes to, and reads the old format."""

from __future__ import annotations

import json

import pytest
from somm_core import registry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(path))
    monkeypatch.setenv("SOMM_REGISTRY_ALLOW_TMP", "1")
    monkeypatch.setattr(registry, "_registered_projects", set())
    return path


def _ledger(tmp_path, name):
    p = tmp_path / name / ".somm" / "calls.sqlite"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    return p.resolve()


def test_a_project_writing_to_two_ledgers_keeps_both(reg, tmp_path):
    first, second = _ledger(tmp_path, "a"), _ledger(tmp_path, "b")
    registry.register_project("proj", first)
    registry.register_project("proj", second)

    entry = json.loads(reg.read_text())["projects"]["proj"]
    assert entry["db_path"] == str(second)  # older releases read the newest path
    assert set(entry["db_paths"]) == {str(first), str(second)}
    assert sorted(p for _, p in registry.registered_ledgers()) == sorted([first, second])
    assert registry.registered_db_path("proj") == second
    assert set(registry.fleet_db_paths()) == {first, second}


def test_the_old_single_path_format_still_reads(reg, tmp_path):
    old = _ledger(tmp_path, "old")
    reg.write_text(json.dumps({"projects": {"p": {"db_path": str(old), "last_seen": "t"}}}))

    assert registry.registry_entries() == {"p": {str(old): "t"}}
    assert registry.registered_db_path("p") == old

    new = _ledger(tmp_path, "new")
    registry.register_project("p", new)
    assert set(registry.registry_entries()["p"]) == {str(old), str(new)}


def test_a_vanished_path_is_dropped_but_the_project_kept(reg, tmp_path):
    keep, gone = _ledger(tmp_path, "keep"), _ledger(tmp_path, "gone")
    registry.register_project("p", keep)
    registry.register_project("p", gone)
    gone.unlink()

    assert registry.registered_db_path("p") == keep
    assert registry.prune_registry() == 1
    entry = json.loads(reg.read_text())["projects"]["p"]
    assert entry["db_path"] == str(keep)
    assert list(entry["db_paths"]) == [str(keep)]


def test_forget_and_adopt_change_only_the_registry(reg, tmp_path):
    a, b = _ledger(tmp_path, "a"), _ledger(tmp_path, "b")
    registry.register_project("p", a)

    assert registry.adopt_ledger("p", b, dry_run=True) is True
    assert set(registry.registry_entries()["p"]) == {str(a)}
    assert registry.adopt_ledger("p", b) is True
    assert registry.adopt_ledger("p", b) is False
    # Adopting keeps the current path: `--project p` still reads `a`.
    assert registry.registered_db_path("p") == a

    assert registry.forget_project("p", path=a, dry_run=True) == [str(a)]
    assert set(registry.registry_entries()["p"]) == {str(a), str(b)}
    assert registry.forget_project("p", path=a) == [str(a)]
    assert registry.registered_db_path("p") == b
    assert registry.forget_project("p") == [str(b)]
    assert "p" not in registry.registry_entries()
    assert a.exists() and b.exists()  # files are never touched

    with pytest.raises(KeyError):
        registry.forget_project("p")
    with pytest.raises(FileNotFoundError):
        registry.adopt_ledger("p", tmp_path / "missing.sqlite")

"""Registry read-modify-write must not lose entries under concurrency.

The registry is a single shared JSON file. Two projects initializing at
once each load, mutate their own entry, and write back; without a
cross-process lock the last writer silently drops the other's entry and
fleet-wide plan pacing then paces from an incomplete fleet.
"""

from __future__ import annotations

import json
import threading

from somm_core import registry


def _make_db(tmp_path, name):
    db = tmp_path / f"{name}.sqlite"
    db.write_text("", encoding="utf-8")  # register_project only checks existence
    return db


def test_concurrent_registration_keeps_all_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("SOMM_REGISTRY_ALLOW_TMP", "1")
    # Fresh process-level dedup set so every thread actually writes.
    monkeypatch.setattr(registry, "_registered_projects", set())

    projects = [(f"proj{i}", _make_db(tmp_path, f"proj{i}")) for i in range(12)]
    barrier = threading.Barrier(len(projects))

    def worker(name, db):
        barrier.wait()  # maximize overlap on the shared file
        registry.register_project(name, db)

    threads = [threading.Thread(target=worker, args=(n, d)) for n, d in projects]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert set(data["projects"]) == {n for n, _ in projects}

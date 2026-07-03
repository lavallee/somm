"""Repo-wide test isolation.

SommLLM init touches machine-wide state under ~/.somm (the fleet
registry, plans.toml). Point both at per-test temp files so the suite
never reads or writes the developer's real fleet metadata.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_somm_machine_state(tmp_path_factory, monkeypatch):
    d = tmp_path_factory.mktemp("somm-home")
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(d / "registry.json"))
    monkeypatch.setenv("SOMM_PLANS_PATH", str(d / "plans.toml"))

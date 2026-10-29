"""Repo-wide test isolation.

SommLLM init touches machine-wide state under ~/.somm (the fleet
registry, plans.toml). Point both at per-test temp files so the suite
never reads or writes the developer's real fleet metadata.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_somm_machine_state(tmp_path_factory, monkeypatch):
    # The developer's own SOMM_* policy must not leak in: an operator shell
    # exporting SOMM_BUDGET_DEFAULT_CAP_USD_DAILY gave every "no cap" test a
    # cap. Tests that need a SOMM_* variable set it themselves.
    for name in [k for k in os.environ if k.startswith("SOMM_")]:
        monkeypatch.delenv(name, raising=False)
    d = tmp_path_factory.mktemp("somm-home")
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(d / "registry.json"))
    monkeypatch.setenv("SOMM_PLANS_PATH", str(d / "plans.toml"))
    # Cross-project mirroring defaults on; without these two a test's
    # SommLLM would replicate its calls into the developer's real
    # ~/.somm/global.sqlite. Tests that exercise mirroring set their own.
    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(d / "global.sqlite"))
    monkeypatch.setenv("SOMM_CROSS_PROJECT", "0")

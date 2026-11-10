"""Two projects that declare the same workload share one workload id.

The id is content-addressed on name + schemas, not project, so the mirror's
``workloads`` table holds one row for both, owned by whichever project
registered first. Calls carry their own ``project``; every global read must
attribute by ``calls.project`` and use the workload row only for what the id
determines (the name and schemas).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm.cli import _workloads_without_output_schema, main
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


class _Fake:
    name = "fake"

    def generate(self, request):
        return SommResponse(text="ok", model="fake-m", tokens_in=1, tokens_out=1, latency_ms=1)

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path, project: str, mirror: Path) -> Config:
    cfg = Config()
    cfg.project = project
    cfg.db_dir = tmp_path / project / ".somm"
    cfg.cross_project_enabled = True
    cfg.cross_project_path = mirror
    return cfg


def _calls(repo: Repository, project: str, workload_id: str, n: int) -> None:
    repo.write_calls_batch([
        Call(
            id=str(uuid.uuid4()), ts=datetime.now(UTC), project=project,
            workload_id=workload_id, prompt_id=None, provider="fake", model="fake-m",
            tokens_in=1, tokens_out=1, latency_ms=1, cost_usd=0.0, outcome=Outcome.OK,
            error_kind=None, prompt_hash="p", response_hash="r",
        )
        for _ in range(n)
    ])


def test_two_projects_with_one_workload_share_one_mirror_row(tmp_path):
    mirror = tmp_path / "global.sqlite"
    for project in ("first", "second"):
        llm = SommLLM(config=_cfg(tmp_path, project, mirror), providers=[_Fake()])
        try:
            llm.generate("hi", workload="same_event")
        finally:
            llm.close()

    with Repository(mirror)._open() as conn:
        rows = conn.execute("SELECT project FROM workloads WHERE name = 'same_event'").fetchall()
        callers = conn.execute(
            "SELECT project, COUNT(*) FROM calls GROUP BY project ORDER BY project"
        ).fetchall()
    # The collision itself (unchanged by design: ids are not re-computed).
    assert rows == [("first",)]
    assert callers == [("first", 1), ("second", 1)]


def _shared_mirror(tmp_path) -> Repository:
    mirror = Repository(tmp_path / "global.sqlite")
    wl = mirror.register_workload(name="same_event", project="small")
    mirror.register_workload(name="same_event", project="big")  # ignored: same id
    _calls(mirror, "small", wl.id, 120)
    _calls(mirror, "big", wl.id, 300)
    return mirror


def test_doctor_global_credits_calls_to_the_project_that_made_them(tmp_path, capsys, monkeypatch):
    mirror = _shared_mirror(tmp_path)

    assert sorted(_workloads_without_output_schema(mirror)) == [
        ("big", "same_event", 300),
        ("small", "same_event", 120),
    ]

    cfg = Config()
    cfg.project = "small"
    cfg.db_dir = tmp_path / "small" / ".somm"
    cfg.cross_project_path = mirror.db_path
    monkeypatch.setattr("somm.cli._common.load_config", lambda project=None: cfg)
    monkeypatch.setattr(
        "somm.cli.OllamaProvider.health", lambda self: ProviderHealth(available=True, detail="ok")
    )
    main(["doctor", "--global"])
    out = capsys.readouterr().out
    lines = [ln.split() for ln in out.splitlines() if ".same_event" in ln]
    assert sorted(lines) == [["big.same_event", "300", "calls"], ["small.same_event", "120", "calls"]]


def test_global_stats_name_the_workload_for_every_project(tmp_path):
    mirror = _shared_mirror(tmp_path)

    rows = {(r["project"], r["workload"]): r["n_calls"] for r in mirror.stats_global_by_workload(1)}
    assert rows == {("small", "same_event"): 120, ("big", "same_event"): 300}

"""Cross-project mirror tests — D6.

Verifies:
- OFF by default (no mirror writes, no ~/.somm/global.sqlite touched).
- ON via config: calls replicate to mirror DB on batch write.
- Workload registrations mirror so global rollups show names.
- Mirror failures don't break primary writes.
- Two projects writing to the same mirror produce a cross-project view.
"""

from __future__ import annotations

from pathlib import Path

from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.repository import Repository


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "ok"):
        self._text = text

    def generate(self, request):
        return SommResponse(
            text=self._text,
            model=request.model or "fake-m",
            tokens_in=3,
            tokens_out=1,
            latency_ms=5,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path, project: str, mirror_path: Path | None = None) -> Config:
    cfg = Config()
    cfg.project = project
    cfg.db_dir = tmp_path / project / ".somm"
    if mirror_path is not None:
        cfg.cross_project_enabled = True
        cfg.cross_project_path = mirror_path
    return cfg


def test_mirror_off_by_default(tmp_path):
    """With default config, no mirror DB is created or written to."""
    cfg = _cfg(tmp_path, "p1")
    mirror = Path.home() / ".somm" / "global.sqlite"
    mirror_before = mirror.stat().st_mtime if mirror.exists() else 0

    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.generate("hi", workload="w1")
    finally:
        llm.close()

    # Mirror file unchanged (if it already existed from prior runs)
    mirror_after = mirror.stat().st_mtime if mirror.exists() else 0
    assert mirror_after == mirror_before


def test_mirror_on_replicates_calls(tmp_path):
    mirror_path = tmp_path / "global.sqlite"
    cfg = _cfg(tmp_path, "p1", mirror_path=mirror_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        r1 = llm.generate("hi", workload="w_mirror")
        r2 = llm.generate("hi2", workload="w_mirror")
    finally:
        llm.close()

    # Mirror file exists + has both calls
    assert mirror_path.exists()
    mirror = Repository(mirror_path)
    with mirror._open() as conn:
        rows = conn.execute("SELECT id, project FROM calls").fetchall()
    call_ids = {r[0] for r in rows}
    assert r1.call_id in call_ids
    assert r2.call_id in call_ids
    # project column populated
    for row in rows:
        assert row[1] == "p1"


def test_workloads_mirror_on_init(tmp_path):
    """Workloads registered before SommLLM init are mirrored on startup."""
    mirror_path = tmp_path / "global.sqlite"
    # Create the project DB + workload first
    cfg = _cfg(tmp_path, "p1", mirror_path=mirror_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="pre_existing", project="p1")

    # Instantiate SommLLM — mirror_workloads should replicate
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        pass
    finally:
        llm.close()

    mirror = Repository(mirror_path)
    with mirror._open() as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM workloads WHERE project = ?",
                ("p1",),
            ).fetchall()
        ]
    assert "pre_existing" in names


def test_two_projects_share_mirror(tmp_path):
    """Two projects both writing to the same mirror → cross-project view."""
    mirror_path = tmp_path / "global.sqlite"

    # Project A
    cfg_a = _cfg(tmp_path, "proj_a", mirror_path=mirror_path)
    llm_a = SommLLM(config=cfg_a, providers=[FakeProvider()])
    try:
        llm_a.generate("a1", workload="w_a")
        llm_a.generate("a2", workload="w_a")
    finally:
        llm_a.close()

    # Project B
    cfg_b = _cfg(tmp_path, "proj_b", mirror_path=mirror_path)
    llm_b = SommLLM(config=cfg_b, providers=[FakeProvider()])
    try:
        llm_b.generate("b1", workload="w_b")
    finally:
        llm_b.close()

    mirror = Repository(mirror_path)
    with mirror._open() as conn:
        rows = conn.execute("SELECT project, COUNT(*) FROM calls GROUP BY project").fetchall()
    counts = dict(rows)
    assert counts["proj_a"] == 2
    assert counts["proj_b"] == 1


def test_register_workload_via_client_mirrors(tmp_path):
    """SommLLM.register_workload() replicates to mirror."""
    mirror_path = tmp_path / "global.sqlite"
    cfg = _cfg(tmp_path, "p1", mirror_path=mirror_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        llm.register_workload(name="post_init_workload", description="desc")
    finally:
        llm.close()

    mirror = Repository(mirror_path)
    with mirror._open() as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM workloads WHERE project = ?",
                ("p1",),
            ).fetchall()
        ]
    assert "post_init_workload" in names


def test_env_flag_enables_mirror(tmp_path, monkeypatch):
    """SOMM_CROSS_PROJECT=1 + SOMM_GLOBAL_PATH enables via config loader."""
    mirror_path = tmp_path / "env-global.sqlite"
    monkeypatch.setenv("SOMM_CROSS_PROJECT", "1")
    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(mirror_path))
    monkeypatch.setenv("SOMM_PROJECT", "env-proj")
    monkeypatch.chdir(tmp_path)

    from somm_core.config import load

    cfg = load()
    assert cfg.cross_project_enabled is True
    assert cfg.global_db_path == mirror_path


def test_mirror_failure_doesnt_break_primary(tmp_path, monkeypatch):
    """If the mirror write raises, the primary call still succeeds."""
    mirror_path = tmp_path / "global.sqlite"
    cfg = _cfg(tmp_path, "p1", mirror_path=mirror_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    # Sabotage the mirror repo's write_calls_batch
    original = llm._mirror_repo.write_calls_batch

    def bad_write(*args, **kwargs):
        raise RuntimeError("simulated mirror failure")

    llm._mirror_repo.write_calls_batch = bad_write
    try:
        r = llm.generate("hi", workload="w")
    finally:
        # Restore so close() doesn't break
        llm._mirror_repo.write_calls_batch = original
        llm.close()

    # Primary call succeeded
    assert r.text == "ok"
    # Primary DB has the row
    primary = llm.repo.get_call(r.call_id)
    assert primary is not None

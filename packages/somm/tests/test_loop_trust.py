"""The intelligence loop's output must be trustworthy enough to act on.

Each test here first reproduced a defect: duplicate gold spend from a
non-atomic lease, an unparseable judge reply scored as 0.0, a broken gold
route consuming samples, a replay that was not the production request, and
recommendations with no statistical floor or expiry.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm.providers.base import ProviderHealth, SommResponse
from somm.workers.agent import AgentWorker
from somm.workers.shadow_eval import ShadowEvalWorker
from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


def _repo(tmp_path: Path) -> tuple[str, Repository]:
    cfg = Config()
    cfg.project = "trust"
    cfg.db_dir = tmp_path / ".somm"
    return cfg.project, Repository(cfg.db_path)


def _call(
    repo: Repository,
    workload_id: str,
    project: str,
    *,
    provider: str = "ollama",
    model: str = "m",
    cost_usd: float = 0.0,
    cost_source: str | None = None,
    latency_ms: int = 100,
    temperature: float | None = None,
    max_tokens: int | None = None,
    ts: datetime | None = None,
) -> str:
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=ts or datetime.now(UTC),
            project=project,
            workload_id=workload_id,
            prompt_id=None,
            provider=provider,
            model=model,
            tokens_in=10,
            tokens_out=5,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
            temperature=temperature,
            max_tokens=max_tokens,
            cost_source=cost_source,
        )
    )
    return call_id


class _Gold:
    def __init__(self, name: str = "gold", text: str = '{"ok": true}', fail: Exception | None = None):
        self.name = name
        self.text = text
        self.fail = fail
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        return SommResponse(
            text=self.text, model=request.model or "gold-m", tokens_in=5, tokens_out=5,
            latency_ms=1, raw={},
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _shadow(repo: Repository, workload_id: str, **extra) -> None:
    cfg = {
        "gold_provider": "gold",
        "gold_model": "gold-m",
        "sample_rate": 1.0,
        "budget_usd_daily": 5.0,
        "max_grades_per_run": 20,
    }
    cfg.update(extra)
    repo.set_shadow_config(workload_id, cfg)


# ---------------------------------------------------------------------------
# 1. Atomic grading leases


def test_two_workers_claiming_the_same_sample_get_one_lease(tmp_path, monkeypatch):
    """Two workers that both read "no eval row yet" must not both claim.

    The claim used to SELECT and then INSERT; each thread here is held after
    its first statement until the other has run its own, which is the
    interleaving that produced duplicate gold calls."""
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="race", project=project)
    call_id = _call(repo, wl.id, project)

    barrier = threading.Barrier(2)
    real_open = repo._open

    class _Held:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, *args):
            cur = self._conn.execute(sql, *args)
            if "eval_results" in sql and threading.current_thread().name.startswith("claim"):
                try:
                    barrier.wait(timeout=1)
                except threading.BrokenBarrierError:
                    pass
            return cur

    monkeypatch.setattr(repo, "_open", lambda: _Held(real_open()))
    worker_a = ShadowEvalWorker(repo, providers=[])
    worker_b = ShadowEvalWorker(repo, providers=[])
    results: list = []
    threads = [
        threading.Thread(target=lambda w=w: results.append(w._claim_lease(call_id)), name=f"claim{i}")
        for i, w in enumerate((worker_a, worker_b))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    claimed = [r for r in results if r is not None]
    assert len(claimed) == 1
    with real_open() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM eval_results WHERE call_id = ?", (call_id,)
        ).fetchone()[0] == 1

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


# ---------------------------------------------------------------------------
# 2. An unparseable judge is missing, not zero


def test_unparseable_judge_reply_has_no_score():
    from somm_core.graders import normalize_binary_criteria, parse_binary_judge_response

    criteria = normalize_binary_criteria(["correctness"])
    garbled = parse_binary_judge_response("I think it is fine overall.", criteria)
    assert garbled["score"] is None
    assert "unparseable" in garbled["error"]

    partial = parse_binary_judge_response(
        '{"criteria":[{"name":"correctness","pass":true}]}',
        normalize_binary_criteria(["correctness", "completeness"]),
    )
    assert partial["score"] is None
    assert "completeness" in partial["error"]

    ok = parse_binary_judge_response('{"criteria":[{"name":"correctness","pass":false}]}', criteria)
    assert ok["score"] == 0.0 and "error" not in ok


def test_shadow_judge_that_cannot_be_parsed_records_no_score_and_says_so(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="judged", project=project)
    _shadow(repo, wl.id, judge={"provider": "judge", "model": "judge-m"})
    for i in range(3):
        call_id = _call(repo, wl.id, project)
        repo.write_sample(call_id, f"prompt {i}", '{"ok": true}')

    judge = _Gold(name="judge", text="Looks good to me!")
    summary = ShadowEvalWorker(repo, providers=[_Gold(), judge]).run_once()

    assert summary["calls_graded"] == 3
    assert summary["judge_unusable"] == 3
    assert any("judged" in w and "3 of 3" in w for w in summary["warnings"])
    with repo._open() as conn:
        rows = conn.execute("SELECT judge_score, judge_reason FROM eval_results").fetchall()
    assert [r[0] for r in rows] == [None, None, None]
    reason = [item["judge"] for item in json.loads(rows[0][1]) if "judge" in item][0]
    assert "unparseable" in reason["judges"][0]["result"]["error"]


# ---------------------------------------------------------------------------
# 3. A broken gold route does not eat samples


def _samples(repo, wl, project, n):
    ids = []
    for i in range(n):
        call_id = _call(repo, wl.id, project)
        repo.write_sample(call_id, f"prompt {i}", "answer")
        ids.append(call_id)
    return ids


def test_unconfigured_gold_provider_leaves_samples_eligible(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="misconfigured", project=project)
    _shadow(repo, wl.id, gold_provider="nowhere")
    _samples(repo, wl, project, 3)

    summary = ShadowEvalWorker(repo, providers=[_Gold()]).run_once()

    assert summary["calls_graded"] == 0
    assert "not configured" in summary["shadow_broken"]["misconfigured"]
    assert any("misconfigured" in w for w in summary["warnings"])
    with repo._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM shadow_candidates").fetchone()[0] == 3


def test_failing_gold_is_tried_once_per_run_then_reported_broken(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="flaky", project=project)
    _shadow(repo, wl.id)
    _samples(repo, wl, project, 5)
    gold = _Gold(fail=RuntimeError("model not found"))
    worker = ShadowEvalWorker(repo, providers=[gold])

    first = worker.run_once()
    assert len(gold.requests) == 1  # one failure stops the workload for this run
    assert len(first["errors"]) == 1
    assert "flaky" not in first.get("shadow_broken", {})

    for _ in range(2):
        worker.run_once()
    assert len(gold.requests) == 3
    fourth = worker.run_once()
    assert len(gold.requests) == 3  # broken: not retried until the cooldown passes
    assert "3 consecutive" in fourth["shadow_broken"]["flaky"]
    assert "model not found" in fourth["shadow_broken"]["flaky"]
    with repo._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM shadow_candidates").fetchone()[0] == 5

    # A day later the route is probed again, and a success clears the streak.
    with repo._open() as conn:
        conn.execute(
            "UPDATE calls SET ts = ? WHERE observation_role = 'shadow_gold'",
            ((datetime.now(UTC) - timedelta(days=2)).isoformat(),),
        )
    gold.fail = None
    recovered = worker.run_once()
    assert recovered["calls_graded"] == 5
    assert not recovered.get("shadow_broken")


def test_loop_output_prints_shadow_warnings(tmp_path, capsys, monkeypatch):
    from somm import loop
    from somm.cli import main

    cfg = Config()
    cfg.project = "trust"
    cfg.db_dir = tmp_path / ".somm"
    Repository(cfg.db_path)

    class _Shadow:
        def run_once(self):
            return {
                "calls_graded": 0,
                "shadow_broken": {"w": "gold_provider 'x' not configured"},
                "warnings": ["w: shadow config broken: gold_provider 'x' not configured"],
                "errors": [],
            }

    monkeypatch.setattr("somm.cli.load_config", lambda project=None: cfg)
    monkeypatch.setattr(loop, "build_workers_factory", lambda c, r: lambda n: _Shadow())
    assert main(["loop", "run", "--only", "shadow_eval"]) == 0
    out = capsys.readouterr().out
    assert "warning: w: shadow config broken: gold_provider 'x' not configured" in out


# ---------------------------------------------------------------------------
# 4. Replay the real request


def test_replay_sends_captured_messages_and_production_params(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="chat", project=project)
    _shadow(repo, wl.id)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "summarise that"},
    ]
    chat = _call(repo, wl.id, project, temperature=0.7, max_tokens=300)
    repo.write_sample(chat, json.dumps(messages), "a summary")
    plain = _call(repo, wl.id, project)  # an older row: no params recorded
    repo.write_sample(plain, '["not", "messages"]', "x")

    gold = _Gold()
    ShadowEvalWorker(repo, providers=[gold]).run_once()

    by_prompt = {
        json.dumps(r.messages) if r.messages else r.prompt: r for r in gold.requests
    }
    replay = by_prompt[json.dumps(messages)]
    assert replay.messages == messages
    assert (replay.temperature, replay.max_tokens) == (0.7, 300)
    legacy = by_prompt['["not", "messages"]']
    assert legacy.messages is None
    assert (legacy.temperature, legacy.max_tokens) == (0.0, 1024)

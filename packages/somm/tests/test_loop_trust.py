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

    monkeypatch.setattr("somm.cli._common.load_config", lambda project=None: cfg)
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


# ---------------------------------------------------------------------------
# 5-7. A floor under every recommendation, an expiry, and cost of pass


def _graded(repo, wl, project, model, scores, *, judge=False, cost=0.001, cost_source=None,
            latency_ms=100, eval_age_days=0):
    for score in scores:
        call_id = _call(
            repo, wl.id, project, model=model, cost_usd=cost, cost_source=cost_source,
            latency_ms=latency_ms,
        )
        with repo._open() as conn:
            conn.execute(
                "INSERT INTO eval_results (call_id, gold_model, structural_score, "
                "embedding_score, judge_score, ts) VALUES (?, 'gold-m', ?, ?, ?, "
                "datetime('now', ?))",
                (
                    call_id, None if judge else score, None if judge else score,
                    score if judge else None, f"-{eval_age_days} days",
                ),
            )


def _recs(repo, action="switch_model"):
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT evidence_json, expected_impact, confidence FROM recommendations "
            "WHERE action = ?",
            (action,),
        ).fetchall()
    return [(json.loads(r[0]), r[1], r[2]) for r in rows]


def test_noisy_difference_does_not_become_a_switch(tmp_path):
    """A 0.15 lift in means from two noisy arms of 20 is not evidence."""
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="noisy", project=project)
    _graded(repo, wl, project, "current", [0.0, 1.0] * 15)  # mean 0.5, sd 0.5
    _graded(repo, wl, project, "better", [0.3, 1.0] * 10)  # mean 0.65, n 20

    AgentWorker(repo).run_once()
    assert _recs(repo) == []


def test_thin_arm_does_not_become_a_switch(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="thin", project=project)
    _graded(repo, wl, project, "current", [0.4] * 30)
    _graded(repo, wl, project, "better", [0.9] * 8)  # clear but only 8 samples

    AgentWorker(repo).run_once()
    assert _recs(repo) == []


def test_stale_evidence_does_not_become_a_switch(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="stale", project=project)
    _graded(repo, wl, project, "current", [0.4] * 40, eval_age_days=60)
    _graded(repo, wl, project, "better", [0.9] * 30, eval_age_days=60)

    AgentWorker(repo).run_once()
    assert _recs(repo) == []


def test_a_clear_judged_win_records_its_interval_and_cost_of_pass(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="judged", project=project)
    _graded(repo, wl, project, "current", [1.0] * 10 + [0.0] * 20, judge=True, cost=0.002)
    _graded(
        repo, wl, project, "better", [1.0] * 20 + [0.0] * 5, judge=True, cost=0.001,
    )

    AgentWorker(repo).run_once()
    [(evidence, impact, confidence)] = _recs(repo)
    assert evidence["basis"] == "measured quality"
    assert evidence["score"] == "judge"
    assert evidence["current"]["n"] == 30 and evidence["candidate"]["n"] == 25
    lo, hi = evidence["interval"]
    assert 0 < lo < evidence["score_delta"] < hi
    assert evidence["window_days"] == 30
    # cost per passing answer = mean known cost / pass rate
    assert evidence["current"]["cost_per_pass"] == round(0.002 / (10 / 30), 6)
    assert evidence["candidate"]["cost_per_pass"] == round(0.001 / (20 / 25), 6)
    assert 0.5 < confidence < 1
    assert "expires_at" in evidence


def test_cost_of_pass_is_unknown_without_a_known_price_or_a_pass_score(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="similar", project=project)
    _graded(repo, wl, project, "current", [0.3] * 30, latency_ms=500)
    _graded(
        repo, wl, project, "better", [0.8] * 25, cost_source="unknown:no-price",
        latency_ms=100,
    )

    AgentWorker(repo).run_once()
    [(evidence, _, _)] = _recs(repo)
    assert evidence["score"] == "structural"
    assert evidence["candidate"]["cost_usd"] == "unknown"
    assert evidence["candidate"]["cost_per_pass"] == "unknown"
    assert evidence["current"]["cost_per_pass"] == "unknown"  # similarity is not pass/fail


def test_catalog_rule_says_it_is_not_a_quality_finding(tmp_path):
    from somm_core.pricing import write_intel

    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="catalog", project=project)
    for _ in range(12):
        _call(repo, wl.id, project, provider="openrouter", model="old")
    write_intel(repo, "openrouter", "old", 1.0, 4.0, None, None, "test")
    with repo._open() as conn:
        conn.execute(
            "UPDATE model_intel SET first_seen = datetime('now', '-90 days') WHERE model = 'old'"
        )
    write_intel(repo, "openrouter", "new", 0.1, 0.4, None, None, "test")

    AgentWorker(repo).run_once()
    [(evidence, impact, confidence)] = _recs(repo, "new_model_landed")
    assert evidence["basis"] == "catalog"
    assert impact.startswith("catalog fact, not a quality finding")
    assert confidence is None


def test_open_recommendations_expire_closed_not_applied(tmp_path, capsys, monkeypatch):
    from somm.cli import main
    from somm_core.read import north_star

    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="old", project=project)
    _call(repo, wl.id, project)
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO recommendations (workload_id, action, evidence_json, created_at) "
            "VALUES (?, 'chronic_cooldown', '{}', datetime('now', '-40 days'))",
            (wl.id,),
        )
        conn.execute(
            "INSERT INTO recommendations (workload_id, action, evidence_json) "
            "VALUES (?, 'switch_model', '{}')",
            (wl.id,),
        )

    summary = AgentWorker(repo).run_once()
    assert summary["expired"] == 1
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT action, dismissed_at IS NOT NULL, applied_at, "
            "json_extract(evidence_json, '$.closed_as') FROM recommendations ORDER BY id"
        ).fetchall()
    assert rows == [("chronic_cooldown", 1, None, "expired"), ("switch_model", 0, None, None)]
    applied = north_star(repo.db_path)["recommendations_applied"]
    assert (applied["eligible"], applied["acted"], applied["applied"]) == (1, 0, 0)

    cfg = Config()
    cfg.project = project
    cfg.db_dir = repo.db_path.parent
    monkeypatch.setattr("somm.cli._common.load_config", lambda project=None: cfg)
    assert main(["inbox", "list", "--all"]) == 0
    assert "expired" in capsys.readouterr().out


def test_concurrent_agent_runs_write_one_recommendation(tmp_path):
    project, repo = _repo(tmp_path)
    wl = repo.register_workload(name="once", project=project)
    _graded(repo, wl, project, "current", [0.3] * 30, latency_ms=500)
    _graded(repo, wl, project, "better", [0.8] * 25, latency_ms=100)
    worker = AgentWorker(repo)
    [rec] = worker._rec_switch_model()
    assert worker._write(rec) is True
    assert worker._write(rec) is False  # the open one blocks a twin, in one statement


def test_replay_uses_the_captured_request(tmp_path):
    """A sample captured with its request is replayed as that request."""
    import json as _json

    from somm.workers.shadow_eval import ShadowEvalWorker
    from somm_core.repository import Repository

    repo = Repository(tmp_path / "calls.sqlite")
    with repo._open() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")  # the sample's call row is not needed here
        conn.execute(
            "INSERT INTO samples (call_id, prompt_body, response_body, request_json) "
            "VALUES ('c1', 'p', 'r', ?)",
            (_json.dumps({"system": "be brief", "reasoning_effort": "low", "max_tokens": 77,
                          "tools": [{"name": "t"}], "temperature": None}),),
        )
        conn.commit()
    worker = ShadowEvalWorker.__new__(ShadowEvalWorker)
    worker.repo = repo
    params = worker._production_params("c1")
    assert params["system"] == "be brief"
    assert params["reasoning_effort"] == "low"
    assert params["tools"] == [{"name": "t"}]
    assert params["max_tokens"] == 77
    assert params["temperature"] == 0.0  # not captured: the old default stands
    assert worker._production_params("missing") == {"temperature": 0.0, "max_tokens": 1024}

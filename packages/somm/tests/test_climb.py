"""The climb, read from (invented) chipsocket receipts."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm import climb
from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository

SCHEMA = "chip.socket-receipt/v0alpha1"


def _line(answered: str | None, socket="demo.sort", days_ago=1.0, call_id=None, schema=SCHEMA) -> str:
    started = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    attempts = [{"member": "code:rules@sha256:aa", "trigger": "primary", "outcome": "abstained"}]
    if answered and answered.startswith("model:"):
        meta = {"model": "m1", **({"call_id": call_id} if call_id else {})}
        attempts.append({"member": answered + "@sha256:bb", "trigger": "abstained", "outcome": "value", "meta": meta})
    r = {"schema": schema, "socket": socket, "run": "srun_x", "answeredBy": answered + "@sha256:bb" if answered else None,
         "attempts": attempts, "shadows": [{"member": "model:other@sha256:cc", "meta": {"call_id": "shadow-id"}}],
         "timing": {"startedAt": started}, "input": {"answeredBy": "model:fake", "text": 'say "hi"'}}
    return json.dumps(r, separators=(",", ":"))


def _write(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _mix():
    return ([_line("code:rules")] * 6 + [_line("code:cache")] * 2 + [_line("model:judge", call_id="c1")]
            + [_line(None)])


def test_rungs_shares_and_derived_share(tmp_path):
    _write(tmp_path / "r-2026-11-01.jsonl", _mix())
    r = climb.climb(tmp_path, since_days=30)
    assert r["decisions"] == 10
    assert {x["rung"]: x["decisions"] for x in r["rungs"]} == {
        "code:rules": 6, "code:cache": 2, "model:judge": 1, "exhausted": 1}
    assert r["share_without_model"] == 0.8
    assert r["model"]["join"] == "none" and r["cost_per_1000_decisions"] is None


def test_gz_alone_is_read_and_plain_twin_is_not_double_counted(tmp_path):
    with gzip.open(tmp_path / "r-2026-11-01.jsonl.gz", "wt") as f:
        f.write("\n".join(_mix()) + "\n")
    assert climb.climb(tmp_path)["decisions"] == 10
    _write(tmp_path / "r-2026-11-01.jsonl", _mix())  # crash left both behind
    r = climb.climb(tmp_path)
    assert r["decisions"] == 10 and r["skipped"]["gz_twin_files"] == 1
    # a truncated gz beside a complete plain file does not matter either
    (tmp_path / "r-2026-11-01.jsonl.gz").write_bytes(b"\x1f\x8b\x08junk")
    assert climb.climb(tmp_path)["decisions"] == 10


def test_unknown_versions_garbage_and_foreign_files_are_skipped_and_counted(tmp_path):
    good = _line("code:rules")
    newer = _line("code:rules", schema="chip.socket-receipt/v1")
    _write(tmp_path / "a.jsonl", [good, newer, "not json", "", '{"schema":"other"}', good])
    _write(tmp_path / "harvest.jsonl", ['{"run":"srun_x","input":{}}'])
    (tmp_path / "bad.jsonl.gz").write_bytes(b"not gzip at all")
    r = climb.climb(tmp_path)
    assert r["decisions"] == 2
    assert r["skipped"]["unknown_lines"] == 3 and r["skipped"]["foreign_files"] == 1
    assert r["skipped"]["unreadable_files"] == 1


def test_unknown_fields_ignored_and_nested_lookalikes_do_not_confuse(tmp_path):
    line = json.loads(_line("code:rules"))
    line["future"] = {"answeredBy": "model:x", "socket": "other"}
    _write(tmp_path / "a.jsonl", [json.dumps(line, separators=(",", ":"))])
    r = climb.climb(tmp_path, workload="demo.sort")
    assert [x["rung"] for x in r["rungs"]] == ["code:rules"]


def test_workload_and_window_filters(tmp_path):
    _write(tmp_path / "a.jsonl", [_line("code:rules", socket="a", days_ago=1), _line("code:rules", socket="a", days_ago=40),
                                  _line("code:rules", socket="ab", days_ago=1)])
    assert climb.climb(tmp_path, since_days=30)["decisions"] == 2
    assert climb.climb(tmp_path, workload="a", since_days=30)["decisions"] == 1
    assert climb.climb(tmp_path, workload="a", since_days=None)["decisions"] == 2


def test_single_file_and_missing_path_and_env(tmp_path, monkeypatch):
    f = tmp_path / "one.jsonl"
    _write(f, [_line("code:rules")])
    assert climb.climb(f)["decisions"] == 1
    assert climb.climb(tmp_path / "nope") is None
    monkeypatch.setenv(climb.ENV_PATH, str(f))
    assert climb.climb()["decisions"] == 1


def _ledger(tmp_path, project="p"):
    cfg = Config()
    cfg.project = project
    cfg.db_dir = tmp_path / ".somm"
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="demo.sort", project=project)
    other = repo.register_workload(name="elsewhere", project=project)

    def call(cid, wid, cost, source="somm:model_intel", role="production"):
        c = Call(id=cid, ts=datetime.now(UTC), project=project, workload_id=wid, prompt_id=None, provider="fake",
                 model="m1", tokens_in=100, tokens_out=10, latency_ms=200, cost_usd=cost, outcome=Outcome.OK,
                 error_kind=None, prompt_hash="p", response_hash="r", cost_source=source)
        c.observation_role = role
        repo.write_call(c)

    call("c1", wl.id, 0.02)
    call("c2", wl.id, 0.04)
    call("c3", wl.id, 0.0, source="unknown:no-price")
    call("c4", wl.id, 9.0, role="shadow")
    call("c5", other.id, 7.0)
    return cfg


def test_join_by_call_id_counts_only_matching_ledger_rows(tmp_path):
    cfg = _ledger(tmp_path)
    lines = [_line("code:rules")] * 6 + [_line("model:judge", call_id="c1"), _line("model:judge", call_id="c2"),
                                         _line("model:judge", call_id="gone"), _line("model:judge")]
    _write(tmp_path / "r" / "a.jsonl", lines)
    r = climb.climb(tmp_path / "r", db_path=cfg.db_path, project=cfg.project)
    m = r["model"]
    assert m["join"] == "call_id" and m["decisions"] == 4 and m["receipts_with_call_id"] == 3
    assert m["ledger"]["calls"] == 2 and m["ledger"]["cost_usd"] == 0.06
    assert m["ledger"]["tokens_in"] == 200 and m["ledger"]["mean_latency_ms"] == 200.0
    assert r["cost_per_1000_decisions"] == 6.0  # 0.06 / 10 decisions * 1000
    assert r["cost_per_1000_if_all_model"] == 30.0  # 0.03 per call * 1000
    assert "3 receipts carrying a call id" in climb.format_report(r)


def test_join_by_exact_workload_name_when_no_call_ids(tmp_path):
    cfg = _ledger(tmp_path)
    _write(tmp_path / "a.jsonl", [_line("code:rules")] * 3 + [_line("model:judge")])
    r = climb.climb(tmp_path, workload="demo.sort", db_path=cfg.db_path, project=cfg.project)
    led = r["model"]["ledger"]
    assert r["model"]["join"] == "workload"
    assert led["calls"] == 3 and led["cost_usd"] == 0.06 and led["cost_unknown_calls"] == 1  # shadow and other excluded
    # without --workload and without call ids there is nothing exact to join by
    assert climb.climb(tmp_path, db_path=cfg.db_path, project=cfg.project)["model"]["join"] == "none"


def test_read_only_and_main_runs_without_a_ledger(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "a.jsonl", _mix())
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert climb.main([str(tmp_path / "a.jsonl"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decisions"] == 10 and out["share_without_model"] == 0.8
    assert climb.main([str(tmp_path / "a.jsonl")]) == 0
    assert "answered without a model" in capsys.readouterr().out
    assert climb.main([str(tmp_path / "missing")]) == 1
    assert before == ["a.jsonl"]

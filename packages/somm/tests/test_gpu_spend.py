"""Rented-GPU spend read from (invented) gipper receipts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm import gpu_spend
from somm.cli import main
from somm_core.config import Config
from somm_core.repository import Repository


def _ts(days_ago: float = 0.0) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _r(kind, job, payload=None, refs=None, ts=None, schema="gipper.receipt/v1"):
    return {"schema": schema, "id": "r_x", "ts": ts or _ts(), "kind": kind, "job": job,
            "refs": refs or {}, "payload": payload or {}}


def _write(d: Path, recs, name="2026-11.jsonl", raw: list[str] | None = None):
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in recs] + (raw or [])
    (d / name).write_text("\n".join(lines) + "\n")


def _job(job, workload, outcome, usd, billed=None, pod="p1", days_ago=1.0):
    refs = {"name": f"{workload}-run", "workload": workload}
    recs = [
        _r("estimate", job, {"usd": 1}, refs, ts=_ts(days_ago + 0.1)),
        _r("pod.create", job, {"pod": pod, "gpu": "RTX 4090", "usd_per_h": 0.5}, ts=_ts(days_ago + 0.09)),
        _r("spend", job, {"provider": "runpod", "pod": pod, "seconds": 600, "usd": usd}, refs, ts=_ts(days_ago)),
        _r("run.end", job, {"outcome": outcome}, refs, ts=_ts(days_ago)),
    ]
    if billed is not None:
        recs.append(_r("spend.billed", job, {"provider": "runpod", "pod": pod, "usd": billed},
                       ts=_ts(days_ago - 0.01)))
    return recs


def test_billed_beats_computed_and_failures_counted(tmp_path):
    d = tmp_path / "receipts"
    _write(d, _job("job_a", "summarize", "succeeded", 1.0, billed=1.25)
           + _job("job_b", "summarize", "failed", 0.5, pod="p2")
           + _job("job_c", "other", "capped", 0.25, pod="p3"))
    jobs, skipped = gpu_spend.read_jobs(d)
    by = {j.job: j for j in jobs}
    assert by["job_a"].usd == 1.25 and by["job_a"].cost_source == "billed"
    assert by["job_b"].usd == 0.5 and by["job_b"].cost_source == "computed"
    assert by["job_a"].gpu == "RTX 4090" and by["job_a"].seconds == 600
    r = gpu_spend.summarize(jobs, 30)
    assert r["total_usd"] == 2.0
    assert r["not_succeeded"] == {"capped": {"jobs": 1, "usd": 0.25}, "failed": {"jobs": 1, "usd": 0.5}}
    assert r["by_workload"]["summarize"]["jobs"] == 2
    assert skipped["estimate_only"] == 0


def test_tolerates_unknown_schema_garbage_and_unknown_fields(tmp_path):
    d = tmp_path / "receipts"
    good = _job("job_a", "w", "succeeded", 1.0)
    good[2]["payload"]["new_field"] = {"x": 1}
    _write(d, good + [_r("spend", "job_z", {"usd": 9}, schema="gipper.receipt/v2")],
           raw=["not json", '{"schema": "gipper.receipt/v1"}', "[1]"])
    jobs, skipped = gpu_spend.read_jobs(d)
    assert [j.job for j in jobs] == ["job_a"]
    assert skipped["unknown_schema"] == 4


def test_estimate_only_and_reaps_and_multi_machine(tmp_path):
    d = tmp_path / "receipts"
    two = _job("job_m", "w", "succeeded", 1.0) + [
        _r("spend", "job_m", {"provider": "vast", "pod": "p9", "seconds": 60, "usd": 0.5})]
    _write(d, [_r("estimate", "job_e", {"usd": 1}, {"workload": "w"})]
           + [_r("reap", None, {"pod": "px"}), _r("reap", "job_e", {"pod": "py"})] + two)
    jobs, skipped = gpu_spend.read_jobs(d)
    by = {j.job: j for j in jobs}
    assert by["job_m"].usd == 1.5 and by["job_m"].provider == "runpod+vast"
    assert by["job_e"].status == "reaped"
    assert skipped["unattributed_reaps"] == 1


def test_window_and_by_day(tmp_path):
    d = tmp_path / "receipts"
    _write(d, _job("job_new", "w", "succeeded", 1.0, days_ago=1)
           + _job("job_old", "w", "succeeded", 5.0, pod="p2", days_ago=40))
    jobs, _ = gpu_spend.read_jobs(d)
    r = gpu_spend.summarize(jobs, 30)
    assert r["jobs"] == 1 and r["total_usd"] == 1.0 and len(r["by_day"]) == 1


def test_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("SOMM_GIPPER_RECEIPTS", raising=False)
    monkeypatch.setenv("GIPPER_DATA", str(tmp_path / "data"))
    assert gpu_spend.receipts_dir() == tmp_path / "data" / "receipts"
    monkeypatch.setenv("SOMM_GIPPER_RECEIPTS", str(tmp_path / "env"))
    assert gpu_spend.receipts_dir() == tmp_path / "env"
    assert gpu_spend.receipts_dir(tmp_path / "arg") == tmp_path / "arg"


def test_no_directory_is_none(tmp_path):
    assert gpu_spend.gpu_spend(tmp_path / "missing") is None


def _cli_env(tmp_path, monkeypatch):
    cfg = Config()
    cfg.project = "gpu-test"
    cfg.db_dir = tmp_path / ".somm"
    monkeypatch.setattr("somm.cli._common.load_config", lambda project=None, **kw: cfg)
    return cfg


def test_cli_json_matches_only_exact_workload_names(tmp_path, monkeypatch, capsys):
    cfg = _cli_env(tmp_path, monkeypatch)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="summarize", project=cfg.project)
    d = tmp_path / "receipts"
    _write(d, _job("job_a", "summarize", "succeeded", 1.0) + _job("job_b", "Summarize", "succeeded", 2.0, pod="p2"))
    assert main(["gpu", "spend", "--path", str(d), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["by_workload"]["summarize"]["matches_somm_workload"] is True
    assert out["by_workload"]["summarize"]["somm_ledger_usd"] == 0.0
    assert out["by_workload"]["Summarize"]["matches_somm_workload"] is False
    assert main(["gpu", "spend", "--path", str(d)]) == 0
    assert "exactly" in capsys.readouterr().out


def test_cli_missing_directory(tmp_path, monkeypatch, capsys):
    _cli_env(tmp_path, monkeypatch)
    assert main(["gpu", "spend", "--path", str(tmp_path / "nope")]) == 1


def test_doctor_line_only_when_receipts_exist(tmp_path, monkeypatch, capsys):
    _cli_env(tmp_path, monkeypatch)
    monkeypatch.setenv("SOMM_GIPPER_RECEIPTS", str(tmp_path / "receipts"))
    main(["doctor"])
    assert "gipper:" not in capsys.readouterr().out
    _write(tmp_path / "receipts", _job("job_a", "w", "succeeded", 1.0))
    main(["doctor"])
    assert "gipper:   1 rented-GPU job(s) in the last 30d, $1.00" in capsys.readouterr().out

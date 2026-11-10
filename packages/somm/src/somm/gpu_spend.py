"""Rented-GPU spend, read from gipper's receipt files. Read-only.

gipper (a separate tool that rents GPUs for batch jobs) appends one JSON line per
event to ``<data>/receipts/YYYY-MM.jsonl``, schema ``gipper.receipt/v1``:
``{schema, id, ts, kind, job, refs, payload}``. somm reads those files by that
documented shape; it does not import gipper and never writes to its data.

What a job's receipts mean here:
  * ``refs.name`` / ``refs.workload`` (on estimate, spend, run.end) name the job.
  * ``spend`` (per machine, written at teardown): ``payload.usd`` is gipper's
    computed cost (rate x lifetime), ``payload.seconds`` the lifetime.
  * ``spend.billed`` (per machine, optional): the provider's own charge. Where one
    exists it wins over ``spend``. A job may rent several machines; costs add up.
  * ``run.end``: ``payload.outcome`` (succeeded, failed, error, capped, ...).
  * ``reap``: gipper's reaper closed a machine; counted as ``reaped`` only when the
    receipt names a job, otherwise it is reported as unattributed.
A job with only an estimate (nothing rented, nothing ended) is not a job here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCHEMA = "gipper.receipt/v1"
ENV_PATH = "SOMM_GIPPER_RECEIPTS"


def receipts_dir(path: str | Path | None = None) -> Path:
    """Argument, else ``SOMM_GIPPER_RECEIPTS``, else gipper's own rule:
    ``$GIPPER_DATA/receipts``, default ``~/.local/share/gipper/receipts``."""
    if path:
        return Path(path).expanduser()
    if os.environ.get(ENV_PATH):
        return Path(os.environ[ENV_PATH]).expanduser()
    data = os.environ.get("GIPPER_DATA") or Path.home() / ".local/share/gipper"
    return Path(data).expanduser() / "receipts"


def _num(v) -> float:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else 0.0


@dataclass
class GpuJob:
    job: str
    name: str | None
    workload: str | None
    provider: str | None
    gpu: str | None
    start: str
    end: str
    seconds: int | None
    usd: float
    cost_source: str  # "billed", "computed", "mixed" (some machines each), or "none"
    status: str  # run.end outcome, "reaped", or "unfinished"


def _parse(path: Path) -> tuple[list[dict], dict]:
    recs: list[dict] = []
    skipped = {"unknown_schema": 0, "unreadable": 0}
    for f in sorted(path.glob("*.jsonl")):
        try:
            lines = f.read_text().splitlines()
        except OSError:
            skipped["unreadable"] += 1
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                ok = isinstance(r, dict) and r.get("schema") == SCHEMA and isinstance(r["ts"], str)
            except (ValueError, KeyError):
                ok = False
            if ok:
                recs.append(r)
            else:
                skipped["unknown_schema"] += 1
    return recs, skipped


def read_jobs(path: str | Path | None = None) -> tuple[list[GpuJob], dict]:
    """Parse receipts into one row per job, plus counts of what was skipped.
    Never raises on content it does not understand."""
    root = receipts_dir(path)
    skipped = {"unknown_schema": 0, "unreadable": 0, "estimate_only": 0, "unattributed_reaps": 0}
    if not root.is_dir():
        return [], skipped
    recs, bad = _parse(root)
    skipped.update(bad)
    by_job: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if isinstance(r.get("job"), str):
            by_job[r["job"]].append(r)
        elif r.get("kind") == "reap":
            skipped["unattributed_reaps"] += 1
    jobs = []
    for job, rs in by_job.items():
        kinds = {r.get("kind") for r in rs}
        if not kinds & {"run.end", "spend", "spend.billed", "reap"}:
            skipped["estimate_only"] += 1
            continue
        jobs.append(_row(job, sorted(rs, key=lambda r: r["ts"])))
    return sorted(jobs, key=lambda j: j.end), skipped


def _row(job: str, rs: list[dict]) -> GpuJob:
    def p(r: dict) -> dict:
        return r["payload"] if isinstance(r.get("payload"), dict) else {}

    def refs(key: str) -> str | None:
        return next((r["refs"][key] for r in rs if isinstance(r.get("refs"), dict) and r["refs"].get(key)), None)

    computed: dict[tuple, float] = defaultdict(float)
    billed: dict[tuple, float] = {}
    seconds, gpus, providers, status = 0, [], [], "unfinished"
    saw_seconds = False
    for r in rs:
        k, q = r.get("kind"), p(r)
        key = (q.get("provider") or "runpod", str(q.get("pod")))
        if k == "pod.create" and q.get("gpu") and q["gpu"] not in gpus:
            gpus.append(q["gpu"])
        elif k == "spend":
            computed[key] += _num(q.get("usd"))
            if isinstance(q.get("seconds"), int | float):
                seconds, saw_seconds = seconds + int(q["seconds"]), True
            if key[0] not in providers:
                providers.append(key[0])
        elif k == "spend.billed":
            billed[key] = max(billed.get(key, 0.0), _num(q.get("usd")))
            if key[0] not in providers:
                providers.append(key[0])
        elif k == "run.end" and q.get("outcome"):
            status = str(q["outcome"])
    if any(r.get("kind") == "reap" for r in rs):
        status = "reaped"
    machines = set(computed) | set(billed)
    usd = sum(billed.get(m, computed.get(m, 0.0)) for m in machines)
    n_billed = sum(1 for m in machines if m in billed)
    source = "none" if not machines else "billed" if n_billed == len(machines) else (
        "computed" if n_billed == 0 else "mixed")
    return GpuJob(
        job=job, name=refs("name"), workload=refs("workload"),
        provider="+".join(providers) or None, gpu="+".join(gpus) or None,
        start=rs[0]["ts"], end=rs[-1]["ts"], seconds=seconds if saw_seconds else None,
        usd=round(usd, 6), cost_source=source, status=status,
    )


def _ledger_by_workload(db_path: Path, project: str, since_iso: str) -> dict[str, float]:
    """Budget-eligible API/seat spend per registered workload name in this project."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT w.name, COALESCE(SUM(c.cost_usd), 0) FROM workloads w "
            "LEFT JOIN calls c ON c.workload_id = w.id AND c.ts >= ? AND c.budget_eligible != 0 "
            "WHERE w.project = ? GROUP BY w.name",
            (since_iso, project),
        ).fetchall()
    return {name: float(usd) for name, usd in rows}


def summarize(jobs: list[GpuJob], days: int, ledger: dict[str, float] | None = None) -> dict:
    """Totals over the last ``days`` days (a job counts on the UTC day it ended).
    ``ledger`` maps somm workload names to ledger spend; a GPU workload is paired
    with it only when the names are exactly equal."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    window = [j for j in jobs if j.end >= cutoff]
    by_day: dict[str, float] = defaultdict(float)
    by_wl: dict[str, dict] = {}
    bad: dict[str, dict] = {}
    for j in window:
        by_day[j.end[:10]] += j.usd
        w = by_wl.setdefault(j.workload or "(none)", {"jobs": 0, "gpu_usd": 0.0})
        w["jobs"] += 1
        w["gpu_usd"] += j.usd
        if j.status != "succeeded":
            b = bad.setdefault(j.status, {"jobs": 0, "usd": 0.0})
            b["jobs"] += 1
            b["usd"] += j.usd
    for name, w in by_wl.items():
        w["gpu_usd"] = round(w["gpu_usd"], 4)
        w["matches_somm_workload"] = bool(ledger) and name in ledger
        w["somm_ledger_usd"] = round(ledger[name], 4) if w["matches_somm_workload"] else None
    return {
        "window_days": days,
        "jobs": len(window),
        "total_usd": round(sum(j.usd for j in window), 4),
        "billed_usd": round(sum(j.usd for j in window if j.cost_source == "billed"), 4),
        "by_day": {d: round(v, 4) for d, v in sorted(by_day.items())},
        "by_workload": dict(sorted(by_wl.items(), key=lambda kv: -kv[1]["gpu_usd"])),
        "not_succeeded": {s: {"jobs": b["jobs"], "usd": round(b["usd"], 4)} for s, b in sorted(bad.items())},
        "jobs_detail": [asdict(j) for j in window],
    }


def gpu_spend(path=None, days: int = 30, db_path: Path | None = None, project: str = "") -> dict | None:
    """The whole report, or None when there is no receipts directory."""
    if not receipts_dir(path).is_dir():
        return None
    jobs, skipped = read_jobs(path)
    ledger = None
    if db_path is not None and Path(db_path).exists():
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        ledger = _ledger_by_workload(Path(db_path), project, since)
    out = summarize(jobs, days, ledger)
    out["skipped"] = skipped
    out["receipts_dir"] = str(receipts_dir(path))
    return out


def format_report(r: dict) -> str:
    L = [f"rented-GPU spend, last {r['window_days']}d  (read from {r['receipts_dir']})",
         f"  {r['jobs']} job(s), ${r['total_usd']:.2f} total (${r['billed_usd']:.2f} of it provider-billed, rest computed by gipper)"]
    if r["by_day"]:
        L += ["", "by day (UTC, the day a job ended)"] + [f"  {d}  ${v:>8.2f}" for d, v in r["by_day"].items()]
    if r["by_workload"]:
        L += ["", f"{'by workload':<30} {'jobs':>5} {'gpu':>9} {'somm ledger':>12}"]
        for n, w in r["by_workload"].items():
            led = f"${w['somm_ledger_usd']:.2f}" if w["matches_somm_workload"] else "-"
            L.append(f"  {n[:28]:<28} {w['jobs']:>5} ${w['gpu_usd']:>8.2f} {led:>12}")
        L.append("  (ledger column: API/seat spend, shown only where the gipper workload name equals a somm workload name exactly)")
    if r["not_succeeded"]:
        L += ["", "not succeeded"] + [f"  {s:<20} {b['jobs']:>4} job(s)  ${b['usd']:.2f}" for s, b in r["not_succeeded"].items()]
    s = {k: v for k, v in r["skipped"].items() if v}
    if s:
        L += ["", "skipped: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in s.items())]
    return "\n".join(L)

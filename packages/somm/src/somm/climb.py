"""The climb: how much of a workload is answered before any model is called. Read-only.

chipsocket (a separate tool) runs a workload through a "socket" in which code members
(rules, lexical cuts, a cache) answer first and only the remainder reaches a model, and it
writes one JSON line per decision, schema ``chip.socket-receipt/v0alpha1``. somm reads those
files by that shape; it does not import chipsocket and never writes to its data.

Fields used (everything else is ignored): ``schema``, ``socket``, ``answeredBy``
(``kind:member@digest``, null when every member failed), ``timing.startedAt``, and for
model-answered decisions the answering attempt's ``meta.call_id`` and ``meta.model``.

* The rung is ``answeredBy`` without its digest: ``code:<name>``, ``model:<name>``,
  ``baseline:<name>``, or ``exhausted``. A cache replaying an earlier model answer is a
  ``code`` member, so it counts as answered without a model; its name is whatever the socket
  author chose.
* One decision is one receipt. chipsocket gzips a finished day in place: it writes
  ``X.jsonl.gz`` completely, then deletes ``X.jsonl``. So where both exist the plain file is
  the complete one (the gzip may be a crash leftover) and the ``.gz`` is skipped; no day is
  read twice and no id set is held in memory.
* Lines with another schema or version are skipped and counted; a file whose first line is
  not a receipt (a harvest file, say) is skipped whole.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

PREFIX = '{"schema":"chip.socket-receipt/v0alpha1"'
ENV_PATH = "SOMM_CLIMB_RECEIPTS"
_SOCKET = re.compile(r'"socket":"([^"]*)"')
_ANSWERED = re.compile(r'"answeredBy":(?:null|"([^"]*)")')
_STARTED = re.compile(r'"startedAt":"([^"]*)"')


def receipt_files(path: str | Path) -> tuple[list[Path], int]:
    """The files to read, and how many ``.gz`` twins were skipped (see module doc)."""
    p = Path(path).expanduser()
    if p.is_file():
        return [p], 0
    if not p.is_dir():
        return [], 0
    plain = {f.name for f in p.glob("*.jsonl")}
    gz = [f for f in p.glob("*.jsonl.gz") if f.name[:-3] not in plain]
    twins = len(list(p.glob("*.jsonl.gz"))) - len(gz)
    return sorted([p / n for n in plain] + gz, key=lambda f: f.name), twins


def scan(path: str | Path, workload: str | None = None, since_days: int | None = None) -> dict:
    """Stream every receipt once; count decisions per rung. Model-answered lines are parsed
    fully (they are few); all other lines only get the three fields needed."""
    cutoff = None
    if since_days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=since_days)).strftime("%Y-%m-%dT%H:%M:%S")
    files, twins = receipt_files(path)
    rungs: Counter = Counter()
    model_ids: list[str] = []
    model_no_id = 0
    sk = Counter(gz_twin_files=twins)
    for f in files:
        try:
            with (gzip.open(f, "rt", encoding="utf-8") if f.name.endswith(".gz") else open(f, encoding="utf-8")) as fh:
                first = True
                for line in fh:
                    if first:
                        first = False
                        if not line.strip():
                            first = True
                            continue
                        if not line.startswith(PREFIX):
                            sk["foreign_files"] += 1
                            break
                    if not line.startswith(PREFIX):
                        sk["unknown_lines"] += bool(line.strip())
                        continue
                    if workload is not None:
                        s = _SOCKET.search(line)
                        if not s or s.group(1) != workload:
                            continue
                    if cutoff is not None:
                        t = _STARTED.search(line)
                        if not t or t.group(1)[:19] < cutoff:
                            continue
                    m = _ANSWERED.search(line)
                    if not m:
                        sk["unknown_lines"] += 1
                        continue
                    rung = m.group(1).rsplit("@", 1)[0] if m.group(1) else "exhausted"
                    rungs[rung] += 1
                    if rung.startswith("model:"):
                        cid = _call_id(line)
                        if cid:
                            model_ids.append(cid)
                        else:
                            model_no_id += 1
        except (OSError, EOFError, UnicodeDecodeError):
            sk["unreadable_files"] += 1
    return {"files": len(files), "rungs": rungs, "model_ids": model_ids,
            "model_without_call_id": model_no_id, "skipped": dict(sk)}


def _call_id(line: str) -> str | None:
    try:
        attempts = json.loads(line).get("attempts") or []
        won = [a for a in attempts if isinstance(a, dict) and a.get("outcome") == "value"]
        cid = ((won[-1].get("meta") or {}).get("call_id")) if won else None
        return cid if isinstance(cid, str) and cid else None
    except (ValueError, AttributeError, TypeError):
        return None


def _ledger(db_path: Path, ids: list[str], workload: str | None, project: str, since_iso: str | None) -> dict:
    """Model-call totals from the ledger, opened read-only: by call id when receipts carry
    ids, else by exact workload name (production calls in the window)."""
    from somm_core.cost import known_cost_sql

    sel = ("SELECT COUNT(*), COALESCE(SUM(c.tokens_in),0), COALESCE(SUM(c.tokens_out),0), "
           "COALESCE(SUM(CASE WHEN {k} THEN c.cost_usd ELSE 0 END),0), "
           "COALESCE(SUM(CASE WHEN {k} THEN 0 ELSE 1 END),0), AVG(c.latency_ms) FROM calls c ").format(
               k=known_cost_sql("c."))
    parts: list[tuple] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        if ids:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                parts.append(conn.execute(sel + f"WHERE c.id IN ({','.join('?' * len(chunk))})", chunk).fetchone())
        else:
            q, a = ("JOIN workloads w ON w.id = c.workload_id WHERE w.name = ? AND w.project = ? "
                    "AND c.observation_role = 'production'", [workload, project])
            if since_iso:
                q, a = q + " AND c.ts >= ?", a + [since_iso]
            parts.append(conn.execute(sel + q, a).fetchone())
    n = sum(p[0] for p in parts)
    lat = sum((p[5] or 0) * p[0] for p in parts)
    return {"calls": n, "tokens_in": sum(p[1] for p in parts), "tokens_out": sum(p[2] for p in parts),
            "cost_usd": sum(p[3] for p in parts), "cost_unknown_calls": sum(p[4] for p in parts),
            "mean_latency_ms": round(lat / n, 1) if n else None}


def climb(path: str | Path | None = None, workload: str | None = None, since_days: int | None = 30,
          db_path: Path | None = None, project: str = "") -> dict | None:
    """The whole report, or None when there are no receipts at ``path``."""
    path = path or os.environ.get(ENV_PATH)
    if not path or not Path(path).expanduser().exists():
        return None
    s = scan(path, workload, since_days)
    total = sum(s["rungs"].values())
    out: dict = {"path": str(Path(path).expanduser()), "window_days": since_days, "workload": workload,
                 "files": s["files"], "decisions": total, "skipped": s["skipped"],
                 "rungs": [{"rung": r, "decisions": n, "share": round(n / total, 4)}
                           for r, n in s["rungs"].most_common()] if total else []}
    code = sum(n for r, n in s["rungs"].items() if r.startswith("code:"))
    out["share_without_model"] = round(code / total, 4) if total else None
    nmodel = sum(n for r, n in s["rungs"].items() if r.startswith("model:"))
    join, led = "none", None
    if nmodel and db_path is not None and Path(db_path).exists():
        if s["model_ids"]:
            join = "call_id"
        elif workload:
            join = "workload"
        if join != "none":
            since = (datetime.now(UTC) - timedelta(days=since_days)).isoformat() if since_days is not None else None
            led = _ledger(Path(db_path), s["model_ids"] if join == "call_id" else [], workload, project, since)
    out["model"] = {"decisions": nmodel, "join": join, "receipts_with_call_id": len(s["model_ids"]),
                    "ledger": led}
    out["cost_per_1000_decisions"] = out["cost_per_1000_if_all_model"] = None
    if led and led["calls"] and total:
        out["cost_per_1000_decisions"] = round(led["cost_usd"] / total * 1000, 4)
        out["cost_per_1000_if_all_model"] = round(led["cost_usd"] / led["calls"] * 1000, 4)
    return out


def format_report(r: dict) -> str:
    win = f"last {r['window_days']}d" if r["window_days"] is not None else "all time"
    L = [f"the climb, {win}" + (f", socket {r['workload']}" if r["workload"] else "")
         + f"  (read from {r['path']}, {r['files']} file(s))", f"  {r['decisions']} decisions"]
    if r["rungs"]:
        L += ["", f"{'answered by':<40} {'decisions':>10} {'share':>7}"]
        L += [f"  {x['rung'][:38]:<38} {x['decisions']:>10} {x['share']:>7.1%}" for x in r["rungs"]]
    m, led = r["model"], r["model"]["ledger"]
    if m["decisions"]:
        L += ["", f"model rung: {m['decisions']} decisions, join: {m['join']}"]
        if led:
            hit = f"{led['calls']} ledger calls matched"
            if m["join"] == "call_id":
                hit += f" of {m['receipts_with_call_id']} receipts carrying a call id ({m['decisions']} model decisions)"
            else:
                hit += " (all production calls of that workload name in the window, not tied to receipts one by one)"
            L.append(f"  {hit}")
            L.append(f"  tokens in/out {led['tokens_in']}/{led['tokens_out']}, known cost ${led['cost_usd']:.4f}"
                     f" ({led['cost_unknown_calls']} call(s) of unknown price), mean latency {led['mean_latency_ms']} ms")
        else:
            L.append("  not joined: no ledger, no call id on the receipts, and no --workload to match by name")
    L += ["", "derived"]
    sw = r["share_without_model"]
    L.append("  answered without a model:      " + ("-" if sw is None else f"{sw:.1%}  (code members only)"))
    for label, key in (("model cost per 1,000 decisions:", "cost_per_1000_decisions"),
                       ("per 1,000 if all went to model:", "cost_per_1000_if_all_model")):
        v = r[key]
        L.append(f"  {label:<32}" + ("-" if v is None else f"${v:.2f}"))
    s = {k: v for k, v in r["skipped"].items() if v}
    if s:
        L += ["", "skipped: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in s.items())]
    return "\n".join(L)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", nargs="?", default=None,
                   help=f"receipts directory or file (default: ${ENV_PATH})")
    p.add_argument("--workload", default=None, help="only this socket id; also the exact somm workload name to join by")
    p.add_argument("--since", type=int, default=30, help="window in days (default 30)")
    p.add_argument("--project", default=None)
    p.add_argument("--json", action="store_true", default=False)


def cmd(args: argparse.Namespace, cfg) -> int:
    r = climb(args.path, args.workload, args.since, cfg.db_path, cfg.project)
    if r is None:
        print(f"no receipts at {args.path or os.environ.get(ENV_PATH) or '(no path given; set ' + ENV_PATH + ')'}",
              file=sys.stderr)
        return 1
    print(json.dumps(r, indent=2) if args.json else format_report(r))
    return 0


def main(argv: list[str] | None = None) -> int:
    from somm_core.config import load

    p = argparse.ArgumentParser(prog="somm climb", description=__doc__.split("\n")[0])
    add_arguments(p)
    args = p.parse_args(argv)
    return cmd(args, load(project=args.project, prefer_registry=True))


if __name__ == "__main__":
    raise SystemExit(main())

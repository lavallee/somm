"""somm CLI entry point — grouped subcommands.

Commands:
  somm status          roll-up per (workload, provider, model)
  somm tail            live call stream
  somm compare         run a prompt through N models side-by-side
  somm frontier        adequacy frontier per (provider, model) for a workload
  somm doctor          health check (config, ollama, db, model_intel, workers, cooldowns)
  somm serve           run the web admin + HTTP API (requires somm-service)
  somm spend           today's LLM spend vs daily budget cap per workload
  somm backfill-costs  recompute cost_usd for calls missing pricing
  somm plans           metered-plan quota usage + pacing
  somm drain-spool     replay spooled telemetry into the DB
  somm workload        register and inspect project workloads
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core import VERSION, list_intel
from somm_core.config import load as load_config
from somm_core.models import PrivacyClass
from somm_core.repository import Repository

from somm.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# somm workload


WORKLOAD_EXAMPLES: dict[str, dict] = {
    "structured-extraction": {
        "description": "Structured extraction workload",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Source text to extract from"}
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "quality_criteria": [
            "Return valid JSON matching the output schema.",
            "Extract only facts supported by the input text.",
        ],
    },
    "freeform": {
        "description": "",
        "input_schema": None,
        "output_schema": None,
        "quality_criteria": [],
    },
}


def _cmd_workload_add(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    template = WORKLOAD_EXAMPLES[args.from_example]
    description = args.description
    if description is None:
        description = template["description"]
    wl = repo.register_workload(
        name=args.name,
        project=cfg.project,
        description=description,
        input_schema=template["input_schema"],
        output_schema=template["output_schema"],
        quality_criteria=template["quality_criteria"],
        privacy_class=PrivacyClass(args.privacy_class),
    )
    print(f"registered workload {wl.name!r} for project {cfg.project!r}")
    print(f"privacy_class: {wl.privacy_class.value}")
    print(f"input_schema: {'yes' if wl.input_schema else 'no'}")
    print(f"output_schema: {'yes' if wl.output_schema else 'no'}")
    return 0


def _workload_rows(repo: Repository, project: str) -> list[dict]:
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT
                w.id,
                w.name,
                w.description,
                w.input_schema_json,
                w.output_schema_json,
                w.quality_criteria_json,
                w.budget_cap_usd_daily,
                w.privacy_class,
                w.capabilities_required_json,
                w.max_p95_latency_ms,
                w.max_capability_failure_rate,
                w.max_cost_per_call_usd,
                w.created_at,
                COUNT(c.id) AS call_count
            FROM workloads w
            LEFT JOIN calls c ON c.workload_id = w.id AND c.project = w.project
            WHERE w.project = ?
            GROUP BY w.id
            ORDER BY w.created_at DESC, w.name
            """,
            (project,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "description": r[2] or "",
            "input_schema": json.loads(r[3]) if r[3] else None,
            "output_schema": json.loads(r[4]) if r[4] else None,
            "quality_criteria": json.loads(r[5]) if r[5] else [],
            "budget_cap_usd_daily": r[6],
            "privacy_class": r[7],
            "capabilities_required": json.loads(r[8]) if r[8] else [],
            "max_p95_latency_ms": r[9],
            "max_capability_failure_rate": r[10],
            "max_cost_per_call_usd": r[11],
            "created_at": r[12],
            "call_count": r[13],
        }
        for r in rows
    ]


def _cmd_workload_list(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = _workload_rows(repo, cfg.project)
    if not rows:
        print(f"No workloads registered for project {cfg.project!r}.")
        print("Register one with `somm workload add <name>`.")
        return 0
    print(f"Project: {cfg.project}")
    print(f"{'name':<28} {'privacy':<10} {'calls':>8}")
    for row in rows:
        print(f"{row['name'][:27]:<28} {row['privacy_class']:<10} {row['call_count']:>8}")
    return 0


def _cmd_workload_show(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = [row for row in _workload_rows(repo, cfg.project) if row["name"] == args.name]
    if not rows:
        print(f"No workload {args.name!r} registered for project {cfg.project!r}.", file=sys.stderr)
        return 2
    row = rows[0]
    print(f"name: {row['name']}")
    print(f"project: {cfg.project}")
    print(f"id: {row['id']}")
    print(f"description: {row['description'] or '—'}")
    print(f"privacy_class: {row['privacy_class']}")
    print(f"call_count: {row['call_count']}")
    print(f"created_at: {row['created_at']}")
    print(f"budget_cap_usd_daily: {row['budget_cap_usd_daily'] if row['budget_cap_usd_daily'] is not None else '—'}")
    print(f"capabilities_required: {', '.join(row['capabilities_required']) or '—'}")
    print("constraints:")
    print(f"  max_p95_latency_ms: {row['max_p95_latency_ms'] if row['max_p95_latency_ms'] is not None else '—'}")
    print(f"  max_capability_failure_rate: {row['max_capability_failure_rate'] if row['max_capability_failure_rate'] is not None else '—'}")
    print(f"  max_cost_per_call_usd: {row['max_cost_per_call_usd'] if row['max_cost_per_call_usd'] is not None else '—'}")
    print("input_schema:")
    print(json.dumps(row["input_schema"], indent=2) if row["input_schema"] else "  —")
    print("output_schema:")
    print(json.dumps(row["output_schema"], indent=2) if row["output_schema"] else "  —")
    if row["quality_criteria"]:
        print("quality_criteria:")
        for item in row["quality_criteria"]:
            print(f"  - {item}")
    else:
        print("quality_criteria: —")
    return 0


# ---------------------------------------------------------------------------
# somm status


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    if getattr(args, "global_view", False):
        db_path = cfg.global_db_path
        if not db_path.exists():
            print(f"No global mirror at {db_path}.")
            print("Enable via SOMM_CROSS_PROJECT=1 and run a project with somm.")
            return 0
        repo = Repository(db_path)
        # Global status sums across projects; call with project=None semantics
        # via a single-table query.
        stats = _stats_global(repo, since_days=args.since)
        if not stats:
            print(f"Global mirror at {db_path} has no rows in the last {args.since} days.")
            return 0
        print(f"GLOBAL ({db_path})  ({args.since}d window)")
        print(
            f"{'project':<18} {'workload':<20} {'provider':<10} {'model':<16} "
            f"{'n':>6} {'tok_in':>8} {'tok_out':>8} {'cost':>10} {'fail':>6}"
        )
        for s in stats:
            cost_s = f"${(s['cost_usd'] or 0):.4f}"
            print(
                f"{s['project'][:17]:<18} {s['workload'][:19]:<20} {s['provider'][:9]:<10} "
                f"{s['model'][:15]:<16} {s['n_calls']:>6} {(s['tokens_in'] or 0):>8} "
                f"{(s['tokens_out'] or 0):>8} {cost_s:>10} {s['n_failed']:>6}"
            )
        return 0

    repo = Repository(cfg.db_path)
    stats = repo.stats_by_workload(cfg.project, since_days=args.since)
    if not stats:
        print(f"No calls yet for project {cfg.project!r} in the last {args.since} days.")
        print(f"Run `somm.llm({cfg.project!r}).generate(...)` in your Python code.")
        return 0
    print(f"Project: {cfg.project}  ({args.since}d window)")
    print(
        f"{'workload':<24} {'provider':<12} {'model':<18} "
        f"{'n':>6} {'tok_in':>8} {'tok_out':>8} {'cost':>10} {'fail':>6}"
    )
    for s in stats:
        cost_s = f"${(s['cost_usd'] or 0):.4f}"
        print(
            f"{s['workload'][:23]:<24} {s['provider'][:11]:<12} {s['model'][:17]:<18} "
            f"{s['n_calls']:>6} {(s['tokens_in'] or 0):>8} {(s['tokens_out'] or 0):>8} "
            f"{cost_s:>10} {s['n_failed']:>6}"
        )
    return 0


def _stats_global(repo, since_days: int) -> list[dict]:
    """Cross-project roll-up from the global mirror. Same shape as
    stats_by_workload but with `project` column."""
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT
                c.project,
                COALESCE(w.name, '(unregistered)') AS workload,
                c.provider, c.model,
                COUNT(*) AS n_calls,
                SUM(c.tokens_in) AS tokens_in,
                SUM(c.tokens_out) AS tokens_out,
                SUM(c.cost_usd) AS cost_usd,
                AVG(c.latency_ms) AS latency_ms_avg,
                SUM(CASE WHEN c.outcome != 'ok' THEN 1 ELSE 0 END) AS n_failed
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            WHERE c.ts >= datetime('now', ?)
            GROUP BY c.project, workload, c.provider, c.model
            ORDER BY cost_usd DESC NULLS LAST
            """,
            (f"-{since_days} days",),
        ).fetchall()
    return [
        {
            "project": r[0],
            "workload": r[1],
            "provider": r[2],
            "model": r[3],
            "n_calls": r[4],
            "tokens_in": r[5],
            "tokens_out": r[6],
            "cost_usd": r[7],
            "latency_ms_avg": r[8],
            "n_failed": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# somm tail


def _cmd_tail(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)

    # Seek cursor: start from now unless --since-minutes specified.
    since = datetime.now(UTC) - timedelta(minutes=args.since_minutes)
    seen_ids: set[str] = set()

    print(f"Tailing calls for project {cfg.project!r} (Ctrl-C to exit)")
    print(
        f"{'time':<20} {'workload':<20} {'provider':<10} {'model':<22} "
        f"{'lat_ms':>7} {'tok_i':>6} {'tok_o':>6} {'cost':>8}  outcome"
    )
    try:
        while True:
            rows = _fetch_since(repo, cfg.project, since, workload=args.workload)
            for r in rows:
                if r["id"] in seen_ids:
                    continue
                seen_ids.add(r["id"])
                ts = r["ts"][:19].replace("T", " ")
                cost_s = f"${r['cost_usd']:.4f}"
                print(
                    f"{ts:<20} {r['workload'][:19]:<20} {r['provider'][:9]:<10} "
                    f"{r['model'][:21]:<22} {r['latency_ms']:>7} {r['tokens_in']:>6} "
                    f"{r['tokens_out']:>6} {cost_s:>8}  {r['outcome']}"
                )
                # Advance cursor past the newest seen row
                row_ts = datetime.fromisoformat(r["ts"])
                if row_ts.tzinfo is None:
                    row_ts = row_ts.replace(tzinfo=UTC)
                if row_ts > since:
                    since = row_ts
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("")
        return 0


def _fetch_since(
    repo: Repository, project: str, since: datetime, workload: str | None = None
) -> list[dict]:
    q = [
        "SELECT c.id, c.ts, COALESCE(w.name, '(unregistered)') AS workload, "
        "       c.provider, c.model, c.latency_ms, c.tokens_in, c.tokens_out, "
        "       c.cost_usd, c.outcome "
        "FROM calls c LEFT JOIN workloads w ON w.id = c.workload_id "
        "WHERE c.project = ? AND c.ts > ? "
    ]
    params: list = [project, since.isoformat()]
    if workload:
        q.append("AND w.name = ? ")
        params.append(workload)
    q.append("ORDER BY c.ts ASC LIMIT 200")

    with repo._open() as conn:
        rows = conn.execute("".join(q), params).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "workload": r[2],
            "provider": r[3],
            "model": r[4],
            "latency_ms": r[5],
            "tokens_in": r[6],
            "tokens_out": r[7],
            "cost_usd": r[8] or 0.0,
            "outcome": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# somm compare


def _cmd_compare(args: argparse.Namespace) -> int:
    """Run a prompt through N models side-by-side. Non-routed, explicit picks.

    Use: somm compare "Summarize X" --models ollama/gemma4:e4b,openai/gpt-4o-mini
    """
    from somm import SommLLM

    specs = _parse_model_specs(args.models)
    if not specs:
        print("no --models supplied. example:", file=sys.stderr)
        print("  somm compare 'hi' --models ollama/gemma4:e4b,openai/gpt-4o-mini", file=sys.stderr)
        return 2

    llm = SommLLM(project=args.project or "compare")
    try:
        results = []
        for provider_name, model in specs:
            # Ensure this provider is in the chain
            if provider_name not in {p.name for p in llm.providers}:
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": f"provider {provider_name!r} not configured (missing env key?)",
                    }
                )
                continue
            try:
                r = llm.generate(
                    prompt=args.prompt,
                    workload=args.workload,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    model=model,
                    provider=provider_name,
                )
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "text": r.text,
                        "tokens_in": r.tokens_in,
                        "tokens_out": r.tokens_out,
                        "latency_ms": r.latency_ms,
                        "cost_usd": r.cost_usd,
                        "outcome": r.outcome.value,
                        "call_id": r.call_id,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
    finally:
        llm.close()

    _print_comparison(results, truncate=args.truncate)
    return 0


def _parse_model_specs(raw: list[str] | None) -> list[tuple[str, str]]:
    """Parse --models. Accepts 'provider/model' or 'provider:model'. Comma-
    separated or repeated --models flag."""
    if not raw:
        return []
    specs: list[tuple[str, str]] = []
    for item in raw:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue
            if "/" in token:
                p, _, m = token.partition("/")
            elif ":" in token:
                p, _, m = token.partition(":")
            else:
                p, m = token, ""
            specs.append((p, m))
    return specs


def _print_comparison(results: list[dict], truncate: int = 240) -> None:
    print(
        f"{'provider':<12} {'model':<26} {'lat_ms':>7} {'tok_i':>6} "
        f"{'tok_o':>6} {'cost':>10}  response"
    )
    for r in results:
        provider = r["provider"][:11]
        model = r["model"][:25]
        if "error" in r:
            print(
                f"{provider:<12} {model:<26} {'—':>7} {'—':>6} {'—':>6} "
                f"{'—':>10}  ERROR: {r['error']}"
            )
            continue
        cost_s = f"${r['cost_usd']:.4f}"
        text = r["text"].replace("\n", "\\n")
        if len(text) > truncate:
            text = text[:truncate] + "…"
        print(
            f"{provider:<12} {model:<26} {r['latency_ms']:>7} {r['tokens_in']:>6} "
            f"{r['tokens_out']:>6} {cost_s:>10}  {text}"
        )
        print(
            f"{'':<12} {'':<26} {'':>7} {'':>6} {'':>6} {'':>10}  [{r['outcome']} · {r['call_id']}]"
        )


# ---------------------------------------------------------------------------
# somm doctor (enhanced)


def _cmd_frontier(args: argparse.Namespace) -> int:
    """Print the adequacy frontier for one workload.

    For each (provider, model) the workload has touched in the window:
    capability-failure rate, detractor rate, p50/p95 latency, mean cost
    per ok call, and whether each workload constraint is exceeded.
    """
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = repo.workload_by_name(args.workload, cfg.project)
    if wl is None:
        print(
            f"No workload {args.workload!r} registered for project {cfg.project!r}.\n"
            f"Register one with somm.workload(...) in your code, then re-run.",
            file=sys.stderr,
        )
        return 2

    rows = repo.workload_frontier(wl.id, since_days=args.since)
    if not rows:
        print(
            f"No calls for workload {wl.name!r} in the last {args.since}d. "
            f"Run the workload, then come back."
        )
        return 0

    cons = {
        "max_p95_latency_ms": wl.max_p95_latency_ms,
        "max_capability_failure_rate": wl.max_capability_failure_rate,
        "max_cost_per_call_usd": wl.max_cost_per_call_usd,
    }
    print(f"Workload: {wl.name}  ({args.since}d window)")
    cons_pretty = ", ".join(
        f"{k.removeprefix('max_')}≤{v}" for k, v in cons.items() if v is not None
    )
    print(f"Constraints: {cons_pretty or '(none set — inspect with `somm workload show <name>`)'}")
    print(
        f"\n{'provider':<14} {'model':<28} {'n':>5} {'cap%':>6} {'det%':>6} "
        f"{'p50ms':>7} {'p95ms':>7} {'$/ok':>9} fitness"
    )
    for r in rows:
        cap_pct = 100.0 * r["capability_failure_rate"]
        det_pct = 100.0 * r["detractor_rate"]
        p50 = r["p50_latency_ms"] if r["p50_latency_ms"] is not None else "-"
        p95 = r["p95_latency_ms"] if r["p95_latency_ms"] is not None else "-"
        cost = r["mean_cost_per_ok_call"]
        cost_s = f"${cost:.5f}" if cost is not None else "-"
        flags = []
        f = r["fitness"]
        if f["exceeds_max_capability_failure_rate"]:
            flags.append("UNFIT(cap)")
        if f["exceeds_max_p95_latency_ms"]:
            flags.append("UNFIT(slow)")
        if f["exceeds_max_cost_per_call_usd"]:
            flags.append("UNFIT($)")
        fitness_s = ",".join(flags) if flags else "ok"
        print(
            f"{r['provider'][:13]:<14} {r['model'][:27]:<28} {r['n_calls']:>5} "
            f"{cap_pct:>5.1f}% {det_pct:>5.1f}% {str(p50):>7} {str(p95):>7} {cost_s:>9} {fitness_s}"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo_exists = cfg.db_path.exists()
    print(f"somm v{VERSION}")
    print(f"project: {cfg.project}  mode: {cfg.mode}")
    print(f"db_path: {cfg.db_path}   exists: {repo_exists}")
    if repo_exists:
        mode = oct(cfg.db_path.stat().st_mode)[-3:]
        ok = mode in ("600",)
        print(f"db perms: {mode} {'(ok)' if ok else '(WARN — expect 600)'}")

    ok_overall = True

    # Ollama
    p = OllamaProvider(base_url=cfg.ollama_url, default_model=cfg.ollama_model)
    h = p.health()
    ok_ollama = h.available
    print(f"ollama:   {'ok' if ok_ollama else 'UNAVAILABLE'}  ({h.detail})")
    ok_overall = ok_overall and ok_ollama

    if not repo_exists:
        print("db missing — skipping intel/workers/cooldowns checks")
        return 0 if ok_overall else 1

    repo = Repository(cfg.db_path)

    # Model intel freshness
    rows = list_intel(repo)
    if not rows:
        print("model_intel: empty (run `somm-serve admin refresh-intel` to populate)")
    else:
        # Bucket by source; show latest last_seen
        by_src: dict[str, list] = {}
        for r in rows:
            by_src.setdefault(r["source"], []).append(r)
        print(f"model_intel: {len(rows)} entries across {len(by_src)} source(s)")
        for src, entries in sorted(by_src.items()):
            latest = max((e["last_seen"] or "") for e in entries)
            age = _age_since(latest) if latest else "—"
            print(f"  {src:<16} {len(entries):>5} models   latest {age}")

    # Worker heartbeats
    with repo._open() as conn:
        heartbeat_rows = conn.execute(
            "SELECT worker_name, last_run_at, last_success_at, consecutive_failures "
            "FROM worker_heartbeat ORDER BY worker_name"
        ).fetchall()
    if not heartbeat_rows:
        print("worker_heartbeat: no heartbeats recorded")
    else:
        print("worker_heartbeat:")
        print(
            f"  {'worker_name':<16} {'last_run_at':<19} "
            f"{'last_success_at':<19} consecutive_failures"
        )
        for name, last_run_at, last_success_at, failures in heartbeat_rows:
            print(
                f"  {name[:15]:<16} {(last_run_at or 'never'):<19} "
                f"{(last_success_at or 'never'):<19} {failures}"
            )

    # Cooldowns
    with repo._open() as conn:
        now = datetime.now(UTC).isoformat()
        cool_rows = conn.execute(
            "SELECT provider, model, cooldown_until, consecutive_failures "
            "FROM provider_health WHERE cooldown_until > ? "
            "ORDER BY cooldown_until",
            (now,),
        ).fetchall()
    if not cool_rows:
        print("cooldowns: none active")
    else:
        print(f"cooldowns: {len(cool_rows)} active")
        for provider, model, until, failures in cool_rows:
            remaining = _age_until(until)
            slot = f"{provider}/{model}" if model else provider
            print(f"  {slot:<42} expires in {remaining}   (failures={failures})")

    return 0 if ok_overall else 1


def _age_since(iso: str) -> str:
    if not iso:
        return "never"
    try:
        dt = datetime.fromisoformat(iso.replace(" ", "T"))
    except (ValueError, TypeError):
        return iso[:19]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - dt
    return _fmt_delta(delta) + " ago"


def _age_until(iso: str) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace(" ", "T"))
    except (ValueError, TypeError):
        return iso[:19]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt - datetime.now(UTC)
    return _fmt_delta(delta) if delta.total_seconds() > 0 else "now"


def _fmt_delta(delta: timedelta) -> str:
    s = int(abs(delta.total_seconds()))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


# ---------------------------------------------------------------------------
# somm spend


def spend_today(
    db_path: Path,
    project: str,
    default_cap: float | None,
) -> list[dict]:
    """Query today's (UTC) spend per workload. Read-only; no writes.

    Returns list of dicts sorted by spent_usd desc:
      {"workload": str, "spent_usd": float, "cap_usd": float | None}

    cap_usd is the workload's budget_cap_usd_daily; falls back to
    default_cap (config-level ceiling); None when neither is set.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(w.name, '(unregistered)') AS workload,
                SUM(c.cost_usd) AS spent_usd,
                MAX(w.budget_cap_usd_daily) AS cap_usd
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            WHERE c.project = ?
              AND date(c.ts) = date('now')
            GROUP BY COALESCE(w.name, '(unregistered)')
            ORDER BY spent_usd DESC
            """,
            (project,),
        ).fetchall()

    result = []
    for name, spent, cap in rows:
        effective_cap = cap if cap is not None else default_cap
        result.append(
            {
                "workload": name,
                "spent_usd": float(spent or 0.0),
                "cap_usd": effective_cap,
            }
        )
    return result


_CATALOG_STALE_DAYS = 90


def _cmd_plans(args: argparse.Namespace) -> int:
    from somm_core.plans import limit_statuses, load_catalog, load_plans, plans_path
    from somm_core.registry import fleet_db_paths

    catalog = load_catalog()
    if args.catalog:
        if not catalog:
            print("plan catalog is empty")
            return 0
        print("Known plans (bundled catalog — verify against source before trusting):\n")
        for key in sorted(catalog):
            e = catalog[key]
            age = e.age_days()
            age_s = f"verified {age}d ago" if age is not None else "unverified"
            lims = "; ".join(
                f"{lim.quota:g} {lim.unit}/{lim.window}" for lim in e.limits
            ) or "no limits recorded"
            print(f"  {key:<28} {e.display}")
            print(f"    {lims}  [{age_s}]")
            if e.notes:
                print(f"    note: {e.notes}")
            if e.source:
                print(f"    source: {e.source}")
        return 0

    cfg = load_config(project=args.project)
    try:
        plans = load_plans(catalog=catalog)
    except Exception as exc:
        print(f"plans.toml is invalid: {exc}")
        return 1
    if not plans:
        print(f"No plans declared. Create {plans_path()} — example:")
        print(
            "\n  [minimax]"
            '\n  mode = "metered"'
            '\n  plan = "coding-pro"'
            "\n  soft_target_pct = 80"
            "\n  enforce = false"
            "\n  [[minimax.limits]]"
            '\n  window = "month"      # or rolling: "5h", "7d", "1w"'
            "\n  anchor_day = 1        # calendar reset day"
            "\n  quota = 40.0"
            '\n  unit = "usd_equiv"    # requests | tokens_in | tokens_out | tokens_total | usd_equiv'
            "\n\n  [gemini]"
            '\n  mode = "payg"'
        )
        return 0

    dbs = (
        [cfg.db_path]
        if args.project_only
        else fleet_db_paths(include=cfg.db_path if cfg.db_path.exists() else None)
    )
    statuses = limit_statuses(dbs, plans)
    scope = "this project only" if args.project_only else f"fleet ({len(dbs)} project DBs)"

    if args.json:
        out = []
        for st in statuses:
            out.append({
                "provider": st.provider,
                "plan": st.plan_name,
                "window": st.limit.window,
                "unit": st.limit.unit,
                "used": round(st.used, 4),
                "quota": st.limit.quota,
                "used_pct": round(st.used_pct, 1),
                "elapsed_pct": round(st.elapsed_pct, 1),
                "pace_ratio": round(st.pace_ratio, 2),
                "projected_pct": round(st.projected_pct, 1),
                "window_end": st.window_end.isoformat(),
                "state": st.state,
                "mode": st.mode,
            })
        from somm_core.plans import payg_burn_rates

        burn = [
            {
                "provider": b.provider,
                "spend_1d": round(b.spend_1d, 4),
                "spend_7d": round(b.spend_7d, 4),
                "spend_30d": round(b.spend_30d, 4),
                "per_day": round(b.per_day, 4),
                "projected_month": round(b.projected_month, 2),
            }
            for b in payg_burn_rates(dbs, plans)
        ]
        print(json.dumps({"scope": scope, "limits": out, "payg_burn": burn}, indent=1))
        return 0

    print(f"Plan usage — {scope}\n")
    if statuses:
        print(
            f"{'provider':<12} {'plan/budget':<14} {'window':<7} {'used / quota':>26} "
            f"{'used%':>6} {'elapsed%':>8} {'pace':>6} {'proj%':>6}  state"
        )
        for st in statuses:
            used_s = (
                f"${st.used:,.2f} / ${st.limit.quota:,.2f}"
                if st.limit.unit == "usd_equiv"
                else f"{st.used:,.0f} / {st.limit.quota:,.0f} {st.limit.unit}"
            )
            label = st.plan_name or ("budget" if st.mode == "payg" else "—")
            print(
                f"{st.provider:<12} {label:<14} {st.limit.window:<7} "
                f"{used_s:>26} {st.used_pct:>5.0f}% {st.elapsed_pct:>7.0f}% "
                f"{st.pace_ratio:>5.1f}x {st.projected_pct:>5.0f}%  {st.state}"
            )

    # Value multiple: what a metered subscription delivered vs. its price,
    # in notional list-price dollars this calendar month.
    from somm_core.plans import PlanLimit as _PL
    from somm_core.plans import usage_in_window as _usage

    value_lines = []
    for prov, pl in sorted(plans.items()):
        if pl.mode != "metered" or pl.price_usd_month <= 0:
            continue
        notional = _usage(dbs, prov, _PL(window="month", quota=1, unit="usd_equiv"))
        if notional > 0:
            mult = notional / pl.price_usd_month
            value_lines.append(
                f"  {prov:<12} ${notional:,.2f} list-price value this month "
                f"on a ${pl.price_usd_month:,.0f}/mo plan (≈{mult:.1f}x)"
            )
    if value_lines:
        print("\nplan value (notional list-price consumed vs subscription price):")
        for line in value_lines:
            print(line)

    metered_no_limits = [
        p for p, pl in plans.items() if pl.mode == "metered" and not pl.limits
    ]
    if metered_no_limits:
        print(
            f"\nmetered, no limits declared (labelled only): "
            f"{', '.join(sorted(metered_no_limits))}"
        )

    # PAYG burn rates: no vendor window here — the number that matters is
    # velocity and where it lands by month-end.
    from somm_core.plans import payg_burn_rates

    burn = payg_burn_rates(dbs, plans)
    if burn:
        print("\npayg burn (real dollars):")
        print(f"  {'provider':<12} {'1d':>9} {'7d':>9} {'30d':>9} {'$/day':>8} {'→month':>9}")
        for b in burn:
            print(
                f"  {b.provider:<12} ${b.spend_1d:>8.2f} ${b.spend_7d:>8.2f} "
                f"${b.spend_30d:>8.2f} ${b.per_day:>7.2f} "
                f"${b.projected_month:>8.2f}"
            )
    free = sorted(p for p, pl in plans.items() if pl.mode == "free")
    if free:
        print(f"\nfree/local: {', '.join(free)}")

    # Empirical ceilings: vendors stopped publishing numeric limits, but
    # every quota-429 in your own telemetry is ground truth.
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    from somm_core.plans import observed_ceilings, recent_ok_calls

    all_ceilings: dict[str, list] = {}
    printed_header = False
    for prov, pl in sorted(plans.items()):
        if pl.mode != "metered":
            continue
        ceilings = observed_ceilings(dbs, prov)
        if not ceilings:
            continue
        all_ceilings[prov] = ceilings
        if not printed_header:
            print(
                "\nobserved ceilings — inferred from your own quota errors "
                "(median trailing-window usage at each 429):"
            )
            printed_header = True
        for c in ceilings:
            est = f"${c.estimate:,.2f}" if c.unit == "usd_equiv" else f"{c.estimate:,.0f} {c.unit}"
            print(
                f"  {prov:<12} ~{est}/{c.window}  "
                f"(from {c.n_events} event(s), last {c.last_event[:10]})"
            )

    # Quota drift: declared limits are guesses/marketing copy; vendors
    # reset and resize quotas without notice. Two tell-tales, both from
    # your own telemetry:
    #   1. usage exceeds a declared quota while calls keep succeeding
    #      → the real limit is higher (raised, reset, or wrong guess);
    #   2. recent 429-derived ceilings diverge >25% from the declared
    #      quota → the real limit moved.
    for st in statuses:
        if st.mode != "metered":
            continue
        if st.used_pct >= 100 and recent_ok_calls(dbs, st.provider, hours=24) > 0:
            print(
                f"\n⚠ {st.provider}: usage is {st.used_pct:.0f}% of the declared "
                f"{st.limit.window} quota but calls are still succeeding — the "
                f"real limit is likely higher (raised or reset). Update the "
                f"quota in plans.toml, or trust the observed ceilings above."
            )
            continue
        for c in all_ceilings.get(st.provider, []):
            if c.window != st.limit.window or c.unit != st.limit.unit:
                continue
            try:
                last = _dt.fromisoformat(c.last_event)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=_UTC)
                if (_dt.now(_UTC) - last).days > 14:
                    continue  # stale evidence; don't second-guess with it
            except ValueError:
                continue
            drift = abs(c.estimate - st.limit.quota) / st.limit.quota if st.limit.quota else 0
            if drift > 0.25:
                direction = "lower" if c.estimate < st.limit.quota else "higher"
                print(
                    f"\n⚠ {st.provider}: recent 429s put the real "
                    f"{st.limit.window} ceiling ~{drift * 100:.0f}% {direction} "
                    f"than the declared quota ({c.estimate:,.0f} vs "
                    f"{st.limit.quota:,.0f} {st.limit.unit}) — the vendor may "
                    f"have changed it; consider updating plans.toml."
                )

    # Staleness: plan limits are vendor marketing copy, not an API.
    # Nudge re-verification when a referenced catalog entry ages out.
    for pl in plans.values():
        if not pl.catalog_ref:
            continue
        entry = catalog.get(pl.catalog_ref)
        if entry is None:
            continue
        age = entry.age_days()
        if age is None or age > _CATALOG_STALE_DAYS:
            age_s = f"{age}d ago" if age is not None else "never"
            print(
                f"\n⚠ catalog entry {pl.catalog_ref} last verified {age_s} — "
                f"limits may have changed; check {entry.source or 'the vendor page'}"
            )
    return 0


def _cmd_drain_spool(args: argparse.Namespace) -> int:
    from somm_core.repository import Repository

    from somm.telemetry import drain_spool

    cfg = load_config(project=args.project)
    if not cfg.db_path.exists():
        print("no telemetry database found")
        return 1
    n = drain_spool(Repository(cfg.db_path), cfg.spool_dir)
    print(f"drained {n} spooled call(s) into {cfg.db_path}")
    return 0


def _cmd_backfill_costs(args: argparse.Namespace) -> int:
    from somm_core.pricing import backfill_costs, sync_bundled_pricing
    from somm_core.repository import Repository

    cfg = load_config(project=args.project)
    if not cfg.db_path.exists():
        print("no telemetry database found")
        return 1
    repo = Repository(cfg.db_path)
    # Make sure current pricing intel is present before joining against it,
    # so backfill works even on a DB the library hasn't opened since upgrade.
    synced = sync_bundled_pricing(repo)
    if synced:
        print(f"synced {synced} pricing row(s) from the bundled snapshot")
    n, total = backfill_costs(repo, since_days=args.since, dry_run=args.dry_run)
    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {n} call(s), ${total:.4f} in previously untracked spend")
    if args.dry_run and n:
        print("re-run without --dry-run to apply")
    return 0


def _cmd_spend(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    use_json = getattr(args, "json", False)

    if not cfg.db_path.exists():
        print("[]" if use_json else "no spend recorded today")
        return 0

    rows = spend_today(cfg.db_path, cfg.project, cfg.budget_default_cap_usd_daily)

    if use_json:
        out = []
        for r in rows:
            cap = r["cap_usd"]
            pct = None if cap is None or cap == 0.0 else round(100.0 * r["spent_usd"] / cap, 4)
            out.append(
                {
                    "workload": r["workload"],
                    "spent_usd": r["spent_usd"],
                    "cap_usd": cap,
                    "pct_of_cap": pct,
                }
            )
        print(json.dumps(out, indent=2))
        return 0

    if not rows:
        print("no spend recorded today")
        return 0

    print(f"{'workload':<28} {'spent':>10} {'cap':>10} {'pct':>8}")
    print("-" * 60)
    for r in rows:
        name = r["workload"][:27]
        spent_s = f"${r['spent_usd']:.2f}"
        cap = r["cap_usd"]
        if cap is None:
            cap_s = "—"
            pct_s = "—"
        elif cap == 0.0:
            cap_s = f"${cap:.2f}"
            pct_s = "∞"
        else:
            cap_s = f"${cap:.2f}"
            pct_s = f"{100.0 * r['spent_usd'] / cap:.1f}%"
        print(f"{name:<28} {spent_s:>10} {cap_s:>10} {pct_s:>8}")

    return 0


# ---------------------------------------------------------------------------
# somm serve (thin shim to somm-service)


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from somm_service.cli import main as serve_main
    except ImportError:
        print(
            "somm serve requires somm-service.\n"
            "  uv add somm-service    # or: pip install somm-service",
            file=sys.stderr,
        )
        return 2
    forwarded = []
    if args.project:
        forwarded += ["--project", args.project]
    forwarded += ["--host", args.host, "--port", str(args.port)]
    return serve_main(forwarded)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm", description="somm — self-hosted LLM telemetry")
    p.add_argument("--version", action="version", version=f"somm {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("status", help="show call roll-up for the current project")
    ps.add_argument("--project", default=None)
    ps.add_argument("--since", type=int, default=7, help="window in days (default 7)")
    ps.add_argument(
        "--global",
        dest="global_view",
        action="store_true",
        help="read from ~/.somm/global.sqlite (requires SOMM_CROSS_PROJECT)",
    )
    ps.set_defaults(func=_cmd_status)

    pt = sub.add_parser("tail", help="stream new calls as they land")
    pt.add_argument("--project", default=None)
    pt.add_argument("--workload", default=None, help="filter to a single workload")
    pt.add_argument(
        "--since-minutes", type=int, default=0, help="start from N minutes ago (default: now)"
    )
    pt.add_argument("--poll-interval", type=float, default=0.5)
    pt.set_defaults(func=_cmd_tail)

    pc = sub.add_parser("compare", help="run a prompt through N models side-by-side")
    pc.add_argument("prompt", help="the prompt text")
    pc.add_argument(
        "--models",
        action="append",
        required=True,
        help="provider/model (repeatable or comma-separated)",
    )
    pc.add_argument("--project", default=None)
    pc.add_argument("--workload", default="compare")
    pc.add_argument("--max-tokens", type=int, default=256)
    pc.add_argument("--temperature", type=float, default=0.0)
    pc.add_argument(
        "--truncate",
        type=int,
        default=240,
        help="truncate response text at N chars (0 = no truncate)",
    )
    pc.set_defaults(func=_cmd_compare)

    pf = sub.add_parser(
        "frontier",
        help="adequacy frontier per (provider, model) for one workload",
    )
    pf.add_argument("--workload", required=True, help="workload name to inspect")
    pf.add_argument("--project", default=None)
    pf.add_argument("--since", type=int, default=30, help="window in days (default 30)")
    pf.set_defaults(func=_cmd_frontier)

    pw = sub.add_parser("workload", help="register and inspect project workloads")
    pw_sub = pw.add_subparsers(dest="workload_cmd", required=True)

    pwa = pw_sub.add_parser("add", help="register a workload in the project DB")
    pwa.add_argument("name", help="workload name")
    pwa.add_argument("--project", default=None)
    pwa.add_argument("--description", default=None)
    pwa.add_argument(
        "--privacy-class",
        choices=[pc.value for pc in PrivacyClass],
        default=PrivacyClass.INTERNAL.value,
    )
    pwa.add_argument(
        "--from-example",
        choices=sorted(WORKLOAD_EXAMPLES),
        default="freeform",
        help="built-in workload template (default: freeform)",
    )
    pwa.set_defaults(func=_cmd_workload_add)

    pwl = pw_sub.add_parser("list", help="list registered workloads")
    pwl.add_argument("--project", default=None)
    pwl.set_defaults(func=_cmd_workload_list)

    pws = pw_sub.add_parser("show", help="show one registered workload")
    pws.add_argument("name", help="workload name")
    pws.add_argument("--project", default=None)
    pws.set_defaults(func=_cmd_workload_show)

    pd = sub.add_parser("doctor", help="check config + ollama + db + intel + workers + cooldowns")
    pd.add_argument("--project", default=None)
    pd.set_defaults(func=_cmd_doctor)

    psr = sub.add_parser("serve", help="run the web admin + HTTP API (localhost:7878)")
    psr.add_argument("--project", default=None)
    psr.add_argument("--host", default="127.0.0.1")
    psr.add_argument("--port", type=int, default=7878)
    psr.set_defaults(func=_cmd_serve)

    pspend = sub.add_parser(
        "spend", help="today's LLM spend vs daily budget cap per workload"
    )
    pspend.add_argument("--project", default=None)
    pspend.add_argument(
        "--json",
        dest="json",
        action="store_true",
        default=False,
        help="emit a JSON array instead of the aligned table",
    )
    pspend.set_defaults(func=_cmd_spend)

    pbf = sub.add_parser(
        "backfill-costs",
        help="recompute cost_usd for $0 calls that now have pricing intel",
    )
    pbf.add_argument("--project", default=None)
    pbf.add_argument("--since", type=int, default=None, help="only calls from the last N days")
    pbf.add_argument("--dry-run", action="store_true", help="report without writing")
    pbf.set_defaults(func=_cmd_backfill_costs)

    ppl = sub.add_parser(
        "plans",
        help="metered-plan quota usage + pacing (PAYG vs metered, fleet-wide)",
    )
    ppl.add_argument("--project", default=None)
    ppl.add_argument(
        "--project-only",
        action="store_true",
        help="count only this project's usage (default: whole fleet — quotas are shared)",
    )
    ppl.add_argument("--json", action="store_true")
    ppl.add_argument(
        "--catalog",
        action="store_true",
        help="list known plans from the bundled catalog (with sources + verified dates)",
    )
    ppl.set_defaults(func=_cmd_plans)

    pds = sub.add_parser(
        "drain-spool",
        help="replay spooled JSONL telemetry (written during DB outages) into the DB",
    )
    pds.add_argument("--project", default=None)
    pds.set_defaults(func=_cmd_drain_spool)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

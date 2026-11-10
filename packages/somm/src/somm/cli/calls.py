"""status, cache-advice, generate, bench, tail, compare, frontier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core.cost import cost_is_known
from somm_core.models import Outcome
from somm_core.read import search_calls
from somm_core.repository import Repository

from somm.cli import _common
from somm.compare import compare_models, parse_model_specs


def _fmt_stat_int(value: object) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def _fmt_stat_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _fmt_cost(row: dict) -> str:
    """A row's cost; ``unknown`` when its 0.0 means somm had no price."""
    if not row.get("cost_known", cost_is_known(row.get("cost_source"))):
        return "unknown"
    return f"${float(row['cost_usd'] or 0.0):.4f}"


def _mark_cost_known(repo: Repository | None, rows: list[dict]) -> None:
    """Set ``cost_known`` on result rows from their ledger rows' cost_source.

    Run after the client closed (its writer flushed); a row whose call is not
    in the ledger is left as known.
    """
    if repo is None:
        return
    for row in rows:
        call = repo.get_call(row["call_id"]) if row.get("call_id") else None
        row["cost_known"] = call is None or cost_is_known(call.cost_source)


def _fmt_stat_pct(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.0f}%"


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    if getattr(args, "global_view", False):
        db_path = cfg.global_db_path
        if not db_path.exists():
            if getattr(args, "json", False):
                print(
                    json.dumps(
                        {
                            "scope": "global",
                            "project": cfg.project,
                            "db_path": str(db_path),
                            "window_days": args.since,
                            "count": 0,
                            "rows": [],
                        },
                        indent=2,
                    )
                )
                return 0
            print(f"No global mirror at {db_path}.")
            print("Enable via SOMM_CROSS_PROJECT=1 and run a project with somm.")
            return 0
        repo = Repository(db_path)
        # Global status sums across projects; call with project=None semantics
        # via a single-table query.
        stats = _stats_global(repo, since_days=args.since)
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "scope": "global",
                        "project": cfg.project,
                        "db_path": str(db_path),
                        "window_days": args.since,
                        "count": len(stats),
                        "rows": stats,
                    },
                    indent=2,
                )
            )
            return 0
        if not stats:
            print(f"Global mirror at {db_path} has no rows in the last {args.since} days.")
            return 0
        print(f"GLOBAL ({db_path})  ({args.since}d window)")
        print(
            f"{'project':<18} {'workload':<20} {'provider':<10} {'model':<16} "
            f"{'n':>6} {'tok_in':>8} {'tok_out':>8} {'cost':>10} {'fail':>6} "
            f"{'p95':>7} {'ttft95':>7} {'tpot':>7} {'out/s':>8} {'cache':>6} {'good':>6}"
        )
        for s in stats:
            cost_s = f"${(s['cost_usd'] or 0):.4f}"
            print(
                f"{s['project'][:17]:<18} {s['workload'][:19]:<20} {s['provider'][:9]:<10} "
                f"{s['model'][:15]:<16} {s['n_calls']:>6} {(s['tokens_in'] or 0):>8} "
                f"{(s['tokens_out'] or 0):>8} {cost_s:>10} {s['n_failed']:>6} "
                f"{_fmt_stat_int(s.get('p95_latency_ms')):>7} "
                f"{_fmt_stat_int(s.get('p95_ttft_ms')):>7} "
                f"{_fmt_stat_float(s.get('tpot_ms')):>7} "
                f"{_fmt_stat_float(s.get('output_tokens_per_second')):>8} "
                f"{_fmt_stat_pct(s.get('cache_read_ratio')):>6} "
                f"{_fmt_stat_pct(s.get('goodput_under_slo')):>6}"
            )
        return 0

    repo = Repository(cfg.db_path)
    stats = repo.stats_by_workload(cfg.project, since_days=args.since)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "scope": "project",
                    "project": cfg.project,
                    "db_path": str(cfg.db_path),
                    "window_days": args.since,
                    "count": len(stats),
                    "rows": stats,
                },
                indent=2,
            )
        )
        return 0
    if not stats:
        print(f"No calls yet for project {cfg.project!r} in the last {args.since} days.")
        print(f"Run `somm.llm({cfg.project!r}).generate(...)` in your Python code.")
        return 0
    print(f"Project: {cfg.project}  ({args.since}d window)")
    print(
        f"{'workload':<24} {'provider':<12} {'model':<18} "
        f"{'n':>6} {'tok_in':>8} {'tok_out':>8} {'cost':>10} {'fail':>6} "
        f"{'p95':>7} {'ttft95':>7} {'tpot':>7} {'out/s':>8} {'cache':>6} {'good':>6}"
    )
    for s in stats:
        cost_s = f"${(s['cost_usd'] or 0):.4f}"
        print(
            f"{s['workload'][:23]:<24} {s['provider'][:11]:<12} {s['model'][:17]:<18} "
            f"{s['n_calls']:>6} {(s['tokens_in'] or 0):>8} {(s['tokens_out'] or 0):>8} "
            f"{cost_s:>10} {s['n_failed']:>6} "
            f"{_fmt_stat_int(s.get('p95_latency_ms')):>7} "
            f"{_fmt_stat_int(s.get('p95_ttft_ms')):>7} "
            f"{_fmt_stat_float(s.get('tpot_ms')):>7} "
            f"{_fmt_stat_float(s.get('output_tokens_per_second')):>8} "
            f"{_fmt_stat_pct(s.get('cache_read_ratio')):>6} "
            f"{_fmt_stat_pct(s.get('goodput_under_slo')):>6}"
        )
    return 0


def _prefix_cache_advice(
    stats: list[dict],
    *,
    min_tokens_in: int,
    max_cache_read_ratio: float,
) -> list[dict]:
    out: list[dict] = []
    for row in stats:
        tokens_in = int(row.get("tokens_in") or 0)
        if tokens_in < min_tokens_in:
            continue
        cache_read_ratio = row.get("cache_read_ratio")
        effective_ratio = float(cache_read_ratio or 0.0)
        if effective_ratio > max_cache_read_ratio:
            continue
        cache_tokens_in = int(row.get("cache_tokens_in") or 0)
        n_calls = int(row.get("n_calls") or 0)
        issue = "no_cache_reads" if cache_tokens_in == 0 else "low_cache_reuse"
        out.append(
            {
                "workload": row["workload"],
                "provider": row["provider"],
                "model": row["model"],
                "n_calls": n_calls,
                "tokens_in": tokens_in,
                "cache_tokens_in": cache_tokens_in,
                "cache_read_ratio": effective_ratio,
                "issue": issue,
                "advice": (
                    "Stabilize repeated system/context prefixes and batch calls with "
                    "shared prefixes together so provider prefix caches can hit."
                ),
            }
        )
    return sorted(out, key=lambda row: (row["cache_read_ratio"], -row["tokens_in"]))


def _cmd_cache_advice(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    stats = repo.stats_by_workload(cfg.project, since_days=args.since)
    rows = _prefix_cache_advice(
        stats,
        min_tokens_in=args.min_tokens_in,
        max_cache_read_ratio=args.max_cache_read_ratio,
    )
    payload = {
        "project": cfg.project,
        "window_days": args.since,
        "count": len(rows),
        "rows": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not rows:
        print("No prefix-cache opportunities found for the current thresholds.")
        return 0
    print(f"Prefix-cache advice: {cfg.project}  ({args.since}d window)")
    print(
        f"{'workload':<24} {'provider':<12} {'model':<18} {'n':>6} {'tok_in':>9} {'cache':>7} issue"
    )
    for row in rows:
        print(
            f"{row['workload'][:23]:<24} {row['provider'][:11]:<12} "
            f"{row['model'][:17]:<18} {row['n_calls']:>6} "
            f"{row['tokens_in']:>9} {_fmt_stat_pct(row['cache_read_ratio']):>7} "
            f"{row['issue']}"
        )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    if args.prompt_file and args.prompt:
        raise ValueError("pass either PROMPT or --prompt-file, not both")
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt == "-":
        prompt = sys.stdin.read()
    else:
        prompt = args.prompt
    if not prompt:
        raise ValueError("prompt is required (pass PROMPT, --prompt-file, or '-')")

    cfg = _common.load_config(project=args.project)
    from somm.client import SommLLM

    llm = SommLLM(config=cfg)
    try:
        result = llm.generate(
            prompt,
            system=args.system,
            workload=args.workload,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            claude_cli_text_only=args.claude_cli_text_only,
            **({"allow_fallback": False} if args.claude_cli_text_only else {}),
        )
    finally:
        llm.close()

    payload = {
        "ok": result.outcome == Outcome.OK,
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "call_id": result.call_id,
        "outcome": result.outcome.value,
        "error_kind": result.error_kind,
        "error_message": _error_message(result.error_kind, result.error_detail),
        "error_detail": result.error_detail,
        "ttft_ms": result.ttft_ms,
        "cache_tokens_in": result.cache_tokens_in,
        "cache_tokens_out": result.cache_tokens_out,
        "citations": result.citations,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(result.text)
    if not payload["ok"] and not args.json:
        print(result.error_detail or result.error_kind or result.outcome.value, file=sys.stderr)
    return 0 if payload["ok"] else 1


def _error_message(error_kind: str | None, error_detail: str | None) -> str | None:
    """The human-readable reason for a failed call, for the ``--json`` envelope.

    ``error_detail`` is the ledger's bounded operator string
    (``Kind: message | http_status=… | provider=… | model=…``). Callers had
    to open the calls table to read it; this is the message part without
    the kind prefix, capped at 400 chars. None when the call succeeded.
    """
    if not error_detail:
        return None
    message = error_detail
    if error_kind and message.startswith(f"{error_kind}: "):
        message = message[len(error_kind) + 2 :]
    return message[:400]


def _bench_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file and args.prompt:
        raise ValueError("pass either PROMPT or --prompt-file, not both")
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt == "-":
        return sys.stdin.read()
    if not args.prompt:
        raise ValueError("prompt is required (pass PROMPT, --prompt-file, or '-')")
    return args.prompt


def _bench_tpot_ms(result) -> float | None:
    if result.ttft_ms is None or result.tokens_out <= 1:
        return None
    if result.latency_ms < result.ttft_ms:
        return None
    return (result.latency_ms - result.ttft_ms) / (result.tokens_out - 1)


def _bench_percentile(values: list[float | int | None], pct: int) -> float | None:
    ordered = sorted(float(v) for v in values if v is not None)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, (pct * len(ordered) + 99) // 100 - 1))
    return ordered[index]


def _bench_summary(rows: list[dict], *, wall_seconds: float) -> dict:
    ok_rows = [row for row in rows if row["outcome"] == Outcome.OK.value]
    total_latency_ms = sum(row["latency_ms"] for row in ok_rows)
    total_input_tokens = sum(row["tokens_in"] for row in ok_rows)
    total_output_tokens = sum(row["tokens_out"] for row in ok_rows)
    total_tokens = total_input_tokens + total_output_tokens
    total_cost = sum(row["cost_usd"] for row in rows)
    unpriced = sum(1 for row in rows if not row.get("cost_known", True))
    return {
        "calls": len(rows),
        "ok_calls": len(ok_rows),
        "failed_calls": len(rows) - len(ok_rows),
        "wall_seconds": wall_seconds,
        "latency_ms": {
            "p50": _bench_percentile([row["latency_ms"] for row in ok_rows], 50),
            "p95": _bench_percentile([row["latency_ms"] for row in ok_rows], 95),
            "p99": _bench_percentile([row["latency_ms"] for row in ok_rows], 99),
            "mean": (total_latency_ms / len(ok_rows)) if ok_rows else None,
        },
        "ttft_ms": {
            "p50": _bench_percentile([row["ttft_ms"] for row in ok_rows], 50),
            "p95": _bench_percentile([row["ttft_ms"] for row in ok_rows], 95),
            "p99": _bench_percentile([row["ttft_ms"] for row in ok_rows], 99),
        },
        "tpot_ms": {
            "mean": (
                sum(row["tpot_ms"] for row in ok_rows if row["tpot_ms"] is not None)
                / len([row for row in ok_rows if row["tpot_ms"] is not None])
            )
            if any(row["tpot_ms"] is not None for row in ok_rows)
            else None,
            "p95": _bench_percentile([row["tpot_ms"] for row in ok_rows], 95),
        },
        "throughput": {
            "requests_per_second": (len(ok_rows) / wall_seconds) if wall_seconds > 0 else None,
            "input_tokens_per_second": (total_input_tokens / wall_seconds)
            if wall_seconds > 0
            else None,
            "output_tokens_per_second": (total_output_tokens / wall_seconds)
            if wall_seconds > 0
            else None,
            "total_tokens_per_second": (total_tokens / wall_seconds) if wall_seconds > 0 else None,
        },
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_tokens,
        },
        "cost_usd": total_cost,
        "unpriced_calls": unpriced,
    }


def _cmd_bench(args: argparse.Namespace) -> int:
    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2
    cfg = _common.load_config(project=args.project)
    prompt = _bench_prompt(args)
    workload = args.workload or f"bench_{args.bench_cmd}"
    from somm.client import SommLLM

    llm = SommLLM(config=cfg)
    rows: list[dict] = []
    start: float | None = None
    try:
        for idx in range(args.warmup + args.iterations):
            if idx == args.warmup:
                start = time.monotonic()
            result = llm.generate(
                prompt,
                workload=workload,
                provider=args.provider,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                allow_fallback=args.allow_fallback and not args.no_fallback,
            )
            if idx < args.warmup:
                continue
            rows.append(
                {
                    "call_id": result.call_id,
                    "provider": result.provider,
                    "model": result.model,
                    "outcome": result.outcome.value,
                    "latency_ms": result.latency_ms,
                    "ttft_ms": result.ttft_ms,
                    "tpot_ms": _bench_tpot_ms(result),
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cost_usd": result.cost_usd,
                    "error_kind": result.error_kind,
                }
            )
    finally:
        llm.close()
    _mark_cost_known(getattr(llm, "repo", None), rows)
    wall_seconds = time.monotonic() - (start or time.monotonic())
    summary = _bench_summary(rows, wall_seconds=wall_seconds)
    payload = {
        "mode": args.bench_cmd,
        "project": cfg.project,
        "workload": workload,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "summary": summary,
        "runs": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    lat = summary["latency_ms"]
    ttft = summary["ttft_ms"]
    tpot = summary["tpot_ms"]
    thr = summary["throughput"]
    print(
        f"bench {args.bench_cmd}: {cfg.project}/{workload}  n={args.iterations} warmup={args.warmup}"
    )
    print(
        "latency_ms "
        f"p50={_fmt_stat_float(lat['p50'])} "
        f"p95={_fmt_stat_float(lat['p95'])} "
        f"p99={_fmt_stat_float(lat['p99'])} "
        f"ttft95={_fmt_stat_float(ttft['p95'])} "
        f"tpot_mean={_fmt_stat_float(tpot['mean'])}"
    )
    print(
        "throughput "
        f"req/s={_fmt_stat_float(thr['requests_per_second'])} "
        f"out_tok/s={_fmt_stat_float(thr['output_tokens_per_second'])} "
        f"total_tok/s={_fmt_stat_float(thr['total_tokens_per_second'])} "
        f"cost=${summary['cost_usd']:.4f}"
        + (f" (+{summary['unpriced_calls']} unpriced)" if summary["unpriced_calls"] else "")
    )
    return 0


def _stats_global(repo, since_days: int) -> list[dict]:
    """Cross-project roll-up from the global mirror. Same shape as
    stats_by_workload but with `project` column."""
    return repo.stats_global_by_workload(since_days=since_days)


def _cmd_tail(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
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
                cost_s = _fmt_cost(r)
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
    with repo._open() as conn:
        return search_calls(
            conn,
            project,
            since=since.isoformat(),
            workload=workload,
            limit=200,
            newest_first=False,
        )


def _cmd_compare(args: argparse.Namespace) -> int:
    """Run a prompt through N models side-by-side. Non-routed, explicit picks.

    Use: somm compare "Summarize X" --models minimax/MiniMax-M3,openai/gpt-6-luna
    """
    from somm import SommLLM

    specs = _parse_model_specs(args.models)
    if not specs:
        print("no --models supplied. example:", file=sys.stderr)
        print("  somm compare 'hi' --models minimax/MiniMax-M3,openai/gpt-6-luna", file=sys.stderr)
        return 2

    llm = SommLLM(project=args.project or "compare")
    try:
        results = compare_models(
            llm,
            specs,
            prompt=args.prompt,
            workload=args.workload,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    finally:
        llm.close()
    _mark_cost_known(getattr(llm, "repo", None), results)

    _print_comparison(results, truncate=args.truncate)
    return 0


_parse_model_specs = parse_model_specs


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
        cost_s = _fmt_cost(r)
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


def _cmd_frontier(args: argparse.Namespace) -> int:
    """Print the adequacy frontier for one workload.

    For each (provider, model) the workload has touched in the window:
    capability-failure rate, detractor rate, p50/p95 latency, mean cost
    per ok call, and whether each workload constraint is exceeded.
    """
    cfg = _common.load_config(project=args.project)
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
        "max_p95_ttft_ms": wl.max_p95_ttft_ms,
        "max_tpot_ms": wl.max_tpot_ms,
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
        f"{'p50ms':>7} {'p95ms':>7} {'ttft95':>7} {'tpot':>7} {'$/ok':>9} fitness"
    )
    for r in rows:
        cap_pct = 100.0 * r["capability_failure_rate"]
        det_pct = 100.0 * r["detractor_rate"]
        p50 = r["p50_latency_ms"] if r["p50_latency_ms"] is not None else "-"
        p95 = r["p95_latency_ms"] if r["p95_latency_ms"] is not None else "-"
        ttft95 = r["p95_ttft_ms"] if r["p95_ttft_ms"] is not None else "-"
        tpot = _fmt_stat_float(r["tpot_ms"])
        cost = r["mean_cost_per_ok_call"]
        cost_s = f"${cost:.5f}" if cost is not None else "-"
        flags = []
        f = r["fitness"]
        if f["exceeds_max_capability_failure_rate"]:
            flags.append("UNFIT(cap)")
        if f["exceeds_max_p95_latency_ms"]:
            flags.append("UNFIT(slow)")
        if f["exceeds_max_p95_ttft_ms"]:
            flags.append("UNFIT(ttft)")
        if f["exceeds_max_tpot_ms"]:
            flags.append("UNFIT(tpot)")
        if f["exceeds_max_cost_per_call_usd"]:
            flags.append("UNFIT($)")
        fitness_s = ",".join(flags) if flags else "ok"
        print(
            f"{r['provider'][:13]:<14} {r['model'][:27]:<28} {r['n_calls']:>5} "
            f"{cap_pct:>5.1f}% {det_pct:>5.1f}% {str(p50):>7} {str(p95):>7} "
            f"{str(ttft95):>7} {tpot:>7} {cost_s:>9} {fitness_s}"
        )
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    ps = sub.add_parser("status", help="show call roll-up for the current project")
    ps.add_argument("--project", default=None)
    ps.add_argument("--since", type=int, default=7, help="window in days (default 7)")
    ps.add_argument("--json", action="store_true", help="emit stable JSON")
    ps.add_argument(
        "--global",
        dest="global_view",
        action="store_true",
        help="read from ~/.somm/global.sqlite (requires SOMM_CROSS_PROJECT)",
    )
    ps.set_defaults(func=_cmd_status)

    pcache = sub.add_parser("cache-advice", help="find low prefix-cache reuse opportunities")
    pcache.add_argument("--project", default=None)
    pcache.add_argument("--since", type=int, default=7, help="window in days (default 7)")
    pcache.add_argument("--min-tokens-in", type=int, default=1_000)
    pcache.add_argument("--max-cache-read-ratio", type=float, default=0.20)
    pcache.add_argument("--json", action="store_true", help="emit stable JSON")
    pcache.set_defaults(func=_cmd_cache_advice)

    pg = sub.add_parser("generate", help="one-shot LLM call through somm")
    pg.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
    pg.add_argument("--prompt-file", default=None, help="read prompt text from a file")
    pg.add_argument("--project", default=None)
    pg.add_argument("--workload", default="generate")
    pg.add_argument("--system", default="")
    pg.add_argument("--provider", default=None)
    pg.add_argument("--model", default=None)
    pg.add_argument("--max-tokens", type=int, default=1024)
    pg.add_argument("--temperature", type=float, default=0.0)
    pg.add_argument("--json", action="store_true", help="emit a machine-readable result")
    pg.add_argument("--claude-cli-text-only", action="store_true",
                    help="disable Claude CLI tools, MCP config, and settings; requires --provider claude-cli")
    pg.set_defaults(func=_cmd_generate)

    pb = sub.add_parser("bench", help="run latency and throughput probes")
    pb_sub = pb.add_subparsers(dest="bench_cmd", required=True)
    for bench_cmd, default_iterations in (("latency", 5), ("throughput", 10)):
        pbench = pb_sub.add_parser(bench_cmd, help=f"run a {bench_cmd} benchmark")
        pbench.add_argument("prompt", nargs="?", help="prompt text, or '-' to read stdin")
        pbench.add_argument("--prompt-file", default=None, help="read prompt text from a file")
        pbench.add_argument("--project", default=None)
        pbench.add_argument("--workload", default=None)
        pbench.add_argument("--provider", default=None)
        pbench.add_argument("--model", default=None)
        pbench.add_argument("--iterations", type=int, default=default_iterations)
        pbench.add_argument("--warmup", type=int, default=0)
        pbench.add_argument("--max-tokens", type=int, default=256)
        pbench.add_argument("--temperature", type=float, default=0.0)
        pbench.add_argument(
            "--allow-fallback",
            action="store_true",
            help="let a failed --provider/--model pin be rescued by the router chain "
            "(off by default: a benchmark that silently changes model measures nothing)",
        )
        # Deprecated: stickiness is the default now. Kept so existing bench
        # scripts keep running instead of dying on an unknown flag.
        pbench.add_argument("--no-fallback", action="store_true", help=argparse.SUPPRESS)
        pbench.add_argument("--json", action="store_true", help="emit a machine-readable result")
        pbench.set_defaults(func=_cmd_bench)

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

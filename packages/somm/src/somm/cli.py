"""somm CLI entry point — grouped subcommands (D3e).

Commands:
  somm status       roll-up per (workload, provider, model)
  somm tail         live call stream
  somm compare      run a prompt through N models side-by-side
  somm doctor       health check (config, ollama, db, model_intel, workers, cooldowns)
  somm serve        run the web admin + HTTP API (requires somm-service)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta

from somm_core import VERSION, list_intel
from somm_core.config import load as load_config
from somm_core.repository import Repository

from somm.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# somm status


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
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

    # Worker heartbeats (from `jobs` table)
    with repo._open() as conn:
        jobs_rows = conn.execute(
            "SELECT job_name, last_started_at, last_success_at, "
            "       consecutive_failures, interval_seconds, due_at, locked_until "
            "FROM jobs ORDER BY job_name"
        ).fetchall()
    if not jobs_rows:
        print("workers: not yet started (start `somm serve` to seed + run)")
    else:
        print("workers:")
        for r in jobs_rows:
            name, started, success, failures, interval, due_at, locked_until = r
            status = "ok" if failures == 0 else f"WARN ({failures} failures)"
            last_ok = _age_since(success) if success else "never"
            print(f"  {name:<14} last_ok={last_ok:<20} interval={interval}s  {status}")

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

    pd = sub.add_parser("doctor", help="check config + ollama + db + intel + workers + cooldowns")
    pd.add_argument("--project", default=None)
    pd.set_defaults(func=_cmd_doctor)

    psr = sub.add_parser("serve", help="run the web admin + HTTP API (localhost:7878)")
    psr.add_argument("--project", default=None)
    psr.add_argument("--host", default="127.0.0.1")
    psr.add_argument("--port", type=int, default=7878)
    psr.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

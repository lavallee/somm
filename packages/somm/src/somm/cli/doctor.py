"""somm doctor."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core import VERSION, list_intel
from somm_core.aliases import needs_alias
from somm_core.cost import known_cost_sql
from somm_core.read import fleet_summary, north_star
from somm_core.repository import Repository

from somm.cli import _common
from somm.cli.boundary import _print_boundary_section
from somm.cli.fleet import _cmd_fleet_status
from somm.providers.ollama import OllamaProvider


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    global_view = bool(getattr(args, "global_view", False))
    db_path = cfg.global_db_path if global_view else cfg.db_path
    repo_exists = db_path.exists()
    print(f"somm v{VERSION}")
    if global_view:
        print("scope: global (every project that mirrors here)")
    else:
        print(f"project: {cfg.project}  mode: {cfg.mode}")
    print(f"db_path: {db_path}   exists: {repo_exists}")
    if repo_exists:
        mode = oct(db_path.stat().st_mode)[-3:]
        ok = mode in ("600",)
        print(f"db perms: {mode} {'(ok)' if ok else '(WARN — expect 600)'}")

    ok_overall = True

    from somm import connect

    for line in connect.status_lines(connect.found_harnesses()):
        print(f"agent {line}")

    # Ollama
    p = OllamaProvider(base_url=cfg.ollama_url, default_model=cfg.ollama_model)
    h = p.health()
    ok_ollama = h.available
    print(f"ollama:   {'ok' if ok_ollama else 'UNAVAILABLE'}  ({h.detail})")
    ok_overall = ok_overall and ok_ollama

    from somm import gpu_spend

    try:
        gpu = gpu_spend.gpu_spend(days=30)  # None (silent) when no gipper receipts directory exists
    except Exception:  # noqa: BLE001 - a doctor must not crash on a report
        gpu = None
    if gpu is not None:
        print(f"gipper:   {gpu['jobs']} rented-GPU job(s) in the last 30d, ${gpu['total_usd']:.2f}  (somm gpu spend)")

    if not repo_exists:
        print("db missing — skipping intel/workers/cooldowns checks")
        return 0 if ok_overall else 1

    repo = Repository(db_path)

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

    # Entries that have failed for so long they look withdrawn, not flaky.
    from somm.routing import dead_entries

    dead = dead_entries(repo)
    if dead:
        print(
            f"dead entries: {len(dead)} — no success in a week after 100+ consecutive "
            "failures; drop them from the roster (SOMM_OPENROUTER_ROSTER) or clear "
            "the cooldown once the provider lists them again"
        )
        for rec in dead:
            slot = f"{rec.provider}/{rec.model}" if rec.model else rec.provider
            last_ok = rec.last_ok_at.date().isoformat() if rec.last_ok_at else "never"
            print(f"  {slot:<42} failures={rec.consecutive_failures}   last ok {last_ok}")

    # Provider-exhaustion regression gate. Always computed and printed —
    # the July 2026 incident (SommProvidersExhausted at 90%+ of calls for
    # weeks, root-caused to a saturated minimax model default) had no cheap
    # trailing-window check that a cron/timer could have alerted on before
    # it was noticed by eyeballing spend. See docs/incidents/2026-07-provider-exhaustion.md.
    rate, exhausted, total = _exhausted_rate_24h(repo)
    if total == 0:
        print("exhausted_rate_24h: no calls in the last 24h")
    else:
        print(f"exhausted_rate_24h: {rate:.1%}  ({exhausted}/{total} calls)")
        if args.max_exhausted_rate is not None and rate > args.max_exhausted_rate:
            print(
                f"  WARN — exceeds --max-exhausted-rate {args.max_exhausted_rate:.1%}"
            )
            ok_overall = False

    # Call-site capture coverage. A call with no recorded site cannot be joined
    # back to the code that made it, which is what a static audit needs.
    sited, total_sited = _call_site_coverage_24h(repo)
    if total_sited == 0:
        print("call_site coverage: no calls in the last 24h")
    else:
        print(
            f"call_site coverage: {sited / total_sited:.1%}  "
            f"({sited}/{total_sited} calls in the last 24h)"
        )

    # Calls whose price somm did not know (recorded 0.0, source unknown:*).
    # Budgets count them as nothing, so this is spend the caps cannot see.
    unknown = _unknown_cost_by_provider(repo)
    if not unknown:
        print("unknown cost (30d): none")
    else:
        detail = ", ".join(f"{provider} {n}" for provider, n in unknown)
        print(f"unknown cost (30d): {sum(n for _, n in unknown)} calls  ({detail})")

    # Undeclared output contracts. Not a health failure — a workload runs fine
    # without a declared output schema. It is an evaluation gap: without one,
    # nothing downstream can decide whether a fixture could falsify the
    # workload's result, so held-out grading and placement assessment stay
    # guesses. Warn on the workloads with enough traffic to be worth declaring.
    undeclared = _workloads_without_output_schema(repo)
    if not undeclared:
        print("output schemas: every workload above the traffic floor declares one")
    else:
        print(
            f"output schemas: {len(undeclared)} workload(s) with "
            f">={_OUTPUT_SCHEMA_MIN_CALLS} calls declare none"
        )
        for project, name, calls in undeclared[:10]:
            print(f"  {f'{project}.{name}'[:46]:<48} {calls:>8} calls")
        if len(undeclared) > 10:
            print(f"  ... and {len(undeclared) - 10} more")

    _print_boundary_section(cfg, repo, global_view=global_view)

    _print_fleet_section(db_path)
    if global_view:
        # Read from each registered ledger, not the mirror: a project that does
        # not mirror (or whose mirror fell behind) is still in the fleet.
        try:
            _cmd_fleet_status(argparse.Namespace(json=False))
        except Exception as exc:  # noqa: BLE001 - a doctor must not crash on a report
            print(f"fleet: unavailable ({exc})")

    _print_north_star(db_path, global_view=global_view)

    return 0 if ok_overall else 1


def _print_north_star(db_path: Path, *, global_view: bool) -> None:
    """VISION.md's three numbers with their denominators, or why one cannot be
    measured. A mirror carries decisions and calls but never recommendations,
    so `--global` totals that one over the registered project ledgers."""
    try:
        ns = north_star(db_path)
        if global_view:
            from somm_core.registry import registered_ledgers

            keys = ("acted", "eligible", "pending", "applied")
            total = dict.fromkeys(keys, 0)
            # Count each file once: several project names can share a ledger.
            seen = {Path(db_path).resolve()}
            for _project, path in registered_ledgers():
                if Path(path).resolve() in seen or not Path(path).exists():
                    continue
                seen.add(Path(path).resolve())
                rec = north_star(Path(path))["recommendations_applied"]
                for k in keys:
                    total[k] += rec.get(k, 0)
            total["window_days"] = ns["recommendations_applied"]["window_days"]
            total["measurable"] = total["eligible"] > 0
            if not total["measurable"]:
                total["reason"] = "no project ledger has a recommendation closed or 14 days old"
            ns["recommendations_applied"] = total
    except Exception as exc:  # noqa: BLE001 - a doctor must not crash on a report
        print(f"north star: unavailable ({exc})")
        return

    def pct(n: int, d: int) -> str:
        return f"{n / d:.1%}" if d else "—"

    rec, route, recs = (
        ns["decisions_recalled"], ns["evidence_routed"], ns["recommendations_applied"]
    )
    print("north star:" + ("  (recommendations totalled over project ledgers)" if global_view else ""))
    if rec["measurable"]:
        print(
            f"  decisions recalled       {rec['recalled']} of {rec['decisions']} decisions"
            f"  ({rec['excluded']} excluded: workload has no calls here)"
        )
    else:
        print(f"  decisions recalled       not measurable yet: {rec['reason']}")
    if route["measurable"]:
        print(
            f"  evidence-routed share    {pct(route['evidence'], route['marked'])}  "
            f"({route['evidence']} of {route['marked']} production calls, "
            f"{route['window_days']}d; {route['unmarked']} unknown)"
        )
    else:
        print(f"  evidence-routed share    not measurable yet: {route['reason']}")
    if recs["measurable"]:
        print(
            f"  recommendations applied  {pct(recs['acted'], recs['eligible'])}  "
            f"({recs['acted']} of {recs['eligible']} applied or dismissed within "
            f"{recs['window_days']}d; {recs['applied']} applied ever; "
            f"{recs['pending']} younger and open)"
        )
    else:
        print(f"  recommendations applied  not measurable yet: {recs['reason']}")


def _print_loop_state(loop: dict, ledgers: list[dict], *, is_mirror: bool) -> None:
    """One line for this ledger, one per other registered ledger.

    A mirror never carries samples, evals, or recommendations, so an empty
    loop there is not dormancy — the project ledgers are where to look."""

    def line(state: dict) -> str:
        if state["shadow_configured_workloads"] and not state["eval_results"]:
            return (
                f"DORMANT — {state['shadow_configured_workloads']} workload(s) opted into "
                "online eval, nothing graded, no worker has run; `somm loop run`"
            )
        if not state["eval_results"] and not state["recommendations"]:
            return f"idle — no evals or recommendations ({state['decisions']} decisions)"
        last = state.get("last_worker_success") or "never"
        return (
            f"{state['eval_results']} evals, {state['recommendations']} recommendations "
            f"({state['recommendations_open']} open), last worker success {last}"
        )

    if is_mirror:
        print(
            "intelligence loop: not visible here — the mirror carries calls and workloads, "
            "not samples, evals, or recommendations; per project ledger:"
        )
    else:
        print(f"intelligence loop: {line(loop)}")
    for state in ledgers:
        label = state.get("project") or Path(state["db_path"]).parent.parent.name
        print(f"  {label[:30]:<32} {line(state)}")
    if is_mirror and not ledgers:
        print("  (no project ledgers registered in ~/.somm/registry.json)")


def _print_fleet_section(db_path: Path, *, since_days: int = 7) -> None:
    """What only a cross-project view can say: who writes with which client,
    whether the intelligence loop has ever run, and what pollutes the rollups.

    Printed for the project ledger too — a single project can be on an old
    client, and the loop can be idle there as well."""
    try:
        fleet = fleet_summary(db_path, since_days=since_days)
        fleet["is_mirror"] = Path(db_path).resolve() == _common.load_config().global_db_path.resolve()
    except Exception as exc:  # noqa: BLE001 - a doctor must not crash on a report
        print(f"fleet: unavailable ({exc})")
        return
    print(f"clients ({since_days}d):")
    if not fleet["clients"]:
        print("  no calls in the window")
    for row in fleet["clients"][:12]:
        stale = "" if row["client_version"] == VERSION else "  (older than this somm)"
        print(
            f"  {row['project'][:20]:<22} {row['client_version']:<14} "
            f"{row['calls']:>8} calls{stale}"
        )
    _print_loop_state(fleet["loop"], fleet["ledgers"], is_mirror=fleet.get("is_mirror", False))
    if fleet["sentinel_rows"]:
        print(
            f"sentinel rows ({since_days}d): {fleet['sentinel_rows']} calls with a blank or "
            "`*` model — exhaustion rows from pre-0.17 clients; they are excluded from "
            "per-model rollups"
        )
    for norm, spellings in fleet["provider_spellings"].items():
        print(f"provider spelling: {', '.join(spellings)} are the same provider ({norm})")
    # A route that is already canonical by rule (minimax/MiniMax-M3) has no
    # alias row and never will: `somm alias seed` writes only rows that
    # change something. Warn only about routes the seed can fix or that are
    # genuinely unresolved, not about every unaliased identity.
    unaliased = [(p, m) for p, m in fleet["unaliased"] if needs_alias(p, m)]
    if unaliased:
        shown = ", ".join(f"{p}/{m}" for p, m in unaliased[:6])
        more = len(unaliased) - 6
        print(
            f"model aliases: {len(unaliased)} route(s) with no canonical id "
            f"({shown}{', …' if more > 0 else ''}); run `somm alias seed`"
        )


def _exhausted_rate_24h(
    repo: Repository, now: datetime | None = None
) -> tuple[float, int, int]:
    """Trailing-24h SommProvidersExhausted rate: (rate, exhausted_count, total_count).

    Pure/testable: takes an explicit `now` so tests don't race real time.
    rate is 0.0 (not NaN) when total is 0 — callers that care about the
    no-data case should check `total` themselves (doctor prints a distinct
    message rather than a misleading 0.0%).
    """
    now = now or datetime.now(UTC)
    since = (now - timedelta(hours=24)).isoformat()
    with repo._open() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE ts >= ?", (since,)
        ).fetchone()[0]
        exhausted = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE ts >= ? AND error_kind = 'SommProvidersExhausted'",
            (since,),
        ).fetchone()[0]
    rate = (exhausted / total) if total else 0.0
    return rate, exhausted, total



# Traffic floor for the output-schema check. Below this a workload has not yet
# earned the upkeep of a declared contract; above it, the absence is a gap.
_OUTPUT_SCHEMA_MIN_CALLS = 100


def _call_site_coverage_24h(
    repo: Repository, now: datetime | None = None
) -> tuple[int, int]:
    """Trailing-24h call-site capture: (calls_with_a_site, total_calls).

    Rows predating schema 23 have no site and are honestly counted as missing,
    so coverage climbs as old rows age out of the window rather than jumping.
    """
    now = now or datetime.now(UTC)
    since = (now - timedelta(hours=24)).isoformat()
    with repo._open() as conn:
        total = conn.execute("SELECT COUNT(*) FROM calls WHERE ts >= ?", (since,)).fetchone()[0]
        sited = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE ts >= ? AND call_site IS NOT NULL",
            (since,),
        ).fetchone()[0]
    return sited, total


def _unknown_cost_by_provider(repo: Repository, days: int = 30) -> list[tuple[str, int]]:
    """(provider, calls) with an unknown cost in the trailing window, most first."""
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT provider, COUNT(*) FROM calls "
            f"WHERE ts >= ? AND NOT {known_cost_sql()} "
            "GROUP BY provider ORDER BY COUNT(*) DESC, provider",
            (since,),
        ).fetchall()
    return [(str(r[0]), int(r[1])) for r in rows]


def _workloads_without_output_schema(
    repo: Repository, *, min_calls: int = _OUTPUT_SCHEMA_MIN_CALLS
) -> list[tuple[str, str, int]]:
    """Workloads with real traffic and no declared output schema, busiest first.

    Credited to ``calls.project``: a workload id is name + schemas, so in a
    mirror two projects declaring the same workload share one workloads row,
    owned by whichever registered first."""
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT c.project, w.name, COUNT(c.id) AS n "
            "FROM workloads w JOIN calls c ON c.workload_id = w.id "
            "WHERE w.output_schema_json IS NULL "
            "OR TRIM(w.output_schema_json) IN ('', 'null', '{}') "
            "GROUP BY c.project, w.name HAVING n >= ? ORDER BY n DESC",
            (min_calls,),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


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


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pd = sub.add_parser("doctor", help="check config + ollama + db + intel + workers + cooldowns")
    pd.add_argument("--project", default=None)
    pd.add_argument(
        "--max-exhausted-rate",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "regression gate: exit nonzero when the trailing-24h "
            "SommProvidersExhausted rate exceeds FRACTION (e.g. 0.30). "
            "The rate is always printed; this flag only affects the exit "
            "code — suitable for a cron/timer health check."
        ),
    )
    pd.add_argument(
        "--global",
        dest="global_view",
        action="store_true",
        help="check ~/.somm/global.sqlite (every mirroring project) instead of this project",
    )
    pd.set_defaults(func=_cmd_doctor)

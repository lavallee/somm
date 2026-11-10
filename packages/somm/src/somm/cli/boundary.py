"""somm boundary: route tiers, ledger audit, per-workload boundary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from somm_core.boundary import TIER_NAMES
from somm_core.repository import Repository

from somm.cli import _common


def _parse_since_days(raw: str | None, default: float = 30.0) -> float:
    """'30d', '12h', '90m' or a bare number of days."""
    if raw is None or raw == "":
        return default
    text = str(raw).strip().lower()
    units = {"d": 1.0, "h": 1 / 24, "m": 1 / 1440, "w": 7.0}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _boundary_base_urls(cfg) -> dict[str, str]:
    """Endpoints to assume when classifying historic rows (no base URL stored)."""
    urls = {"ollama": cfg.ollama_url}
    if getattr(cfg, "openai_base_url", None):
        urls["openai"] = cfg.openai_base_url
    return urls


def _cmd_boundary_openrouter_evidence(args: argparse.Namespace, model: str) -> int:
    """Each upstream endpoint for one OpenRouter model, judged from live evidence."""
    from somm_core.boundary import load_policy

    from somm.providers.openrouter_evidence import load_evidence, qualifying_upstreams

    policy = load_policy()
    evidence = load_evidence() or {"zdr": [], "providers": [], "fetched_at": None}
    hq = {p.get("slug"): p.get("headquarters") for p in evidence["providers"]}
    listed = {e["tag"]: e for e in evidence["zdr"] if e.get("model_id") == model and e.get("tag")}
    qualifying = set(qualifying_upstreams(
        model, evidence["zdr"], evidence["providers"], policy.zdr_headquarters
    ))
    tags = set(listed)
    try:  # every endpoint, including those not on the ZDR list (public, no key)
        import httpx

        r = httpx.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=10.0)
        r.raise_for_status()
        tags |= {e["tag"] for e in r.json()["data"]["endpoints"] if e.get("tag")}
        all_listed = True
    except Exception:  # noqa: BLE001 — show what the ZDR list alone says
        all_listed = False
    rows = []
    for tag in sorted(tags):
        slug = tag.split("/")[0]
        entry = listed.get(tag)
        rows.append({
            "endpoint": tag,
            "headquarters": hq.get(slug),
            "on_zdr_list": entry is not None,
            "safety_retention": entry.get("safety_retention") if entry else None,
            "qualifies": tag in qualifying,
        })
    fetched = evidence["fetched_at"]
    fetched_text = (
        datetime.fromtimestamp(fetched, UTC).strftime("%Y-%m-%d %H:%M UTC") if fetched else None
    )
    if args.json:
        print(json.dumps({
            "route": f"openrouter/{model}", "evidence_fetched_at": fetched_text,
            "zdr_headquarters": list(policy.zdr_headquarters), "endpoints": rows,
        }, indent=2))
        return 0
    print(f"openrouter/{model}")
    print(f"evidence fetched: {fetched_text or 'never (no evidence: nothing is promoted)'}"
          f"   allowed headquarters: {', '.join(policy.zdr_headquarters) or '(none)'}")
    if not all_listed:
        print("(model endpoint list unreachable; showing ZDR-listed endpoints only)")
    print()
    print(f"{'endpoint':<32} {'hq':<6} {'zdr list':<9} {'safety ret.':<12} qualifies")

    def yn(v):
        return "-" if v is None else ("yes" if v else "no")

    for r in rows:
        print(f"{r['endpoint'][:31]:<32} {(r['headquarters'] or '?'):<6} "
              f"{yn(r['on_zdr_list']):<9} {yn(r['safety_retention']):<12} {yn(r['qualifies'])}")
    print()
    print("under a zdr or subscribed boundary: "
          + (f"admitted as zdr, pinned to {len(qualifying)} endpoint(s)" if qualifying
             else "refused (no qualifying endpoint)"))
    return 0


def _cmd_boundary_routes(args: argparse.Namespace) -> int:
    from somm_core.boundary import boundary_path, load_policy

    if (args.route or "").startswith("openrouter/"):
        return _cmd_boundary_openrouter_evidence(args, args.route.split("/", 1)[1])

    from somm.boundary import provider_routes
    from somm.client import build_default_providers
    from somm.providers.registry import build_pinned_only_providers

    cfg = _common.load_config(project=args.project)
    policy = load_policy()
    providers = build_default_providers(cfg, full=True)
    names = {p.name for p in providers}
    providers += list(build_pinned_only_providers(cfg, exclude=names).values())
    configured: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for p in providers:
        for route in provider_routes(p):
            seen.add((route.provider, route.model))
            configured.append({"configured": True, **route.as_dict()})
    if cfg.db_path.exists():
        repo = Repository(cfg.db_path)
        with repo._open() as conn:
            rows = conn.execute(
                "SELECT DISTINCT provider, model FROM calls WHERE project = ? "
                "AND provider != '' AND ts >= datetime('now', '-30 days') "
                "ORDER BY provider, model",
                (cfg.project,),
            ).fetchall()
        base_urls = _boundary_base_urls(cfg)
        for provider, model in rows:
            if (provider, model) in seen:
                continue
            route = policy.classify(provider, model, base_url=base_urls.get(provider))
            configured.append({"configured": False, **route.as_dict()})
    default, default_source = policy.project_default(cfg.project)
    payload = {
        "boundary_file": str(boundary_path()),
        "boundary_file_error": policy.owner_error,
        "project": cfg.project,
        "project_default": default.value if default else None,
        "project_default_source": default_source or None,
        "routes": configured,
        "rules": policy.routes(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"boundary file: {payload['boundary_file']}"
          + (f"  (UNREADABLE: {policy.owner_error})" if policy.owner_error else ""))
    print(
        f"project default: {payload['project_default'] or 'unvetted (fleet default)'}"
        + (f"  (from {default_source})" if default_source else "")
    )
    print()
    print(f"{'route':<52} {'tier':<11} {'origin':<8} basis")
    for r in configured:
        route = f"{r['provider']}/{r['model']}" if r["model"] else r["provider"]
        mark = "" if r["configured"] else " (seen in ledger)"
        print(f"{(route + mark)[:51]:<52} {r['tier']:<11} {r['origin']:<8} {r['basis'][:90]}")
    return 0


def _cmd_boundary_audit(args: argparse.Namespace) -> int:
    from somm_core.boundary_audit import audit

    cfg = _common.load_config(project=args.project)
    try:
        since_days = _parse_since_days(args.since)
    except ValueError:
        print(f"invalid --since {args.since!r}; use e.g. 30d, 12h", file=sys.stderr)
        return 2
    if not cfg.db_path.exists():
        print(f"no ledger at {cfg.db_path}", file=sys.stderr)
        return 2
    repo = Repository(cfg.db_path)
    with repo._open() as conn:
        rows = audit(
            conn, cfg.project, since_days=since_days, base_urls=_boundary_base_urls(cfg)
        )
    violations = [r for r in rows if r["violation"]]
    if args.json:
        print(
            json.dumps(
                {
                    "project": cfg.project,
                    "since_days": since_days,
                    "violations": len(violations),
                    "workloads": rows,
                },
                indent=2,
            )
        )
        return 1 if violations else 0
    print(f"Data-boundary audit — project {cfg.project!r}, last {args.since or '30d'}")
    print(
        f"{'workload':<28} {'declared':<11} {'effective':<11} {'worst served':<13} "
        f"{'calls':>7} {'refused':>8}"
    )
    for r in rows:
        flag = "  VIOLATION" if r["violation"] else ("  assumed" if r["assumed"] else "")
        print(
            f"{r['workload'][:27]:<28} {(r['declared_boundary'] or '-'):<11} "
            f"{r['effective_boundary']:<11} {(r['worst_tier'] or '-'):<13} "
            f"{r['calls']:>7} {r['refusals']:>8}{flag}"
        )
        if r["violation"] or args.verbose:
            for route in r["routes"]:
                note = "" if route["recorded"] else "  (classified now)"
                print(
                    f"    {route['provider']}/{route['model']}"[:60].ljust(62)
                    + f"{route['tier']:<11} {route['origin']:<8} {route['calls']:>7}{note}"
                )
    print()
    if any(r["assumed"] for r in rows):
        print(
            "assumed: the worst tier came from a seat whose real tier depends on an account "
            "setting somm cannot see (e.g. a training toggle). Declare it in boundary.toml."
        )
    if violations:
        print(
            f"{len(violations)} workload(s) were served beyond their boundary in the window "
            "(rows written before enforcement, or a boundary set after the fact)."
        )
    else:
        print("No workload was served beyond its effective boundary in the window.")
    print(
        "Tighten from evidence: `somm boundary set <workload> <tier>` once the worst "
        "served tier is within it."
    )
    return 1 if violations else 0


def _cmd_boundary_set(args: argparse.Namespace) -> int:
    from somm_core.boundary import resolve_boundary

    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = repo.workload_by_name(args.workload, cfg.project)
    if wl is None:
        print(
            f"No workload {args.workload!r} registered for project {cfg.project!r}.",
            file=sys.stderr,
        )
        return 2
    tier = None if args.tier in ("none", "clear", "derive") else args.tier
    try:
        repo.set_workload_data_boundary(wl.id, tier, created_by="somm boundary set")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    wl = repo.workload_by_name(args.workload, cfg.project)
    effective = resolve_boundary(wl, project=cfg.project)
    print(
        f"workload {wl.name!r}: data_boundary={wl.data_boundary or '(derive)'} "
        f"→ effective {effective.tier.value} (from {effective.source})"
    )
    return 0


def _print_boundary_section(cfg, repo: Repository, *, global_view: bool) -> None:
    from somm_core.boundary import audit, boundary_path, load_policy

    policy = load_policy()
    path = boundary_path()
    owner = len(policy.owner_rules)
    if policy.owner_error:
        state = f"UNREADABLE ({policy.owner_error}) — every project default is local"
    elif path.exists():
        state = f"{owner} owner rule(s)"
    else:
        state = "absent (shipped defaults only)"
    print(f"boundary: {path}  {state}")
    default, source = policy.project_default(cfg.project)
    print(
        f"  project default: {default.value if default else 'unvetted (fleet default)'}"
        + (f" (from {source})" if source else "")
    )
    if global_view:
        return
    try:
        with repo._open() as conn:
            rows = audit(conn, cfg.project, since_days=7, base_urls=_boundary_base_urls(cfg))
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        print(f"  audit: unavailable ({type(exc).__name__}: {exc})")
        return
    bounded = [r for r in rows if r["boundary_source"] != "fleet default"]
    violations = [r for r in rows if r["violation"]]
    refusals = sum(r["refusals"] for r in rows)
    print(
        f"  workloads with a boundary: {len(bounded)}/{len(rows)}   "
        f"refusals (7d): {refusals}   violations (7d): {len(violations)}"
    )
    for r in violations[:5]:
        print(
            f"    {r['workload'][:30]:<32} boundary {r['effective_boundary']}, "
            f"served {r['worst_tier']}"
        )
    if violations:
        print("  see `somm boundary audit --since 7d`")


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pbd = sub.add_parser(
        "boundary", help="data boundary: route tiers, ledger audit, per-workload boundary"
    )
    _common.add_project(pbd)
    pbd_sub = pbd.add_subparsers(dest="boundary_cmd", required=True)
    pbr = pbd_sub.add_parser("routes", help="every configured route with its tier and basis")
    pbr.add_argument(
        "route", nargs="?", default=None,
        help="openrouter/<model-id>: each upstream endpoint, judged from live ZDR evidence",
    )
    _common.add_project(pbr)
    pbr.add_argument("--json", action="store_true")
    pbr.set_defaults(func=_cmd_boundary_routes)
    pba = pbd_sub.add_parser(
        "audit", help="per workload: declared boundary vs the worst tier that served it"
    )
    _common.add_project(pba)
    pba.add_argument("--since", default="30d", help="window, e.g. 30d, 12h (default 30d)")
    pba.add_argument("--json", action="store_true")
    pba.add_argument("-v", "--verbose", action="store_true", help="list routes per workload")
    pba.set_defaults(func=_cmd_boundary_audit)
    pbs = pbd_sub.add_parser("set", help="set a workload's data boundary")
    pbs.add_argument("workload")
    pbs.add_argument("tier", choices=[*TIER_NAMES, "none"], help="tier, or none to derive")
    _common.add_project(pbs)
    pbs.set_defaults(func=_cmd_boundary_set)

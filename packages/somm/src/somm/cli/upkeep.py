"""somm backfill-costs, plans, drain-spool."""

from __future__ import annotations

import argparse
import json

from somm.cli import _common


def _cmd_backfill_costs(args: argparse.Namespace) -> int:
    from somm_core.pricing import backfill_costs, sync_bundled_pricing
    from somm_core.repository import Repository

    cfg = _common.load_config(project=args.project)
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

_CATALOG_STALE_DAYS = 90


def _cmd_plans(args: argparse.Namespace) -> int:
    from somm_core.plans import (
        learn_observed_limits,
        limit_statuses,
        load_catalog,
        load_plans,
        plans_path,
    )
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
            lims = (
                "; ".join(f"{lim.quota:g} {lim.unit}/{lim.window}" for lim in e.limits)
                or "no limits recorded"
            )
            print(f"  {key:<28} {e.display}")
            print(f"    {lims}  [{age_s}]")
            if e.notes:
                print(f"    note: {e.notes}")
            if e.source:
                print(f"    source: {e.source}")
        return 0

    cfg = _common.load_config(project=args.project)
    try:
        plans = load_plans(catalog=catalog)
    except Exception as exc:
        print(f"plans.toml is invalid: {exc}")
        return 1
    cfg = _common.load_config(project=args.project)
    dbs = (
        [cfg.db_path]
        if args.project_only
        else fleet_db_paths(include=cfg.db_path if cfg.db_path.exists() else None)
    )

    if args.learn:
        updates = learn_observed_limits(
            dbs,
            plans,
            dry_run=args.dry_run,
            min_events=args.min_events,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "path": str(plans_path()),
                        "dry_run": args.dry_run,
                        "updates": [
                            {
                                "provider": u.provider,
                                "window": u.window,
                                "unit": u.unit,
                                "old_quota": u.old_quota,
                                "new_quota": u.new_quota,
                                "n_events": u.n_events,
                                "last_event": u.last_event,
                                "action": u.action,
                            }
                            for u in updates
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        if not updates:
            print("no observed quota ceilings to learn")
        else:
            verb = "would update" if args.dry_run else "updated"
            changed = [u for u in updates if u.action in {"added", "updated"}]
            print(f"{verb} {len(changed)} learned plan limit(s) in {plans_path()}")
            for u in updates:
                old = "-" if u.old_quota is None else f"{u.old_quota:g}"
                print(
                    f"  {u.action:<9} {u.provider:<12} {u.window:<5} {u.unit:<12} "
                    f"{old} -> {u.new_quota:g} ({u.n_events} event(s))"
                )
        if not args.dry_run:
            try:
                plans = load_plans(catalog=catalog)
            except Exception as exc:
                print(f"plans.toml is invalid after learning: {exc}")
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
    statuses = limit_statuses(dbs, plans)
    scope = "this project only" if args.project_only else f"fleet ({len(dbs)} project DBs)"

    if args.json:
        out = []
        for st in statuses:
            out.append(
                {
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
                }
            )
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

    if "minimax" in plans and plans["minimax"].mode == "metered":
        import os

        from somm.providers.minimax import plan_remains

        key = os.environ.get("MINIMAX_API_KEY")
        live = plan_remains(key) if key else None
        if live:
            print(
                f"\nvendor-reported: minimax {live['interval_used_pct']}% of this 5h window, "
                f"{live['weekly_used_pct']}% of this week"
            )

    metered_no_limits = [p for p, pl in plans.items() if pl.mode == "metered" and not pl.limits]
    if metered_no_limits:
        print(
            f"\nmetered, no limits declared (labelled only): {', '.join(sorted(metered_no_limits))}"
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

    cfg = _common.load_config(project=args.project)
    if not cfg.db_path.exists():
        print("no telemetry database found")
        return 1
    n = drain_spool(Repository(cfg.db_path), cfg.spool_dir)
    print(f"drained {n} spooled call(s) into {cfg.db_path}")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pbf = sub.add_parser(
        "backfill-costs",
        help="recompute cost_usd for $0 calls that now have pricing intel",
    )
    _common.add_project(pbf)
    pbf.add_argument("--since", type=int, default=None, help="only calls from the last N days")
    pbf.add_argument("--dry-run", action="store_true", help="report without writing")
    pbf.set_defaults(func=_cmd_backfill_costs)

    ppl = sub.add_parser(
        "plans",
        help="metered-plan quota usage + pacing (PAYG vs metered, fleet-wide)",
    )
    _common.add_project(ppl)
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
    ppl.add_argument(
        "--learn",
        action="store_true",
        help="write observed quota ceilings from recent 429s into plans.toml",
    )
    ppl.add_argument("--dry-run", action="store_true", help="show learned limits without writing")
    ppl.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="minimum quota-error events required to learn a ceiling",
    )
    ppl.set_defaults(func=_cmd_plans)

    pds = sub.add_parser(
        "drain-spool",
        help="replay spooled JSONL telemetry (written during DB outages) into the DB",
    )
    _common.add_project(pds)
    pds.set_defaults(func=_cmd_drain_spool)

"""somm spend, somm gpu spend and somm climb."""

from __future__ import annotations

import argparse
import json
import sys

from somm_core.read import spend_today

from somm import climb
from somm.cli import _common


def _cmd_spend(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
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

def _cmd_gpu_spend(args: argparse.Namespace) -> int:
    from somm import gpu_spend

    cfg = _common.load_config(project=args.project)
    r = gpu_spend.gpu_spend(args.path, args.since, cfg.db_path, cfg.project)
    if r is None:
        print(f"no gipper receipts directory at {gpu_spend.receipts_dir(args.path)}", file=sys.stderr)
        return 1
    print(json.dumps(r, indent=2) if args.json else gpu_spend.format_report(r))
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pspend = sub.add_parser("spend", help="today's LLM spend vs daily budget cap per workload")
    _common.add_project(pspend)
    pspend.add_argument(
        "--json",
        dest="json",
        action="store_true",
        default=False,
        help="emit a JSON array instead of the aligned table",
    )
    pspend.set_defaults(func=_cmd_spend)

    pclimb = sub.add_parser(
        "climb", help="what a workload answers before any model (reads chipsocket receipts)"
    )
    climb.add_arguments(pclimb)
    pclimb.set_defaults(func=lambda a: climb.cmd(a, _common.load_config(project=a.project)))

    pgpu = sub.add_parser("gpu", help="rented-GPU spend read from gipper receipts (read-only)")
    _common.add_project(pgpu)
    pgpu_sub = pgpu.add_subparsers(dest="gpu_cmd", required=True)
    pgs = pgpu_sub.add_parser("spend", help="GPU spend by day and workload; failed jobs; ledger match")
    _common.add_project(pgs)
    pgs.add_argument("--since", type=int, default=30, help="window in days (default 30)")
    pgs.add_argument("--path", default=None, help="receipts directory (default: $SOMM_GIPPER_RECEIPTS, "
                     "else $GIPPER_DATA/receipts, else ~/.local/share/gipper/receipts)")
    pgs.add_argument("--json", action="store_true", default=False)
    pgs.set_defaults(func=_cmd_gpu_spend)

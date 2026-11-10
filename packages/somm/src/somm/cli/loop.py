"""somm loop and somm prune."""

from __future__ import annotations

import argparse
import json
import sys

from somm_core.repository import Repository

from somm.cli import _common


def _cmd_loop(args: argparse.Namespace) -> int:
    from somm import loop

    cfg = _common.load_config(project=args.project)
    cmd = args.loop_cmd
    if cmd == "run":
        only = tuple(args.only) if args.only else None
        if args.all_projects:
            reports = loop.run_fleet(force=args.force, only=only)
        else:
            reports = [loop.run_once(cfg, force=args.force, only=only)]
        if args.json:
            print(json.dumps([r.as_dict() for r in reports], indent=2, default=str))
        else:
            for r in reports:
                print(f"{r.project}  ({r.db_path})")
                for name, summary in r.ran.items():
                    brief = {
                        k: v for k, v in summary.items()
                        if k not in ("errors", "warnings", "shadow_broken")
                    }
                    print(f"  ran      {name:<12} {json.dumps(brief, default=str)[:140]}")
                    for w in summary.get("warnings") or []:
                        print(f"           warning: {w}")
                    for e in summary.get("errors") or []:
                        print(f"           error: {e}")
                for name, why in r.skipped.items():
                    print(f"  skipped  {name:<12} {why}")
                for name, err in r.errors.items():
                    print(f"  FAILED   {name:<12} {err}")
        return 1 if any(r.errors for r in reports) else 0
    if cmd == "install":
        try:
            written = loop.install(cfg, every=args.every, enable=not args.no_enable)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for path in written:
            print(f"wrote {path}")
        _, timer = loop.unit_names(cfg.project)
        if args.no_enable:
            print(f"enable with: systemctl --user daemon-reload && systemctl --user enable --now {timer}")
        else:
            print(f"enabled {timer} (every {args.every})")
        return 0
    if cmd == "uninstall":
        try:
            loop.uninstall(cfg)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"disabled {loop.unit_names(cfg.project)[1]}")
        return 0
    info = loop.status(cfg)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"project: {info['project']}")
    print(f"timer:   {info['timer']}  state: {info['timer_state'] or 'not installed'}")
    for name, when in info["heartbeats"].items():
        print(f"  {name:<12} last success: {when or 'never'}")
    return 0

def _cmd_prune(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    db_path = cfg.global_db_path if getattr(args, "global_view", False) else cfg.db_path
    if not db_path.exists():
        print(f"no ledger at {db_path}")
        return 0
    repo = Repository(db_path)
    try:
        counts = repo.prune_calls(older_than_days=args.older_than, dry_run=args.dry_run)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} rows older than {args.older_than} days from {db_path}:")
    for table, n in counts.items():
        print(f"  {table:<14} {n:>10}")
    if args.vacuum and not args.dry_run:
        before = db_path.stat().st_size
        repo.vacuum()
        after = db_path.stat().st_size
        print(f"vacuum: {before / 1e6:.1f} MB → {after / 1e6:.1f} MB")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    plp = sub.add_parser(
        "loop",
        help="run or schedule the intelligence loop (intel refresh, shadow grading, recommendations)",
    )
    _common.add_project(plp)
    plp_sub = plp.add_subparsers(dest="loop_cmd")
    plr = plp_sub.add_parser("run", help="one pass: run the workers that are due, then exit")
    # `--project` is accepted on either side of the subcommand (`somm loop
    # status --project X` reads like every other command).
    _common.add_project(plr)
    plr.add_argument("--force", action="store_true", help="run every worker regardless of staleness")
    plr.add_argument(
        "--only", action="append", choices=["model_intel", "shadow_eval", "agent"],
        help="restrict to one worker (repeatable)",
    )
    plr.add_argument("--all-projects", action="store_true",
                     help="every registered project ledger, not just this one")
    plr.add_argument("--json", action="store_true")
    pli = plp_sub.add_parser("install", help="write a systemd user timer that runs `somm loop run`")
    _common.add_project(pli)
    pli.add_argument("--every", default="15min", help="timer cadence (systemd span, default 15min)")
    pli.add_argument("--no-enable", action="store_true", help="write units without enabling")
    plu = plp_sub.add_parser("uninstall", help="disable the timer for this project")
    _common.add_project(plu)
    pls = plp_sub.add_parser("status", help="timer state and worker heartbeats")
    _common.add_project(pls)
    pls.add_argument("--json", action="store_true")
    plp.set_defaults(func=_cmd_loop, loop_cmd="status", json=False)

    ppr = sub.add_parser(
        "prune",
        help="delete calls (and their samples/updates/evals) older than a retention window",
    )
    _common.add_project(ppr)
    ppr.add_argument(
        "--older-than", type=int, required=True, metavar="DAYS", help="retention window in days"
    )
    ppr.add_argument("--dry-run", action="store_true", help="report counts without deleting")
    ppr.add_argument("--vacuum", action="store_true", help="reclaim file space afterwards")
    ppr.add_argument("--global", dest="global_view", action="store_true",
                     help="prune ~/.somm/global.sqlite instead of this project's ledger")
    ppr.set_defaults(func=_cmd_prune)

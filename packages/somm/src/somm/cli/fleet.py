"""somm ledgers, fleet, mirror."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from somm.cli import _common


def _ledgers_report(scan=()):
    from somm import ledgers

    return ledgers.ledgers_report(mirror=_common.load_config().global_db_path, scan=scan)


def _cmd_ledgers(args: argparse.Namespace) -> int:
    """Read-only: every registered ledger, the mirror, and (--scan) strays."""
    from somm import ledgers

    report = _ledgers_report(scan=[Path(d) for d in (args.scan or [])])
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print("\n".join(ledgers.render_ledgers(report)))
    return 0


def _cmd_ledgers_forget(args: argparse.Namespace) -> int:
    from somm_core.registry import forget_project

    try:
        removed = forget_project(
            args.name, path=Path(args.path) if args.path else None, dry_run=args.dry_run
        )
    except KeyError as exc:
        print(str(exc.args[0]), file=sys.stderr)
        return 2
    verb = "would forget" if args.dry_run else "forgot"
    for raw in removed:
        print(f"{verb} {args.name} -> {raw}")
    print("no file was touched")
    return 0


def _cmd_ledgers_adopt(args: argparse.Namespace) -> int:
    from somm_core.registry import adopt_ledger

    from somm import ledgers

    path = Path(args.path).expanduser().resolve()
    facts = ledgers.ledger_facts(path)
    if not facts["exists"] or facts.get("error") or facts.get("rows") is None:
        detail = facts.get("error", "")
        print(f"not a readable somm ledger: {path} {detail}".rstrip(), file=sys.stderr)
        return 2
    names = [args.name] if args.name else sorted(facts.get("projects", {}))
    if not names:
        print(f"{path} holds no calls to name a project from; pass --project NAME",
              file=sys.stderr)
        return 2
    for name in names:
        if adopt_ledger(name, path, dry_run=args.dry_run):
            print(f"{'would register' if args.dry_run else 'registered'} {name} -> {path}")
        else:
            print(f"already registered: {name} -> {path}")
    return 0


def _cmd_fleet_status(args: argparse.Namespace) -> int:
    """Read-only; the same lines `somm doctor --global` prints under `fleet`."""
    from somm import ledgers

    report = _ledgers_report()
    rows = ledgers.fleet_status(report)
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, default=str))
        return 0
    print("\n".join(ledgers.render_fleet_status(
        rows, window_days=report["window_days"], installed=report["installed_version"],
        schema=report["installed_schema"],
    )))
    return 0


def _cmd_mirror_sync(args: argparse.Namespace) -> int:
    from somm_core.registry import registry_entries

    from somm import ledgers

    mirror = _common.load_config().global_db_path
    entries = registry_entries()
    if args.project and args.project not in entries:
        print(f"project {args.project!r} is not registered; see `somm ledgers`", file=sys.stderr)
        return 2
    targets: dict[str, None] = {}
    for name, paths in entries.items():
        if args.project and name != args.project:
            continue
        for raw in sorted(paths):
            if Path(raw).exists() and Path(raw).resolve() != mirror.resolve():
                targets[str(Path(raw).resolve())] = None
    scope = f"{args.project}'s rows" if args.project else "every row"
    print(f"{'dry run: ' if args.dry_run else ''}copy {scope} the mirror lacks from "
          f"{len(targets)} ledger(s) into {mirror}")
    total = 0
    for raw in targets:
        result = ledgers.mirror_sync(
            Path(raw), mirror, project=args.project, batch=args.batch, dry_run=args.dry_run
        )
        n = result["missing"] if args.dry_run else result["copied"]
        total += n
        print(f"  {n:>9} {'missing' if args.dry_run else 'copied'}  {raw}")
    print(f"{'would copy' if args.dry_run else 'copied'} {total} call row(s)")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pled = sub.add_parser(
        "ledgers",
        help="every ledger on this machine: who writes it, its size, and the mirror gap",
    )
    pled.add_argument("--scan", action="append", metavar="DIR",
                      help="also list calls.sqlite files under DIR that are not registered")
    pled.add_argument("--json", action="store_true")
    pled.set_defaults(func=_cmd_ledgers)
    pled_sub = pled.add_subparsers(dest="ledgers_cmd")
    pledf = pled_sub.add_parser("forget", help="remove a registration (never deletes a file)")
    pledf.add_argument("name", metavar="PROJECT")
    pledf.add_argument("--path", default=None, help="forget only this one of its paths")
    pledf.add_argument("--dry-run", action="store_true")
    pledf.set_defaults(func=_cmd_ledgers_forget)
    pleda = pled_sub.add_parser("adopt", help="register an existing ledger")
    pleda.add_argument("path", metavar="PATH")
    pleda.add_argument("--project", dest="name", default=None,
                       help="register under this name (default: every project the file holds)")
    pleda.add_argument("--dry-run", action="store_true")
    pleda.set_defaults(func=_cmd_ledgers_adopt)

    pfl = sub.add_parser("fleet", help="per-project fleet health across registered ledgers")
    pfl.set_defaults(func=_cmd_fleet_status, json=False)
    pfl_sub = pfl.add_subparsers(dest="fleet_cmd")
    pfls = pfl_sub.add_parser(
        "status", help="versions, schema skew, loop, unknown cost (as in doctor --global)"
    )
    pfls.add_argument("--json", action="store_true")
    pfls.set_defaults(func=_cmd_fleet_status)

    pmir = sub.add_parser("mirror", help="the cross-project mirror (~/.somm/global.sqlite)")
    pmir_sub = pmir.add_subparsers(dest="mirror_cmd", required=True)
    pmirs = pmir_sub.add_parser(
        "sync", help="copy call rows a registered ledger has and the mirror lacks"
    )
    pmirs.add_argument("--project", default=None, help="only this project's ledgers and rows")
    pmirs.add_argument("--dry-run", action="store_true", help="count, copy nothing")
    pmirs.add_argument("--batch", type=int, default=5000, help="rows per transaction")
    pmirs.set_defaults(func=_cmd_mirror_sync)

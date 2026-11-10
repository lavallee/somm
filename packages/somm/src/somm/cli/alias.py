"""somm alias."""

from __future__ import annotations

import argparse
import json

from somm_core.repository import Repository

from somm.cli import _common


def _cmd_alias(args: argparse.Namespace) -> int:
    from somm_core.aliases import seed_model_aliases

    cfg = _common.load_config(project=args.project)
    db_path = cfg.global_db_path if getattr(args, "global_view", False) else cfg.db_path
    if not db_path.exists():
        print(f"no ledger at {db_path}")
        # seed/set are writes that did not happen; a read of no aliases is fine.
        return 1 if args.alias_cmd in ("seed", "set") else 0
    repo = Repository(db_path)
    if args.alias_cmd == "seed":
        n = seed_model_aliases(repo)
        print(f"wrote {n} rule-derived alias row(s) to {db_path}")
        return 0
    if args.alias_cmd == "set":
        repo.set_model_alias(args.canonical, args.provider, args.model, source="manual")
        print(f"{args.provider}/{args.model} → {args.canonical}")
        return 0
    rows = repo.model_aliases()
    if getattr(args, "json", False):
        print(json.dumps([
            {"provider": r.provider, "model": r.model, "canonical_id": r.canonical_id,
             "source": r.source} for r in rows
        ], indent=2))
        return 0
    if not rows:
        print("no aliases; run `somm alias seed`")
        return 0
    print(f"{'provider':<12} {'model':<42} {'canonical':<42} source")
    for r in rows:
        print(f"{r.provider[:11]:<12} {r.model[:41]:<42} {r.canonical_id[:41]:<42} {r.source}")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pal = sub.add_parser("alias",help="canonical model identities (list, seed, set)")
    _common.add_project(pal)
    pal.add_argument("--global", dest="global_view", action="store_true",
                     help="operate on ~/.somm/global.sqlite")
    pal_sub = pal.add_subparsers(dest="alias_cmd")
    pal_list = pal_sub.add_parser("list", help="show alias rows")
    pal_list.add_argument("--json", action="store_true")
    pal_seed = pal_sub.add_parser("seed", help="derive aliases by rule from routes seen in the ledger")
    pal_set = pal_sub.add_parser("set", help="map one provider/model route to a canonical id")
    pal_set.add_argument("provider")
    pal_set.add_argument("model")
    pal_set.add_argument("canonical")
    for sp in (pal_list, pal_seed, pal_set):
        _common.add_project(sp)
        sp.add_argument("--global", dest="global_view", action="store_true",
                        default=argparse.SUPPRESS, help="operate on ~/.somm/global.sqlite")
    pal.set_defaults(func=_cmd_alias, alias_cmd="list")

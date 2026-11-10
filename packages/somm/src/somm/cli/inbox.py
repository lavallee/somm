"""somm inbox: list, show, apply and dismiss recommendations."""

from __future__ import annotations

import argparse
import json
import sys

from somm_core.repository import Repository

from somm.cli import _common


def _decision_mirror_repo(cfg):
    if not getattr(cfg, "cross_project_enabled", False):
        return None
    try:
        return Repository(cfg.global_db_path)
    except Exception:
        return None


def _cmd_inbox_list(args: argparse.Namespace) -> int:
    from somm.recommendations import list_recommendations

    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    recs = list_recommendations(
        repo,
        project=cfg.project,
        workload=args.workload,
        open_only=not args.all,
    )
    if args.json:
        print(json.dumps([rec.as_dict() for rec in recs], indent=2))
        return 0
    if not recs:
        print("no recommendations")
        return 0
    print(
        f"{'id':>5} {'created':<10} {'workload':<24} {'action':<18} "
        f"{'conf':>6} {'state':<9} impact"
    )
    for rec in recs:
        confidence = "-" if rec.confidence is None else f"{rec.confidence:.2f}"
        print(
            f"{rec.id:>5} {str(rec.created_at)[:10]:<10} {rec.workload[:23]:<24} "
            f"{rec.action:<18} {confidence:>6} {_rec_state(rec):<9} {rec.expected_impact}"
        )
    return 0


def _rec_state(rec) -> str:
    if rec.applied_at:
        return "applied"
    if rec.dismissed_at:
        return "expired" if rec.evidence.get("closed_as") == "expired" else "dismissed"
    return "open"


def _cmd_inbox_show(args: argparse.Namespace) -> int:
    from somm.recommendations import get_recommendation

    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rec = get_recommendation(repo, args.recommendation_id)
    if rec is None:
        print(f"recommendation {args.recommendation_id} not found", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rec.as_dict(), indent=2))
        return 0
    confidence = "-" if rec.confidence is None else f"{rec.confidence:.2f}"
    print(f"recommendation {rec.id}  {rec.action}  {_rec_state(rec)}")
    print(f"  workload:   {rec.workload}")
    print(f"  created:    {rec.created_at}")
    if rec.applied_at:
        print(f"  applied:    {rec.applied_at}")
    if rec.dismissed_at:
        print(f"  {_rec_state(rec) + ':':<11} {rec.dismissed_at}")
    elif not rec.applied_at and rec.evidence.get("expires_at"):
        print(f"  expires:    {rec.evidence['expires_at']}")
    print(f"  confidence: {confidence}")
    print(f"  impact:     {rec.expected_impact}")
    print("  evidence:")
    for line in json.dumps(rec.evidence, indent=2, sort_keys=True).splitlines():
        print(f"    {line}")
    return 0


def _cmd_inbox_apply(args: argparse.Namespace) -> int:
    from somm.recommendations import apply_recommendation

    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    try:
        result = apply_recommendation(
            repo,
            args.recommendation_id,
            actor="somm inbox",
            mirror_repo=_decision_mirror_repo(cfg),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(
            f"applied recommendation {result.recommendation.id} "
            f"for {result.recommendation.workload}"
        )
        print(f"decision_id: {result.decision.id}")
        if result.revision is not None:
            print(f"workload_revision: {result.revision}")
    return 0


def _cmd_inbox_dismiss(args: argparse.Namespace) -> int:
    from somm.recommendations import dismiss_recommendation

    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    dismissed = []
    rc = 0
    for rec_id in args.recommendation_ids:
        try:
            dismissed.append(dismiss_recommendation(repo, rec_id))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            rc = 2
    if args.json:
        print(json.dumps([rec.as_dict() for rec in dismissed], indent=2))
    else:
        for rec in dismissed:
            print(f"dismissed recommendation {rec.id} for {rec.workload}")
    return rc


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pinbox = sub.add_parser("inbox", help="list, show, apply, and dismiss recommendations")
    # Bare `somm inbox [--project X]` lists: agents and people both type it
    # that way, and it used to fail with "invalid choice: 'X'".
    _common.add_project(pinbox)
    pinbox.add_argument("--workload", default=None)
    pinbox.add_argument("--all", action="store_true")
    pinbox.add_argument("--json", action="store_true")
    pinbox.set_defaults(func=_cmd_inbox_list)
    pinbox_sub = pinbox.add_subparsers(dest="inbox_cmd", required=False)
    pinbox_list = pinbox_sub.add_parser("list", help="list recommendations")
    _common.add_project(pinbox_list)
    pinbox_list.add_argument("--workload", default=argparse.SUPPRESS)
    pinbox_list.add_argument("--all", action="store_true", default=argparse.SUPPRESS)
    pinbox_list.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    pinbox_list.set_defaults(func=_cmd_inbox_list)

    pinbox_apply = pinbox_sub.add_parser("apply", help="apply a recommendation")
    pinbox_apply.add_argument("recommendation_id", type=int)
    _common.add_project(pinbox_apply)
    pinbox_apply.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    pinbox_apply.set_defaults(func=_cmd_inbox_apply)

    pinbox_dismiss = pinbox_sub.add_parser("dismiss", help="dismiss one or more recommendations")
    pinbox_dismiss.add_argument("recommendation_ids", type=int, nargs="+", metavar="recommendation_id")
    _common.add_project(pinbox_dismiss)
    pinbox_dismiss.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    pinbox_dismiss.set_defaults(func=_cmd_inbox_dismiss)

    pinbox_show = pinbox_sub.add_parser("show", help="print one recommendation with its evidence")
    pinbox_show.add_argument("recommendation_id", type=int)
    _common.add_project(pinbox_show)
    pinbox_show.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    pinbox_show.set_defaults(func=_cmd_inbox_show)

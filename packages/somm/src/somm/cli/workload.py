"""somm workload: register, constrain and inspect workloads."""

from __future__ import annotations

import argparse
import json
import sys

from somm_core.boundary import TIER_NAMES
from somm_core.models import PrivacyClass
from somm_core.repository import Repository

from somm.cli import _common

WORKLOAD_EXAMPLES: dict[str, dict] = {
    "structured-extraction": {
        "description": "Structured extraction workload",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Source text to extract from"}
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "quality_criteria": [
            "Return valid JSON matching the output schema.",
            "Extract only facts supported by the input text.",
        ],
    },
    "freeform": {
        "description": "",
        "input_schema": None,
        "output_schema": None,
        "quality_criteria": [],
    },
}

def _cmd_workload_add(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    template = WORKLOAD_EXAMPLES[args.from_example]
    description = args.description
    if description is None:
        description = template["description"]
    wl = repo.register_workload(
        name=args.name,
        project=cfg.project,
        description=description,
        input_schema=template["input_schema"],
        output_schema=template["output_schema"],
        quality_criteria=template["quality_criteria"],
        privacy_class=PrivacyClass(args.privacy_class),
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_p95_ttft_ms=args.max_p95_ttft_ms,
        max_tpot_ms=args.max_tpot_ms,
        max_capability_failure_rate=args.max_capability_failure_rate,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
        data_boundary=args.data_boundary,
    )
    from somm_core.boundary import resolve_boundary

    effective = resolve_boundary(wl, project=cfg.project)
    print(f"registered workload {wl.name!r} for project {cfg.project!r}")
    print(f"privacy_class: {wl.privacy_class.value}")
    print(f"data_boundary: {effective.tier.value} (from {effective.source})")
    print(f"input_schema: {'yes' if wl.input_schema else 'no'}")
    print(f"output_schema: {'yes' if wl.output_schema else 'no'}")
    return 0


def _workload_rows(repo: Repository, project: str) -> list[dict]:
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT
                w.id,
                w.name,
                w.description,
                w.input_schema_json,
                w.output_schema_json,
                w.quality_criteria_json,
                w.budget_cap_usd_daily,
                w.privacy_class,
                w.capabilities_required_json,
                w.max_p95_latency_ms,
                w.max_p95_ttft_ms,
                w.max_tpot_ms,
                w.max_capability_failure_rate,
                w.max_cost_per_call_usd,
                w.created_at,
                COUNT(c.id) AS call_count,
                w.data_boundary
            FROM workloads w
            LEFT JOIN calls c ON c.workload_id = w.id AND c.project = w.project
            WHERE w.project = ?
            GROUP BY w.id
            ORDER BY w.created_at DESC, w.name
            """,
            (project,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "description": r[2] or "",
            "input_schema": json.loads(r[3]) if r[3] else None,
            "output_schema": json.loads(r[4]) if r[4] else None,
            "quality_criteria": json.loads(r[5]) if r[5] else [],
            "budget_cap_usd_daily": r[6],
            "privacy_class": r[7],
            "capabilities_required": json.loads(r[8]) if r[8] else [],
            "max_p95_latency_ms": r[9],
            "max_p95_ttft_ms": r[10],
            "max_tpot_ms": r[11],
            "max_capability_failure_rate": r[12],
            "max_cost_per_call_usd": r[13],
            "created_at": r[14],
            "call_count": r[15],
            "data_boundary": r[16],
        }
        for r in rows
    ]


def _cmd_workload_list(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = _workload_rows(repo, cfg.project)
    if not rows:
        print(f"No workloads registered for project {cfg.project!r}.")
        print("Register one with `somm workload add <name>`.")
        return 0
    print(f"Project: {cfg.project}")
    print(f"{'name':<28} {'privacy':<10} {'boundary':<11} {'calls':>8}")
    for row in rows:
        print(
            f"{row['name'][:27]:<28} {row['privacy_class']:<10} "
            f"{(row['data_boundary'] or '-'):<11} {row['call_count']:>8}"
        )
    return 0


def _cmd_workload_show(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = [row for row in _workload_rows(repo, cfg.project) if row["name"] == args.name]
    if not rows:
        print(f"No workload {args.name!r} registered for project {cfg.project!r}.", file=sys.stderr)
        return 2
    row = rows[0]
    print(f"name: {row['name']}")
    print(f"project: {cfg.project}")
    print(f"id: {row['id']}")
    print(f"description: {row['description'] or '—'}")
    print(f"privacy_class: {row['privacy_class']}")
    wl = repo.workload_by_name(args.name, cfg.project)
    from somm_core.boundary import resolve_boundary

    effective = resolve_boundary(wl, project=cfg.project)
    print(
        f"data_boundary: {row['data_boundary'] or '—'} "
        f"(effective {effective.tier.value}, from {effective.source})"
    )
    print(f"call_count: {row['call_count']}")
    print(f"created_at: {row['created_at']}")
    print(
        f"budget_cap_usd_daily: {row['budget_cap_usd_daily'] if row['budget_cap_usd_daily'] is not None else '—'}"
    )
    print(f"capabilities_required: {', '.join(row['capabilities_required']) or '—'}")
    print("constraints:")
    print(
        f"  max_p95_latency_ms: {row['max_p95_latency_ms'] if row['max_p95_latency_ms'] is not None else '—'}"
    )
    print(
        f"  max_p95_ttft_ms: {row['max_p95_ttft_ms'] if row['max_p95_ttft_ms'] is not None else '—'}"
    )
    print(f"  max_tpot_ms: {row['max_tpot_ms'] if row['max_tpot_ms'] is not None else '—'}")
    print(
        f"  max_capability_failure_rate: {row['max_capability_failure_rate'] if row['max_capability_failure_rate'] is not None else '—'}"
    )
    print(
        f"  max_cost_per_call_usd: {row['max_cost_per_call_usd'] if row['max_cost_per_call_usd'] is not None else '—'}"
    )
    print("input_schema:")
    print(json.dumps(row["input_schema"], indent=2) if row["input_schema"] else "  —")
    print("output_schema:")
    print(json.dumps(row["output_schema"], indent=2) if row["output_schema"] else "  —")
    if row["quality_criteria"]:
        print("quality_criteria:")
        for item in row["quality_criteria"]:
            print(f"  - {item}")
    else:
        print("quality_criteria: —")
    return 0


def _cmd_workload_set_constraints(args: argparse.Namespace) -> int:
    cfg = _common.load_config(project=args.project)
    repo = Repository(cfg.db_path)
    wl = repo.workload_by_name(args.name, cfg.project)
    if wl is None:
        print(f"No workload {args.name!r} registered for project {cfg.project!r}.", file=sys.stderr)
        return 2
    if not args.clear and all(
        value is None
        for value in (
            args.max_p95_latency_ms,
            args.max_p95_ttft_ms,
            args.max_tpot_ms,
            args.max_capability_failure_rate,
            args.max_cost_per_call_usd,
        )
    ):
        print(
            "No constraints supplied; pass at least one --max-* option or --clear.", file=sys.stderr
        )
        return 2
    repo.set_workload_constraints(
        wl.id,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_p95_ttft_ms=args.max_p95_ttft_ms,
        max_tpot_ms=args.max_tpot_ms,
        max_capability_failure_rate=args.max_capability_failure_rate,
        max_cost_per_call_usd=args.max_cost_per_call_usd,
        clear=args.clear,
    )
    print(f"updated constraints for workload {wl.name!r}")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pw = sub.add_parser("workload", help="register and inspect project workloads")
    pw_sub = pw.add_subparsers(dest="workload_cmd", required=True)

    pwa = pw_sub.add_parser(
        "add", aliases=["create"], help="register a workload in the project DB"
    )
    pwa.add_argument("name", help="workload name")
    pwa.add_argument("--project", default=None)
    pwa.add_argument("--description", default=None)
    pwa.add_argument(
        "--privacy-class",
        choices=[pc.value for pc in PrivacyClass],
        default=PrivacyClass.INTERNAL.value,
    )
    pwa.add_argument(
        "--from-example",
        choices=sorted(WORKLOAD_EXAMPLES),
        default="freeform",
        help="built-in workload template (default: freeform)",
    )
    pwa.add_argument("--max-p95-latency-ms", type=int, default=None)
    pwa.add_argument("--max-p95-ttft-ms", type=int, default=None)
    pwa.add_argument("--max-tpot-ms", type=float, default=None)
    pwa.add_argument("--max-capability-failure-rate", type=float, default=None)
    pwa.add_argument("--max-cost-per-call-usd", type=float, default=None)
    pwa.add_argument(
        "--data-boundary",
        choices=list(TIER_NAMES),
        default=None,
        help="loosest route tier this workload's data may reach (default: derive)",
    )
    pwa.set_defaults(func=_cmd_workload_add)

    pwl = pw_sub.add_parser("list", help="list registered workloads")
    pwl.add_argument("--project", default=None)
    pwl.set_defaults(func=_cmd_workload_list)

    pws = pw_sub.add_parser("show", help="show one registered workload")
    pws.add_argument("name", help="workload name")
    pws.add_argument("--project", default=None)
    pws.set_defaults(func=_cmd_workload_show)

    pwc = pw_sub.add_parser("set-constraints", help="update workload adequacy constraints")
    pwc.add_argument("name", help="workload name")
    pwc.add_argument("--project", default=None)
    pwc.add_argument("--max-p95-latency-ms", type=int, default=None)
    pwc.add_argument("--max-p95-ttft-ms", type=int, default=None)
    pwc.add_argument("--max-tpot-ms", type=float, default=None)
    pwc.add_argument("--max-capability-failure-rate", type=float, default=None)
    pwc.add_argument("--max-cost-per-call-usd", type=float, default=None)
    pwc.add_argument("--clear", action="store_true", help="clear all workload constraints")
    pwc.set_defaults(func=_cmd_workload_set_constraints)

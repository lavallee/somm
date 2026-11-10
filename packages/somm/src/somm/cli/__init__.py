"""somm CLI entry point — grouped subcommands.

Commands:
  somm status          roll-up per (workload, provider, model)
  somm cache-advice    find low prefix-cache reuse opportunities
  somm bench           run latency/throughput probes through somm
  somm generate        one-shot LLM call through somm
    somm tail            live call stream
  somm compare         run a prompt through N models side-by-side
  somm frontier        adequacy frontier per (provider, model) for a workload
  somm doctor          health check (config, ollama, db, intel, workers, loop, fleet)
  somm serve           run the web admin + HTTP API (requires somm-service)
  somm spend           today's LLM spend vs daily budget cap per workload
  somm gpu spend       rented-GPU spend read from gipper receipts, by day and workload
  somm backfill-costs  recompute cost_usd for calls missing pricing
  somm plans           metered-plan quota usage + pacing
  somm drain-spool     replay spooled telemetry into the DB
  somm loop            run or schedule the intelligence loop
  somm prune           delete calls older than a retention window
  somm alias           canonical model identities (list, seed, set)
  somm workload        register, constrain, and inspect project workloads
  somm boundary        data boundary: route tiers, audit, per-workload boundary
  somm prompt          manage prompt versions, labels, and A/B variants
  somm eval            promote calls to datasets and run eval gates
  somm optimize        propose a prompt fork from failing graded calls
  somm campaign        run durable experiment campaigns
  somm inbox           list, apply, and dismiss recommendations
  somm plugin          list and inspect plugins, hooks, and providers

One module per command group; each exposes ``add_parsers(sub)`` and holds its
own handlers. ``_common`` holds what more than one group needs.
"""

from __future__ import annotations

import argparse
import json
import sys

from somm_core import VERSION
from somm_core.read import spend_today  # noqa: F401

from somm.cli import (
    alias,
    boundary,
    calls,
    connect,
    doctor,
    evals,
    fleet,
    inbox,
    loop,
    plugin,
    prompt,
    serve,
    spend,
    upkeep,
    workload,
)

# Names that tests and other callers import from `somm.cli`. Each lives in the
# module that owns it and is re-exported here so `from somm.cli import X` keeps
# working. A test that monkeypatches `load_config` must patch
# `somm.cli._common.load_config`, the one every command module calls.
from somm.cli._common import load_config  # noqa: F401
from somm.cli.calls import (  # noqa: F401
    _error_message,
    _fetch_since,
    _fmt_cost,
    _parse_model_specs,
    _print_comparison,
)
from somm.cli.doctor import (  # noqa: F401
    _age_since,
    _age_until,
    _call_site_coverage_24h,
    _exhausted_rate_24h,
    _fmt_delta,
    _workloads_without_output_schema,
)
from somm.cli.evals import _dataset_judge, _load_eval_judge_config  # noqa: F401
from somm.cli.workload import WORKLOAD_EXAMPLES  # noqa: F401
from somm.providers.ollama import OllamaProvider  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm", description="somm — self-hosted LLM telemetry")
    p.add_argument("--version", action="version", version=f"somm {VERSION}")
    p.add_argument(
        "--project",
        default=None,
        help="project name; also accepted after any command (default: $SOMM_PROJECT, "
        "else this directory's project)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    from somm.cli_chatgpt import add_parser as add_chatgpt_parser

    add_chatgpt_parser(sub)

    # Registration order is the order `somm --help` lists the commands in.
    for group in (
        calls,
        workload,
        boundary,
        prompt,
        evals,
        inbox,
        plugin,
        connect,
        doctor,
        loop,
        fleet,
        alias,
        serve,
        spend,
        upkeep,
    ):
        group.add_parsers(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if getattr(args, "json", False):
            code = _json_error_exit_code(exc)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "exit_code": code,
                        },
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return code
        raise


def _json_error_exit_code(exc: Exception) -> int:
    if isinstance(exc, (ValueError, argparse.ArgumentError)):
        return 2
    if isinstance(exc, FileNotFoundError):
        return 66
    if isinstance(exc, PermissionError):
        return 77
    return 1

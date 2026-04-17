"""somm CLI entry point — minimal D1 scaffold (status + doctor + --version).

Grouped CLI structure (show/admin/service/workload/prompt/provider/mcp)
lands in D2+; D1 ships the smoke-path commands only.
"""

from __future__ import annotations

import argparse
import sys

from somm_core import VERSION
from somm_core.config import load as load_config
from somm_core.repository import Repository

from somm.providers.ollama import OllamaProvider


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    stats = repo.stats_by_workload(cfg.project, since_days=args.since)
    if not stats:
        print(f"No calls yet for project {cfg.project!r} in the last {args.since} days.")
        print(f"Run `somm.llm({cfg.project!r}).generate(...)` in your Python code.")
        return 0
    print(f"Project: {cfg.project}  ({args.since}d window)")
    print(
        f"{'workload':<24} {'provider':<12} {'model':<18} {'n':>6} {'tok_in':>8} {'tok_out':>8} {'fail':>6}"
    )
    for s in stats:
        print(
            f"{s['workload'][:23]:<24} {s['provider'][:11]:<12} {s['model'][:17]:<18} "
            f"{s['n_calls']:>6} {(s['tokens_in'] or 0):>8} {(s['tokens_out'] or 0):>8} {s['n_failed']:>6}"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(project=args.project)
    print(f"somm v{VERSION}")
    print(f"project: {cfg.project}  mode: {cfg.mode}")
    print(f"db_path: {cfg.db_path}   exists: {cfg.db_path.exists()}")
    if cfg.db_path.exists():
        mode = oct(cfg.db_path.stat().st_mode)[-3:]
        ok = mode in ("600",)
        print(f"db perms: {mode} {'(ok)' if ok else '(WARN — expect 600)'}")
    p = OllamaProvider(base_url=cfg.ollama_url, default_model=cfg.ollama_model)
    h = p.health()
    print(f"ollama: {'ok' if h.available else 'UNAVAILABLE'}  ({h.detail})")
    return 0 if h.available else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm", description="somm — self-hosted LLM telemetry")
    p.add_argument("--version", action="version", version=f"somm {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("status", help="show call roll-up for the current project")
    ps.add_argument("--project", default=None)
    ps.add_argument("--since", type=int, default=7, help="window in days (default 7)")
    ps.set_defaults(func=_cmd_status)

    pd = sub.add_parser("doctor", help="check config + ollama + db health")
    pd.add_argument("--project", default=None)
    pd.set_defaults(func=_cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

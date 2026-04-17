"""`somm-serve` + admin CLI — standalone entry point.

`somm serve` in the `somm` package also routes here.
"""

from __future__ import annotations

import argparse
import sys

from somm_core.config import load as load_config
from somm_core.repository import Repository

from somm_service.app import run_server


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm-serve", description="somm web admin + HTTP API")
    sub = p.add_subparsers(dest="cmd")

    # Default (no sub): serve
    p.add_argument("--project", default=None, help="project name (overrides env/config)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: localhost)")
    p.add_argument("--port", type=int, default=7878, help="bind port (default: 7878)")
    p.add_argument("--log-level", default="info")

    # Admin subcommands
    admin = sub.add_parser("admin", help="admin commands")
    admin_sub = admin.add_subparsers(dest="admin_cmd", required=True)

    pri = admin_sub.add_parser("refresh-intel", help="refresh model_intel cache")
    pri.add_argument("--project", default=None)
    pri.set_defaults(func=_cmd_refresh_intel)

    pil = admin_sub.add_parser("list-intel", help="show model_intel entries")
    pil.add_argument("--project", default=None)
    pil.add_argument("--provider", default=None)
    pil.set_defaults(func=_cmd_list_intel)

    return p


def _cmd_refresh_intel(args: argparse.Namespace) -> int:
    from somm_service.workers.model_intel import ModelIntelWorker

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    worker = ModelIntelWorker(repo, ollama_url=cfg.ollama_url)
    print(f"refreshing model_intel (db={cfg.db_path})...")
    summary = worker.run_once()
    print(f"  openrouter: {summary['openrouter']} models")
    print(f"  ollama:     {summary['ollama']} models")
    print(f"  static:     {summary['static']} models")
    if summary["errors"]:
        print("  errors:")
        for e in summary["errors"]:
            print(f"    {e}")
    return 0 if not summary["errors"] else 1


def _cmd_list_intel(args: argparse.Namespace) -> int:
    from somm_core import list_intel

    cfg = load_config(project=args.project)
    repo = Repository(cfg.db_path)
    rows = list_intel(repo, provider=args.provider)
    if not rows:
        print("no model_intel entries. run `somm admin refresh-intel` first.")
        return 0
    print(f"{'provider':<14} {'model':<50} {'in $/1M':>10} {'out $/1M':>10} {'ctx':>8}  source")
    for r in rows:
        ctx = str(r["context_window"]) if r["context_window"] else "—"
        print(
            f"{r['provider'][:13]:<14} {r['model'][:49]:<50} "
            f"{r['price_in_per_1m']:>10.4f} {r['price_out_per_1m']:>10.4f} "
            f"{ctx:>8}  {r['source']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "cmd", None) == "admin":
        return args.func(args)

    cfg = load_config(project=args.project)
    try:
        run_server(config=cfg, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:
        return 130
    except OSError as e:
        if "already in use" in str(e).lower() or "Address already in use" in str(e):
            print(
                "\nSOMM_PORT_BUSY\n\n"
                f"Problem: {args.host}:{args.port} is already in use.\n"
                "Cause: another somm service or local process is bound to this port.\n"
                "Fix:\n"
                f"  somm-serve --port {args.port + 1}\n"
                f"  lsof -i :{args.port}\n"
                "Docs: docs/errors/SOMM_PORT_BUSY.md\n",
                file=sys.stderr,
            )
            return 2
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

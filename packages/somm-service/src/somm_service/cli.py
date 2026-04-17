"""`somm-serve` CLI — standalone entry point.

`somm serve` in the `somm` package also routes here.
"""

from __future__ import annotations

import argparse
import sys

from somm_core.config import load as load_config

from somm_service.app import run_server


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm-serve", description="somm web admin + HTTP API")
    p.add_argument("--project", default=None, help="project name (overrides env/config)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: localhost)")
    p.add_argument("--port", type=int, default=7878, help="bind port (default: 7878)")
    p.add_argument("--log-level", default="info")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

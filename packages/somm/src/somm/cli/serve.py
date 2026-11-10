"""somm serve (thin shim to somm-service)."""

from __future__ import annotations

import argparse
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from somm_service.cli import main as serve_main
    except ImportError:
        print(
            "somm serve requires somm-service.\n"
            "  uv add somm-service    # or: pip install somm-service",
            file=sys.stderr,
        )
        return 2
    forwarded = []
    if args.project:
        forwarded += ["--project", args.project]
    forwarded += ["--host", args.host, "--port", str(args.port)]
    return serve_main(forwarded)


def add_parsers(sub: argparse._SubParsersAction) -> None:
    psr = sub.add_parser("serve", help="run the web admin + HTTP API (localhost:7878)")
    psr.add_argument("--project", default=None)
    psr.add_argument("--host", default="127.0.0.1")
    psr.add_argument("--port", type=int, default=7878)
    psr.set_defaults(func=_cmd_serve)

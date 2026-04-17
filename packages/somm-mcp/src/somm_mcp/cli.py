"""`somm-mcp` CLI — runs the stdio MCP server."""

from __future__ import annotations

import argparse
import sys

from somm_core.config import load as load_config

from somm_mcp.server import build_server


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm-mcp", description="somm MCP server (stdio)")
    p.add_argument("--project", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(project=args.project)
    server = build_server(cfg)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

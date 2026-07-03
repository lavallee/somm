"""`somm-mcp` CLI — runs the stdio MCP server with the full provider chain."""

from __future__ import annotations

import argparse
import sys

from somm_core.config import Config
from somm_core.config import load as load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="somm-mcp", description="somm MCP server (stdio)")
    p.add_argument("--project", default=None)
    return p


def _providers_from_config(cfg: Config):
    """Build the default provider chain — the exact chain SommLLM builds,
    so somm_compare / somm_replay can reach every provider the library can
    (including gemini, deepseek, perplexity, and the CLI executors)."""
    from somm.client import build_default_providers

    return build_default_providers(cfg)


def main(argv: list[str] | None = None) -> int:
    from somm_mcp.server import build_server

    args = build_parser().parse_args(argv)
    cfg = load_config(project=args.project)
    providers = _providers_from_config(cfg)
    server = build_server(cfg, providers=providers)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

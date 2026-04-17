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
    """Build the default provider chain (same as SommLLM's default)."""
    from somm.providers.anthropic import AnthropicProvider
    from somm.providers.minimax import MinimaxProvider
    from somm.providers.ollama import OllamaProvider
    from somm.providers.openai import OpenAIProvider
    from somm.providers.openrouter import OpenRouterProvider

    chain = [OllamaProvider(base_url=cfg.ollama_url, default_model=cfg.ollama_model)]
    if cfg.openrouter_api_key:
        chain.append(
            OpenRouterProvider(
                api_key=cfg.openrouter_api_key,
                roster=cfg.openrouter_roster,
            )
        )
    if cfg.minimax_api_key:
        chain.append(
            MinimaxProvider(
                api_key=cfg.minimax_api_key,
                default_model=cfg.minimax_model,
            )
        )
    if cfg.anthropic_api_key:
        chain.append(
            AnthropicProvider(
                api_key=cfg.anthropic_api_key,
                default_model=cfg.anthropic_model,
            )
        )
    if cfg.openai_api_key:
        chain.append(
            OpenAIProvider(
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_base_url,
                default_model=cfg.openai_model,
            )
        )
    return chain


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

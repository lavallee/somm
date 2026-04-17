"""FastMCP server definition.

Keeps the tool handlers thin — they delegate to `somm-core.Repository`.
The same handlers will be reused by the HTTP transport in D2.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.repository import Repository


def build_server(config: Config | None = None) -> FastMCP:
    cfg = config or load_config()
    repo = Repository(cfg.db_path)
    server = FastMCP("somm")

    @server.tool()
    def somm_stats(since_days: int = 7) -> dict:
        """Rolled-up call counts + token + cost + failure stats per (workload, provider, model).

        Args:
            since_days: Window in days (default 7).

        Returns:
            JSON-serializable dict with 'project', 'window_days', and 'rows' (list of
            {workload, provider, model, n_calls, tokens_in, tokens_out, cost_usd,
             latency_ms_avg, n_failed}).
        """
        rows = repo.stats_by_workload(cfg.project, since_days=since_days)
        return {
            "project": cfg.project,
            "window_days": since_days,
            "rows": rows,
        }

    return server

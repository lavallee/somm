"""somm-mcp — MCP server exposing somm telemetry + recommendations.

v0.1 ships one tool: `somm_stats`. Additional tools (`somm_recommend`,
`somm_compare`, `somm_replay`, `somm_search_calls`, `somm_register_workload`,
`somm_register_prompt`) land in D2+.

Transport: stdio (direct to somm-core). HTTP transport to a running
`somm serve` is D2.
"""

from somm_mcp.server import build_server

__all__ = ["build_server"]

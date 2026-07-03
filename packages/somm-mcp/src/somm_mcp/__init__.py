"""somm-mcp — MCP server exposing somm telemetry + recommendations.

Ships 10 stdio tools: `somm_stats`, `somm_search_calls`, `somm_recommend`,
`somm_register_workload`, `somm_register_prompt`, `somm_compare`,
`somm_replay`, `somm_advise`, `somm_record_decision`,
`somm_search_decisions`.

Transport: stdio (direct to somm-core).
"""

from somm_mcp.server import build_server

__all__ = ["build_server"]

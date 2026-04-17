"""Tests for the somm-mcp server. Exercises the handler function directly
(no stdio transport) since the tool is thin over the Repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from somm_core.config import Config


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "mcp-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def test_build_server_exposes_somm_stats(tmp_path):
    """Sanity: server builds, has a somm_stats tool registered."""
    from somm_mcp.server import build_server

    cfg = _tmp_config(tmp_path)
    server = build_server(cfg)

    # FastMCP exposes tools via `_tool_manager` or `list_tools` helper across versions;
    # use the one that exists. Test accepts both shapes.
    tool_names: list[str] = []
    for attr in ("_tool_manager", "tool_manager"):
        mgr = getattr(server, attr, None)
        if mgr is not None:
            tools = getattr(mgr, "list_tools", lambda: [])()
            tool_names = [t.name for t in tools]
            break
    if not tool_names:
        # Fallback: some versions keep tools on the server directly
        tools_attr = getattr(server, "tools", None) or getattr(server, "_tools", None)
        if tools_attr:
            tool_names = list(tools_attr.keys()) if isinstance(tools_attr, dict) else []

    assert "somm_stats" in tool_names


@pytest.mark.asyncio
async def test_somm_stats_returns_rollup(tmp_path):
    """Call the somm_stats handler end-to-end against a populated DB."""
    from somm.client import SommLLM
    from somm.providers.base import ProviderHealth, SommResponse
    from somm_mcp.server import build_server

    cfg = _tmp_config(tmp_path)

    class FakeProvider:
        name = "fake"

        def generate(self, request):
            return SommResponse(text="ok", model="fake-m", tokens_in=5, tokens_out=2, latency_ms=3)

        def stream(self, request):  # pragma: no cover
            yield

        def health(self):
            return ProviderHealth(available=True)

        def models(self):
            return []

        def estimate_tokens(self, text, model):
            return 1

    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    for _ in range(2):
        llm.generate("p", workload="mcp_rollup")
    llm.close()

    server = build_server(cfg)

    # FastMCP stores handlers in a way we can introspect via the tool_manager.
    # Use call_tool with arguments — works across recent mcp versions.
    result = await server.call_tool("somm_stats", {"since_days": 1})

    # Result is either a list of content items (newer mcp) or a dict.
    # Extract the structured content in either case.
    if isinstance(result, tuple):
        content, structured = result
    else:
        content, structured = result, None

    # FastMCP returns the tool's return value as structured content when it's a dict.
    if structured is not None:
        data = structured
    elif isinstance(content, dict):
        data = content
    else:
        # content is a list of TextContent; parse the JSON from first text block
        import json as _json

        data = _json.loads(content[0].text)

    assert data["project"] == "mcp-test"
    assert data["window_days"] == 1
    assert len(data["rows"]) == 1
    assert data["rows"][0]["workload"] == "mcp_rollup"
    assert data["rows"][0]["n_calls"] == 2

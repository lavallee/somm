"""MCP sommelier tools — somm_advise, somm_record_decision, somm_search_decisions,
plus the cold-start branch of somm_recommend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from somm_core.config import Config
from somm_core.pricing import write_intel
from somm_core.repository import Repository


def _tmp_cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "mcp-som"
    cfg.db_dir = tmp_path / ".somm"
    cfg.cross_project_path = tmp_path / "global.sqlite"
    return cfg


async def _call(server, tool, **args):
    result = await server.call_tool(tool, args)
    if isinstance(result, tuple):
        content, structured = result
    else:
        content, structured = result, None
    if structured is not None:
        return structured
    if isinstance(content, dict):
        return content
    return json.loads(content[0].text)


def _seed_vision_intel(repo: Repository):
    write_intel(
        repo, "openrouter", "google/gemma-3-27b-it:free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=128_000,
        capabilities={"modality": "text+image->text"},
        source="openrouter",
    )
    write_intel(
        repo, "openrouter", "meta-llama/llama-3.3-70b-instruct:free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=128_000,
        capabilities={"modality": "text->text"},
        source="openrouter",
    )


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advise_returns_ranked_vision_candidates(tmp_path, monkeypatch):
    cfg = _tmp_cfg(tmp_path)
    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(cfg.cross_project_path))
    repo = Repository(cfg.db_path)
    _seed_vision_intel(repo)

    from somm_mcp.server import build_server

    server = build_server(config=cfg)
    res = await _call(
        server,
        "somm_advise",
        question="good free vision models on openrouter",
        capabilities=["vision"],
        providers=["openrouter"],
        free_only=True,
    )
    names = {(c["provider"], c["model"]) for c in res["candidates"]}
    assert ("openrouter", "google/gemma-3-27b-it:free") in names
    # Non-vision llama must be filtered out.
    assert ("openrouter", "meta-llama/llama-3.3-70b-instruct:free") not in names
    assert res["constraints"]["capabilities"] == ["vision"]


@pytest.mark.asyncio
async def test_record_decision_mirrors_to_global(tmp_path, monkeypatch):
    cfg = _tmp_cfg(tmp_path)
    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(cfg.cross_project_path))
    Repository(cfg.db_path)  # ensure primary exists

    from somm_mcp.server import build_server

    server = build_server(config=cfg)
    res = await _call(
        server,
        "somm_record_decision",
        question="vision pick",
        rationale="gemma-3 has biggest free vision ctx",
        candidates=[{"provider": "openrouter", "model": "google/gemma-3-27b-it:free"}],
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
        agent="claude-code-test",
    )
    assert res["mirrored"] is True
    assert res["decision_id"]
    # Global store has the row.
    global_repo = Repository(cfg.cross_project_path)
    stored = global_repo.get_decision(res["decision_id"])
    assert stored is not None
    assert stored.agent == "claude-code-test"


@pytest.mark.asyncio
async def test_search_decisions_global_scope(tmp_path, monkeypatch):
    cfg = _tmp_cfg(tmp_path)
    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(cfg.cross_project_path))
    Repository(cfg.db_path)

    from somm_mcp.server import build_server

    server = build_server(config=cfg)
    # Record via the MCP tool so we exercise the real write path.
    await _call(
        server,
        "somm_record_decision",
        question="what vision model for chart critique",
        rationale="gemma-3 best for charts",
        candidates=[{"provider": "openrouter", "model": "google/gemma-3-27b-it:free"}],
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
    )
    # Search by loose LIKE match.
    res = await _call(
        server,
        "somm_search_decisions",
        question="chart critique",
        scope="global",
    )
    assert res["count"] == 1
    assert res["decisions"][0]["chosen_model"].startswith("google/gemma-3-27b")


@pytest.mark.asyncio
async def test_recommend_falls_through_to_cold_start(tmp_path, monkeypatch):
    """When a workload has no shadow data, somm_recommend should surface
    cold-start candidates from the sommelier instead of an empty list."""
    cfg = _tmp_cfg(tmp_path)
    monkeypatch.setenv("SOMM_GLOBAL_PATH", str(cfg.cross_project_path))
    repo = Repository(cfg.db_path)
    _seed_vision_intel(repo)
    repo.register_workload(
        name="critique_visual",
        project=cfg.project,
        capabilities_required=["vision"],
    )

    from somm_mcp.server import build_server

    server = build_server(config=cfg)
    res = await _call(server, "somm_recommend", workload="critique_visual")
    assert res["shadow_rankings"] == []
    assert res["cold_start_candidates"]
    providers = {c["provider"] for c in res["cold_start_candidates"]}
    # Only vision-capable models should have survived the capability filter.
    assert "openrouter" in providers
    for cand in res["cold_start_candidates"]:
        # Vision-free llama should not be in here.
        assert cand["model"] != "meta-llama/llama-3.3-70b-instruct:free"
    assert res["capabilities_required"] == ["vision"]

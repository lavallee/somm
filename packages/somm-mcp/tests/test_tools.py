"""MCP tool tests — exercise each of the 7 tools end-to-end.

FastMCP's internal shape varies across versions, so we call tools via
`server.call_tool()` which is stable. Each test seeds the DB with the
fixtures it needs.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.models import Call, Outcome, PrivacyClass
from somm_core.repository import Repository


def _tmp_cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "mcp"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


async def _call(server, tool, **args):
    """Call an MCP tool and unwrap the structured content into a dict."""
    result = await server.call_tool(tool, args)
    if isinstance(result, tuple):
        content, structured = result
    else:
        content, structured = result, None
    if structured is not None:
        return structured
    if isinstance(content, dict):
        return content
    # FastMCP may return list[TextContent]
    return json.loads(content[0].text)


class FakeProvider:
    """Echoes input; records invocations."""

    def __init__(self, name: str, response_text: str = "ok"):
        self.name = name
        self.response_text = response_text
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return SommResponse(
            text=self.response_text,
            model=request.model or f"{self.name}-default",
            tokens_in=5,
            tokens_out=2,
            latency_ms=10,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


# ---------------------------------------------------------------------------
# tool: somm_stats


@pytest.mark.asyncio
async def test_stats_empty(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    server = build_server(cfg)
    data = await _call(server, "somm_stats", since_days=7)
    assert data["project"] == "mcp"
    assert data["rows"] == []


# ---------------------------------------------------------------------------
# tool: somm_register_workload + somm_register_prompt


@pytest.mark.asyncio
async def test_register_workload_and_prompt(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    server = build_server(cfg)

    wl = await _call(
        server,
        "somm_register_workload",
        name="contact_extract",
        description="extract contacts",
        privacy_class="internal",
    )
    assert "workload_id" in wl
    assert wl["name"] == "contact_extract"

    p1 = await _call(
        server,
        "somm_register_prompt",
        workload="contact_extract",
        body="Extract contacts from {text}",
    )
    assert p1["version"] == "v1"

    p2 = await _call(
        server,
        "somm_register_prompt",
        workload="contact_extract",
        body="Extract contacts and roles from {text}",
    )
    assert p2["version"] == "v1.1"


@pytest.mark.asyncio
async def test_register_workload_bad_privacy_class(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    server = build_server(cfg)
    res = await _call(
        server,
        "somm_register_workload",
        name="x",
        privacy_class="SECRET",
    )
    assert "error" in res
    assert "privacy_class" in res["error"]


@pytest.mark.asyncio
async def test_register_prompt_unknown_workload(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    server = build_server(cfg)
    res = await _call(
        server,
        "somm_register_prompt",
        workload="ghost",
        body="...",
    )
    assert "error" in res


# ---------------------------------------------------------------------------
# tool: somm_search_calls


@pytest.mark.asyncio
async def test_search_calls_filters(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="w1", project=cfg.project)
    for provider, model, outcome in (
        ("ollama", "a", Outcome.OK),
        ("ollama", "a", Outcome.BAD_JSON),
        ("openai", "b", Outcome.OK),
    ):
        repo.write_call(
            Call(
                id=str(uuid.uuid4()),
                ts=datetime.now(UTC),
                project=cfg.project,
                workload_id=wl.id,
                prompt_id=None,
                provider=provider,
                model=model,
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                cost_usd=0.0,
                outcome=outcome,
                error_kind=None,
                prompt_hash="a",
                response_hash="b",
            )
        )

    server = build_server(cfg)
    all_rows = await _call(server, "somm_search_calls", since_days=7)
    assert all_rows["count"] == 3

    only_ollama = await _call(server, "somm_search_calls", provider="ollama")
    assert only_ollama["count"] == 2

    only_bad = await _call(server, "somm_search_calls", outcome="bad_json")
    assert only_bad["count"] == 1
    assert only_bad["rows"][0]["outcome"] == "bad_json"


# ---------------------------------------------------------------------------
# tool: somm_recommend


@pytest.mark.asyncio
async def test_recommend_unknown_workload_errors(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    server = build_server(cfg)
    res = await _call(server, "somm_recommend", workload="ghost")
    assert "error" in res


@pytest.mark.asyncio
async def test_recommend_with_open_rec(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="w_rec", project=cfg.project)
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO recommendations "
            "(workload_id, action, evidence_json, expected_impact, confidence) "
            "VALUES (?, 'switch_model', '{}', 'save $', 0.8)",
            (wl.id,),
        )

    server = build_server(cfg)
    res = await _call(server, "somm_recommend", workload="w_rec")
    assert res["workload"] == "w_rec"
    assert len(res["open_recommendations"]) == 1
    assert res["open_recommendations"][0]["action"] == "switch_model"


@pytest.mark.asyncio
async def test_recommend_shadow_ranking(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="w_rank", project=cfg.project)
    # Seed calls + eval_results for two models
    for provider, model, score in (
        ("ollama", "a", 0.4),
        ("ollama", "a", 0.45),
        ("ollama", "b", 0.85),
        ("ollama", "b", 0.9),
    ):
        cid = str(uuid.uuid4())
        repo.write_call(
            Call(
                id=cid,
                ts=datetime.now(UTC),
                project=cfg.project,
                workload_id=wl.id,
                prompt_id=None,
                provider=provider,
                model=model,
                tokens_in=1,
                tokens_out=1,
                latency_ms=100,
                cost_usd=0.0,
                outcome=Outcome.OK,
                error_kind=None,
                prompt_hash="a",
                response_hash="b",
            )
        )
        with repo._open() as conn:
            conn.execute(
                "INSERT INTO eval_results "
                "(call_id, gold_model, structural_score, embedding_score) "
                "VALUES (?, 'g', ?, ?)",
                (cid, score, score),
            )

    server = build_server(cfg)
    res = await _call(server, "somm_recommend", workload="w_rank")
    rankings = res["shadow_rankings"]
    assert len(rankings) == 2
    # Best model should be "b" (higher score)
    assert rankings[0]["model"] == "b"
    assert rankings[0]["score"] > rankings[1]["score"]


# ---------------------------------------------------------------------------
# tool: somm_compare


@pytest.mark.asyncio
async def test_compare_no_providers_errors(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    server = build_server(cfg)  # no providers
    res = await _call(
        server,
        "somm_compare",
        prompt="hi",
        models=["ollama/gemma4:e4b"],
    )
    assert "error" in res


@pytest.mark.asyncio
async def test_compare_runs_each_model(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    p1 = FakeProvider("ollama", "from-ollama")
    p2 = FakeProvider("openai", "from-openai")
    server = build_server(cfg, providers=[p1, p2])

    res = await _call(
        server,
        "somm_compare",
        prompt="ping",
        models=["ollama/x", "openai/y"],
        max_tokens=32,
    )
    assert "results" in res
    assert len(res["results"]) == 2
    by_provider = {r["provider"]: r for r in res["results"]}
    assert by_provider["ollama"]["text"] == "from-ollama"
    assert by_provider["openai"]["text"] == "from-openai"
    assert p1.calls == 1
    assert p2.calls == 1


@pytest.mark.asyncio
async def test_compare_unknown_provider_in_list(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    p1 = FakeProvider("ollama")
    server = build_server(cfg, providers=[p1])

    res = await _call(
        server,
        "somm_compare",
        prompt="x",
        models=["ollama/a", "ghost/z"],
    )
    # Both slots present; one has an error
    assert len(res["results"]) == 2
    errors = [r for r in res["results"] if "error" in r]
    assert len(errors) == 1
    assert "ghost" in errors[0]["error"]


# ---------------------------------------------------------------------------
# tool: somm_replay


@pytest.mark.asyncio
async def test_replay_call_not_found(tmp_path):
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    p1 = FakeProvider("ollama")
    server = build_server(cfg, providers=[p1])
    res = await _call(
        server,
        "somm_replay",
        call_id="nonexistent-uuid",
        with_provider="ollama",
        with_model="a",
    )
    assert "error" in res
    assert "not found" in res["error"]


@pytest.mark.asyncio
async def test_replay_missing_sample(tmp_path):
    """Original call has no samples row → clear error, no egress."""
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="w", project=cfg.project)
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="ollama",
            model="a",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="a",
            response_hash="b",
        )
    )
    # No samples row inserted

    p1 = FakeProvider("ollama")
    server = build_server(cfg, providers=[p1])
    res = await _call(
        server,
        "somm_replay",
        call_id=call_id,
        with_provider="ollama",
        with_model="x",
    )
    assert "error" in res
    assert "captured" in res["error"]
    assert p1.calls == 0


@pytest.mark.asyncio
async def test_replay_private_workload_refused(tmp_path):
    """Private workload → replay refuses (no upstream egress)."""
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(
        name="secret",
        project=cfg.project,
        privacy_class=PrivacyClass.PRIVATE,
    )
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="ollama",
            model="a",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="a",
            response_hash="b",
        )
    )
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO samples (call_id, prompt_body, response_body) VALUES (?, ?, ?)",
            (call_id, "secret prompt", "secret response"),
        )

    p1 = FakeProvider("ollama")
    server = build_server(cfg, providers=[p1])
    res = await _call(
        server,
        "somm_replay",
        call_id=call_id,
        with_provider="ollama",
        with_model="x",
    )
    assert "error" in res
    assert "SOMM_PRIVACY_VIOLATION" in res["error"]
    assert p1.calls == 0


@pytest.mark.asyncio
async def test_replay_happy_path(tmp_path):
    """Original has sample + non-private workload → replay runs + returns deltas."""
    from somm_mcp.server import build_server

    cfg = _tmp_cfg(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="public_w", project=cfg.project)
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="ollama",
            model="slow",
            tokens_in=5,
            tokens_out=3,
            latency_ms=200,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="a",
            response_hash="b",
        )
    )
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO samples (call_id, prompt_body, response_body) VALUES (?, ?, ?)",
            (call_id, "extract contacts", "original response"),
        )

    p1 = FakeProvider("ollama", response_text="replay response")
    server = build_server(cfg, providers=[p1])
    res = await _call(
        server,
        "somm_replay",
        call_id=call_id,
        with_provider="ollama",
        with_model="fast",
    )
    assert "error" not in res
    assert res["original"]["response"] == "original response"
    assert res["replay"]["response"] == "replay response"
    assert res["replay"]["model"] == "fast"
    # Deltas computed
    assert res["deltas"]["latency_ms"] == res["replay"]["latency_ms"] - 200
    assert p1.calls == 1

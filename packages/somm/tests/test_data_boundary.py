"""Data boundary: classifier, resolution, every enforcement point, ledger, audit.

The audit probe (06-architecture-audit.md, SC1): a PRIVATE workload's
``generate()`` reached a provider named ``openai``. Every path below must now
refuse before any provider is touched.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm import SommLLM
from somm.errors import SommPrivacyViolation, SommTransientError, describe_error
from somm.providers.base import ProviderHealth, SommChunk, SommRequest, SommResponse
from somm.routing import ProviderHealthTracker, Router
from somm_core.boundary import (
    DataTier,
    admit,
    audit,
    classify,
    load_policy,
    parse_tier,
    resolve_boundary,
)
from somm_core.config import Config
from somm_core.models import Call, Outcome, PrivacyClass
from somm_core.repository import Repository
from somm_core.schema import _list_migrations, ensure_schema
from somm_core.version import SCHEMA_VERSION

LOOPBACK = "http://127.0.0.1:11434"


class FakeProvider:
    def __init__(self, name, *, base_url=None, default_model="m", text="ok", fail=None):
        self.name = name
        if base_url is not None:
            self.base_url = base_url
        self.default_model = default_model
        self.text = text
        self.fail = fail
        self.calls = 0
        self.streams = 0

    def generate(self, request):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return SommResponse(
            text=self.text, model=request.model or self.default_model,
            tokens_in=1, tokens_out=1, latency_ms=1,
        )

    def stream(self, request):
        self.streams += 1
        yield SommChunk(text=self.text)
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _cfg(tmp_path: Path, project: str = "boundary-test") -> Config:
    cfg = Config()
    cfg.project = project
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _llm(tmp_path, *providers) -> SommLLM:
    return SommLLM(providers=list(providers), config=_cfg(tmp_path), on_error=lambda _: None)


def _owner(text: str) -> Path:
    path = Path(os.environ["SOMM_BOUNDARY_PATH"])
    path.write_text(text, encoding="utf-8")
    # The policy cache keys on mtime+size; make a rewrite in the same tick visible.
    stamp = time.time() + 1
    os.utime(path, (stamp, stamp))
    return path


def _calls(llm: SommLLM) -> list[dict]:
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY ts")]


def _private(llm: SommLLM, name: str = "secret"):
    return llm.register_workload(name=name, privacy_class=PrivacyClass.PRIVATE)


# ---------------------------------------------------------------------------
# The scale


def test_tiers_order_by_strictness_not_spelling():
    assert DataTier.LOCAL < DataTier.ZDR < DataTier.SUBSCRIBED < DataTier.UNVETTED
    assert sorted(["unvetted", "local", "subscribed", "zdr"], key=parse_tier) == [
        "local", "zdr", "subscribed", "unvetted",
    ]
    assert admit("local", "zdr") and admit("zdr", "zdr")
    assert not admit("subscribed", "zdr")
    with pytest.raises(ValueError, match="local, zdr, subscribed, unvetted"):
        parse_tier("private")


# ---------------------------------------------------------------------------
# Classifier: shipped defaults


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "tier"),
    [
        ("ollama", "qwen3:14b", "http://localhost:11434", "local"),
        ("ollama", "qwen3:14b", "http://10.0.0.5:11434", "local"),
        ("ollama", "qwen3:14b", "http://192.168.1.9:11434", "local"),
        ("ollama", "qwen3:14b", "http://[::1]:11434", "local"),
        ("ollama", "gemma4:cloud", "http://localhost:11434", "unvetted"),
        ("ollama", "qwen3:14b", "http://gpu1.example.com:11434", "unvetted"),
        ("ollama", "qwen3:14b", None, "unvetted"),
        ("ollama", "qwen3-coder:480b-cloud", "http://127.0.0.1:11434", "unvetted"),
        ("ollama", "gpt-oss:120b-cloud", "http://[::1]:11434", "unvetted"),
        ("ollama", "qwen3:14b", "https://ollama.com", "unvetted"),
        ("openai", "gpt-x", "http://127.0.0.1:8000/v1", "local"),
        ("openai", "gpt-x", "https://api.openai.com/v1", "subscribed"),
        ("openai", "gpt-x", "https://us.api.openai.com/v1", "subscribed"),
        ("openai", "gpt-x", "https://my-gateway.example.com/v1", "unvetted"),
        ("codex-cli", "", None, "subscribed"),
        ("chatgpt", "gpt-x", None, "subscribed"),
        ("gemini", "gemini-x", "https://generativelanguage.googleapis.com/v1beta/openai", "unvetted"),
        ("openrouter", "meta/llama:free", None, "unvetted"),
        ("anthropic", "claude-x", None, "subscribed"),
        ("anthropic", "claude-x", "https://api.anthropic.com", "subscribed"),
        ("anthropic", "claude-x", "https://gateway.example.com", "unvetted"),
        ("claude-cli", "", None, "subscribed"),
        ("perplexity", "sonar", "https://api.perplexity.ai", "unvetted"),
        ("deepseek", "deepseek-chat", "https://api.deepseek.com/v1", "unvetted"),
        ("minimax", "MiniMax-M3", None, "unvetted"),
        ("openrouter", "anthropic/claude-x", None, "unvetted"),
        ("gemini", "gemini-x", None, "unvetted"),
        ("acme-plugin", "anything", None, "unvetted"),
        ("", "", None, "unvetted"),
    ],
)
def test_shipped_defaults(provider, model, base_url, tier):
    route = classify(provider, model, base_url=base_url)
    assert route.tier.value == tier
    assert route.basis
    assert route.origin in ("default", "assumed")


def test_loopback_ollama_is_not_local_for_cloud_models():
    """A loopback daemon offloads ':cloud' / '-cloud' tags to Ollama's servers."""
    for model in ("gemma4:cloud", "qwen3-coder:480b-cloud"):
        route = classify("ollama", model, base_url="http://localhost:11434")
        assert route.tier is DataTier.UNVETTED, model
        assert route.rule.startswith("ollama/*")
    assert classify("ollama", "qwen3:14b", base_url="http://localhost:11434").tier is DataTier.LOCAL


def test_seats_are_subscribed_but_assumed():
    for seat in ("claude-cli", "codex-cli", "chatgpt"):
        route = classify(seat, "")
        assert (route.tier, route.origin) == (DataTier.SUBSCRIBED, "assumed"), seat
    assert classify("anthropic", "claude-x").origin == "default"
    # an owner declaration replaces the assumption
    _owner('[[route]]\nmatch = "claude-cli"\ntier = "unvetted"\nnote = "training toggle on"\n')
    assert classify("claude-cli", "").origin == "owner"
    # and an owner file cannot mark its own rules "assumed"
    _owner('[[route]]\nmatch = "x"\ntier = "local"\nassumed = true\n')
    assert classify("x", "").origin == "owner"


def test_gemini_needs_an_owner_declaration_to_be_subscribed():
    assert classify("gemini", "gemini-x").tier is DataTier.UNVETTED
    _owner('[[route]]\nmatch = "gemini"\ntier = "subscribed"\nnote = "billing-enabled project"\n')
    route = classify("gemini", "gemini-x")
    assert (route.tier, route.origin) == (DataTier.SUBSCRIBED, "owner")


def test_nothing_ships_as_zdr_and_vendor_facts_are_sourced():
    policy = load_policy()
    assert policy.default_rules
    for rule in policy.default_rules:
        assert rule.tier is not DataTier.ZDR, rule.match
        assert rule.basis, rule.match
        if rule.tier is not DataTier.UNVETTED:
            # Anything that promotes a route states a vendor fact; it must cite.
            assert rule.source.startswith("https://"), rule.match
            assert rule.last_verified and rule.quote, rule.match
            assert rule.confidence == "high", rule.match  # never promote on med/low
        if rule.source:
            assert rule.last_verified and rule.confidence, rule.match
    gemini = [r for r in policy.default_rules if r.match == "gemini"]
    assert gemini and gemini[0].tier is DataTier.UNVETTED


def test_unknown_handler_is_unvetted_and_declared_handler_is_honoured():
    assert classify("triage", kind="handler").tier is DataTier.UNVETTED
    assert classify("triage", kind="handler", declared_tier="local").tier is DataTier.LOCAL
    assert classify("triage", kind="handler", declared_tier="bogus").tier is DataTier.UNVETTED


# ---------------------------------------------------------------------------
# Classifier: owner overrides


def test_owner_file_promotes_demotes_and_wins():
    _owner(
        """
[[route]]
match = "openai"
tier = "subscribed"
note = "API account, org data controls checked"

[[route]]
match = "openrouter/anthropic/*"
tier = "zdr"
note = "account-level ZDR on"

[[route]]
match = "openrouter/anthropic/claude-exact"
tier = "subscribed"

[[route]]
match = "ollama"
tier = "unvetted"
note = "this ollama forwards to a shared box"

[[route]]
host = "gpu1.example.ts.net"
tier = "local"
note = "my rented box"

[[route]]
match = "deepseek"
tier = "nonsense"
"""
    )
    openai = classify("openai", "gpt-x", base_url="https://api.openai.com/v1")
    assert (openai.tier, openai.origin) == (DataTier.SUBSCRIBED, "owner")
    assert openai.basis == "API account, org data controls checked"
    assert classify("openrouter", "anthropic/claude-y").tier is DataTier.ZDR
    # exact model beats a glob
    assert classify("openrouter", "anthropic/claude-exact").tier is DataTier.SUBSCRIBED
    assert classify("openrouter", "meta/llama:free").tier is DataTier.UNVETTED
    # owner beats the loopback rule
    assert classify("ollama", "q", base_url=LOOPBACK).tier is DataTier.UNVETTED
    # host rule, any provider
    assert classify("vllm", "m", base_url="http://gpu1.example.ts.net:8000").tier is DataTier.LOCAL
    # a rule with a bad tier fails closed rather than vanishing
    assert classify("deepseek", "x").tier is DataTier.UNVETTED


def test_unreadable_owner_file_fails_closed():
    _owner('[[route]]\nmatch = "deepseek"\ntier = "subscribed"\n[[[ broken')
    policy = load_policy()
    assert policy.owner_error
    assert classify("deepseek", "x").tier is DataTier.UNVETTED  # promotion ignored
    tier, source = policy.project_default("anything")
    assert tier is DataTier.LOCAL and "unreadable" in source


def test_owner_file_change_is_picked_up():
    _owner('[[route]]\nmatch = "deepseek"\ntier = "subscribed"\n')
    assert classify("deepseek", "x").tier is DataTier.SUBSCRIBED
    _owner('[[route]]\nmatch = "deepseek"\ntier = "unvetted"\nnote = "changed my mind"\n')
    assert classify("deepseek", "x").tier is DataTier.UNVETTED


# ---------------------------------------------------------------------------
# Resolution order


class _WL:
    def __init__(self, privacy="internal", boundary=None):
        self.privacy_class = privacy
        self.data_boundary = boundary


def test_resolution_order(monkeypatch):
    assert resolve_boundary(_WL()).tier is DataTier.UNVETTED
    assert resolve_boundary(_WL()).is_default
    _owner('[defaults]\nboundary = "subscribed"\n[projects.fab]\nboundary = "zdr"\n')
    assert resolve_boundary(_WL(), project="other").tier is DataTier.SUBSCRIBED
    assert resolve_boundary(_WL(), project="fab").tier is DataTier.ZDR
    monkeypatch.setenv("SOMM_DATA_BOUNDARY", "local")
    r = resolve_boundary(_WL(), project="fab")
    assert (r.tier, r.source) == (DataTier.LOCAL, "SOMM_DATA_BOUNDARY")
    monkeypatch.setenv("SOMM_DATA_BOUNDARY", "loose")
    assert resolve_boundary(_WL(), project="fab").tier is DataTier.LOCAL  # invalid → strictest
    monkeypatch.delenv("SOMM_DATA_BOUNDARY")
    # privacy_class=private derives local, and caps a looser stored value
    assert resolve_boundary(_WL("private")).tier is DataTier.LOCAL
    assert resolve_boundary(_WL("private", "unvetted")).tier is DataTier.LOCAL
    # a stored boundary beats the project default
    r = resolve_boundary(_WL(boundary="unvetted"), project="fab")
    assert (r.tier, r.source) == (DataTier.UNVETTED, "workload")


def test_per_call_may_only_tighten():
    wl = _WL(boundary="subscribed")
    assert resolve_boundary(wl, per_call="local").tier is DataTier.LOCAL
    loosened = resolve_boundary(wl, per_call="unvetted")
    assert (loosened.tier, loosened.source) == (DataTier.SUBSCRIBED, "workload")
    assert resolve_boundary(_WL("private"), per_call="unvetted").tier is DataTier.LOCAL


# ---------------------------------------------------------------------------
# Workload storage


def test_data_boundary_does_not_change_workload_id_and_is_stored(tmp_path):
    repo = Repository(_cfg(tmp_path).db_path)
    plain = repo.register_workload(name="w", project="p")
    bounded = repo.register_workload(name="w", project="p", data_boundary="subscribed")
    assert plain.id == bounded.id
    assert repo.workload_by_name("w", "p").data_boundary == "subscribed"
    # re-registering without a boundary leaves the stored one alone
    assert repo.register_workload(name="w", project="p").data_boundary == "subscribed"
    repo.set_workload_data_boundary(plain.id, "local", created_by="test")
    assert repo.workload_by_name("w", "p").data_boundary == "local"
    assert repo.current_workload_revision(plain.id)["data_boundary"] == "local"
    repo.set_workload_data_boundary(plain.id, None)
    assert repo.workload_by_name("w", "p").data_boundary is None
    with pytest.raises(ValueError):
        repo.set_workload_data_boundary(plain.id, "secret")
    with pytest.raises(ValueError):
        repo.register_workload(name="w2", project="p", data_boundary="nope")


# ---------------------------------------------------------------------------
# Enforcement: the audit probe, every call path


def test_private_generate_never_reaches_cloud_sync(tmp_path):
    openai = FakeProvider("openai")
    llm = _llm(tmp_path, openai)
    _private(llm)
    with pytest.raises(SommPrivacyViolation) as info:
        llm.generate("hi", workload="secret")
    assert openai.calls == 0
    msg = str(info.value)
    assert msg.startswith("SOMM_PRIVACY_VIOLATION")
    assert "'local'" in msg and "openai/m (subscribed)" in msg
    assert info.value.data_boundary == "local"
    assert info.value.refused[0]["tier"] == "subscribed"
    desc = describe_error(info.value)
    assert desc["code"] == "SOMM_PRIVACY_VIOLATION" and desc["retryable"] is False
    assert desc["refused"] and desc["data_boundary"] == "local"
    rows = _calls(llm)
    assert len(rows) == 1
    row = rows[0]
    assert (row["outcome"], row["error_kind"], row["provider"]) == (
        "unknown", "SommPrivacyViolation", "",
    )
    assert row["error_detail"].startswith("refused: data_boundary=local")
    assert row["data_tier"] is None and row["budget_eligible"] == 0 and row["cost_usd"] == 0
    llm.close()


def test_private_generate_never_reaches_cloud_async(tmp_path):
    openai = FakeProvider("openai")
    llm = _llm(tmp_path, openai)
    _private(llm)
    with pytest.raises(SommPrivacyViolation):
        asyncio.run(llm.agenerate("hi", workload="secret"))
    assert openai.calls == 0
    llm.close()


def test_private_stream_never_reaches_cloud(tmp_path):
    openai = FakeProvider("openai")
    llm = _llm(tmp_path, openai)
    _private(llm)
    with pytest.raises(SommPrivacyViolation):
        list(llm.stream("hi", workload="secret"))
    assert openai.streams == 0

    async def consume():
        return [p async for p in llm.astream("hi", workload="secret")]

    with pytest.raises(SommPrivacyViolation):
        asyncio.run(consume())
    assert openai.streams == 0
    llm.close()


def test_private_stream_uses_only_the_local_candidate(tmp_path):
    openai = FakeProvider("openai", text="cloud")
    local = FakeProvider("ollama", base_url=LOOPBACK, text="home")
    llm = _llm(tmp_path, openai, local)  # cloud first in the chain
    _private(llm)
    assert "".join(llm.stream("hi", workload="secret")) == "home"
    assert openai.streams == 0
    assert _calls(llm)[-1]["data_tier"] == "local"
    llm.close()


def test_pin_does_not_override_a_boundary(tmp_path):
    openai = FakeProvider("openai")
    local = FakeProvider("ollama", base_url=LOOPBACK)
    llm = _llm(tmp_path, local, openai)
    _private(llm)
    with pytest.raises(SommPrivacyViolation, match="pinned provider 'openai'"):
        llm.generate("hi", workload="secret", provider="openai", model="gpt-x")
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="secret", provider="openai", allow_fallback=True)
    with pytest.raises(SommPrivacyViolation):
        list(llm.stream("hi", workload="secret", provider="openai"))
    assert openai.calls == 0 and openai.streams == 0 and local.calls == 0
    llm.close()


def test_router_filters_inadmissible_candidates_and_records_tier(tmp_path):
    openai = FakeProvider("openai", text="cloud")
    local = FakeProvider("ollama", base_url=LOOPBACK, text="home")
    llm = _llm(tmp_path, openai, local)
    _private(llm)
    result = llm.generate("hi", workload="secret")
    assert (result.text, result.provider) == ("home", "ollama")
    assert openai.calls == 0
    assert _calls(llm)[-1]["data_tier"] == "local"
    llm.close()


def test_no_fall_through_to_a_looser_route(tmp_path):
    local = FakeProvider("ollama", base_url=LOOPBACK, fail=SommTransientError("down", 30))
    openai = FakeProvider("openai")
    llm = _llm(tmp_path, local, openai)
    _private(llm)
    result = llm.generate("hi", workload="secret")
    assert result.outcome != Outcome.OK
    assert local.calls == 1 and openai.calls == 0
    llm.close()


def test_router_refuses_directly(tmp_path):
    repo = Repository(_cfg(tmp_path).db_path)
    openai = FakeProvider("openai")
    router = Router([openai], ProviderHealthTracker(repo))
    with pytest.raises(SommPrivacyViolation):
        router.dispatch(SommRequest(prompt="hi", data_boundary="zdr"))
    assert openai.calls == 0
    # no boundary: unchanged behaviour
    assert router.dispatch(SommRequest(prompt="hi")).provider == "openai"


def test_workload_policy_chain_is_filtered(tmp_path):
    openai = FakeProvider("openai")
    local = FakeProvider("ollama", base_url=LOOPBACK)
    llm = _llm(tmp_path, openai, local)
    _private(llm)
    llm.set_workload_policy("secret", {"fallback": [{"provider": "openai"}]})
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="secret")
    assert openai.calls == 0
    llm.close()


def test_per_call_boundary_tightens_an_unbounded_workload(tmp_path):
    anthropic = FakeProvider("anthropic")
    llm = _llm(tmp_path, anthropic)
    assert llm.generate("hi", workload="w", data_boundary="subscribed").outcome == Outcome.OK
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="w", data_boundary="zdr")
    assert anthropic.calls == 1
    llm.close()


def test_stored_boundary_admits_within_and_refuses_beyond(tmp_path):
    anthropic = FakeProvider("anthropic", text="a")
    deepseek = FakeProvider("deepseek", text="d")
    llm = _llm(tmp_path, deepseek, anthropic)
    llm.register_workload(name="w", data_boundary="subscribed")
    result = llm.generate("hi", workload="w")
    assert result.provider == "anthropic" and deepseek.calls == 0
    assert _calls(llm)[-1]["data_tier"] == "subscribed"
    llm.close()


def test_upgrade_safety_no_boundary_means_no_change(tmp_path):
    """A workload with no boundary, no private class, no default: every route
    stays admissible, exactly as before 0.21. The tier is still recorded."""
    unknown = FakeProvider("acme")
    llm = _llm(tmp_path, unknown)
    assert llm.generate("hi", workload="plain").text == "ok"
    assert "".join(llm.stream("hi", workload="plain")) == "ok"
    rows = _calls(llm)
    assert [r["outcome"] for r in rows] == ["ok", "ok"]
    assert {r["data_tier"] for r in rows} == {"unvetted"}
    llm.close()


def test_openrouter_roster_walk_filters_per_model(tmp_path):
    from somm.providers.openrouter import OpenRouterProvider

    _owner('[[route]]\nmatch = "openrouter/paid/model"\ntier = "subscribed"\n')

    class Recording(OpenRouterProvider):
        tried: list[str] = []

        def _call_single(self, request, model):
            self.tried.append(model)
            return SommResponse(text="ok", model=model, tokens_in=1, tokens_out=1, latency_ms=1)

    provider = Recording(api_key="k", roster=["free/model:free", "paid/model"])
    resp = provider.generate(SommRequest(prompt="hi", data_boundary="subscribed"))
    assert resp.model == "paid/model" and provider.tried == ["paid/model"]
    with pytest.raises(SommPrivacyViolation):
        provider.generate(SommRequest(prompt="hi", data_boundary="local"))
    # Through the client: admitted on its best roster model, served by it.
    llm = _llm(tmp_path, provider)
    llm.register_workload(name="w", data_boundary="subscribed")
    assert llm.generate("hi", workload="w").model == "paid/model"
    assert _calls(llm)[-1]["data_tier"] == "subscribed"
    llm.close()


# ---------------------------------------------------------------------------
# Workload handlers


def test_undeclared_handler_is_not_consulted_inside_a_boundary(tmp_path):
    from somm import workloads
    from somm.workloads import WorkloadResult

    class Undeclared:
        kind = "remote-kind"
        served = 0

        def serve(self, request):
            Undeclared.served += 1
            return WorkloadResult(text="from handler")

    class Local:
        kind = "local-kind"
        data_tier = "local"

        def serve(self, request):
            return WorkloadResult(text="from local handler")

    workloads.clear()
    try:
        workloads.register(Undeclared())
        workloads.register(Local())
        local = FakeProvider("ollama", base_url=LOOPBACK, text="chain")
        llm = _llm(tmp_path, local)
        _private(llm, "secret")
        _private(llm, "secret2")
        workloads.bind("secret", "remote-kind")
        workloads.bind("secret2", "local-kind")
        assert llm.generate("hi", workload="secret").text == "chain"
        assert Undeclared.served == 0
        assert llm.generate("hi", workload="secret2").text == "from local handler"
        assert _calls(llm)[-1]["data_tier"] == "local"
        # Unbounded workloads keep today's behaviour.
        workloads.bind("open", "remote-kind")
        assert llm.generate("hi", workload="open").text == "from handler"
        llm.close()
    finally:
        workloads.clear()


# ---------------------------------------------------------------------------
# Shadow eval and judges


def test_enable_shadow_requires_an_admissible_gold_route(tmp_path):
    local = FakeProvider("ollama", base_url=LOOPBACK)
    openai = FakeProvider("openai")
    llm = _llm(tmp_path, local, openai)
    _private(llm)
    with pytest.raises(SommPrivacyViolation, match="privacy_class=private"):
        llm.enable_shadow("secret", "ollama", "q")  # private stays banned outright
    llm.register_workload(name="bounded", data_boundary="zdr")
    with pytest.raises(SommPrivacyViolation, match="gold route"):
        llm.enable_shadow("bounded", "openai", "gpt-x")
    llm.enable_shadow("bounded", "ollama", "q")  # local is within zdr
    llm.enable_shadow("unbounded", "openai", "gpt-x")  # no boundary: unchanged
    llm.close()


def _shadow_candidate(repo: Repository, project: str, wl, provider="ollama") -> str:
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id, ts=datetime.now(UTC), project=project, workload_id=wl.id,
            prompt_id=None, provider=provider, model="q", tokens_in=1, tokens_out=1,
            latency_ms=1, cost_usd=0.0, outcome=Outcome.OK, error_kind=None,
            prompt_hash="p", response_hash="r",
        )
    )
    repo.write_sample(call_id, "prompt body", "response body")
    return call_id


def test_shadow_worker_refuses_gold_outside_boundary_and_counts_private(tmp_path):
    from somm.workers.shadow_eval import ShadowEvalWorker

    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    bounded = repo.register_workload(name="bounded", project=cfg.project, data_boundary="local")
    repo.set_shadow_config(
        bounded.id, {"gold_provider": "openai", "gold_model": "gpt-x", "sample_rate": 1.0}
    )
    _shadow_candidate(repo, cfg.project, bounded)
    openai = FakeProvider("openai")
    worker = ShadowEvalWorker(repo, [openai])
    summary = worker.run_once()
    assert summary["boundary_skipped"] == 1 and summary["calls_graded"] == 0
    assert openai.calls == 0


def test_judge_outside_boundary_is_refused_not_called(tmp_path):
    from somm.workers.shadow_eval import ShadowEvalWorker
    from somm_core.boundary import ResolvedBoundary

    repo = Repository(_cfg(tmp_path).db_path)
    judge = FakeProvider("openai", text='{"criteria": []}')
    worker = ShadowEvalWorker(repo, [judge])
    worker._boundary = ResolvedBoundary(DataTier.LOCAL, "workload")
    score, reason, _ = worker._run_judge(
        original_prompt="p", production_text="a", gold_text="b",
        judge_cfg={"provider": "openai", "model": "gpt-x", "criteria": ["correct"]},
        call_row={"call_id": "c", "project": "p", "workload_id": "w"},
        eval_result_id=1,
    )
    assert judge.calls == 0 and score is None
    assert "refused" in reason["judges"][0]["error"]


# ---------------------------------------------------------------------------
# Harness attempts


def test_harness_attempt_outside_boundary_is_refused_before_launch(tmp_path, monkeypatch):
    from somm import harnesses
    from somm.harnesses import HarnessRequest

    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    launched = []
    monkeypatch.setattr(
        harnesses.get("claude-cli"), "start", lambda request: launched.append(request)
    )
    request = HarnessRequest(prompt="do it", cwd=tmp_path, capture_dir=tmp_path / "cap")
    with pytest.raises(SommPrivacyViolation, match="harness 'claude-cli'"):
        harnesses.check_boundary(
            "claude-cli", request, workload="secret", config=cfg, repo=repo
        )
    with pytest.raises(SommPrivacyViolation):
        harnesses.start("claude-cli", request, data_boundary="local")
    assert launched == []
    with repo._open() as conn:
        row = conn.execute("SELECT error_kind, origin FROM calls").fetchone()
    assert row == ("SommPrivacyViolation", "harness")
    # subscribed admits a Claude Code seat
    harnesses.start("claude-cli", request, data_boundary="subscribed")
    assert len(launched) == 1


# ---------------------------------------------------------------------------
# CLI compare / bench / eval go through generate's checks


def test_cli_compare_reports_refused(tmp_path, monkeypatch, capsys):
    from somm import cli

    openai = FakeProvider("openai")
    local = FakeProvider("ollama", base_url=LOOPBACK)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "cmp")
    monkeypatch.setattr(SommLLM, "_default_providers", lambda self: [local, openai])
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    repo.register_workload(name="compare", project="cmp", data_boundary="local")
    rc = cli.main(["compare", "hi", "--models", "openai/gpt-x,ollama/q", "--project", "cmp"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" in out and "subscribed" in out
    assert openai.calls == 0 and local.calls == 1


def test_cli_bench_and_eval_refuse_through_generate(tmp_path):
    openai = FakeProvider("openai")
    llm = _llm(tmp_path, openai)
    _private(llm, "bench_latency")
    # bench and eval call llm.generate(provider=..., model=...): the pin check applies.
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="bench_latency", provider="openai", model="gpt-x")
    assert openai.calls == 0
    llm.close()


# ---------------------------------------------------------------------------
# Migration on an existing ledger


def test_migration_0027_on_an_existing_ledger(tmp_path):
    path = tmp_path / "v26.sqlite"
    with sqlite3.connect(path) as conn:
        for version, migration in _list_migrations():
            if version > 26:
                continue
            conn.executescript(migration.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()
        conn.execute(
            "INSERT INTO workloads (id, name, project, privacy_class) "
            "VALUES ('w1', 'old', 'p', 'private')"
        )
        conn.execute(
            "INSERT INTO calls (id, ts, project, workload_id, provider, model, tokens_in, "
            "tokens_out, latency_ms, cost_usd, outcome, prompt_hash, response_hash) "
            "VALUES ('c1', '2026-11-01', 'p', 'w1', 'openai', 'gpt', 1, 2, 3, 0.0, 'ok', 'p', 'r')"
        )
        conn.commit()
        assert ensure_schema(conn) == SCHEMA_VERSION
        assert conn.execute("SELECT data_boundary FROM workloads").fetchall() == [(None,)]
        assert conn.execute("SELECT id, data_tier FROM calls").fetchall() == [("c1", None)]
        assert ensure_schema(conn) == SCHEMA_VERSION
    repo = Repository(path)
    wl = repo.workload_by_name("old", "p")
    assert wl.data_boundary is None and wl.privacy_class is PrivacyClass.PRIVATE
    assert repo.get_call("c1").data_tier is None


# ---------------------------------------------------------------------------
# Audit on historic rows


def _historic(repo, project, wl, provider, model, *, tier=None, error_kind=None):
    repo.write_call(
        Call(
            id=str(uuid.uuid4()), ts=datetime.now(UTC), project=project, workload_id=wl.id,
            prompt_id=None, provider=provider, model=model, tokens_in=1, tokens_out=1,
            latency_ms=1, cost_usd=0.0,
            outcome=Outcome.OK if error_kind is None else Outcome.UNKNOWN,
            error_kind=error_kind, prompt_hash="p", response_hash="r", data_tier=tier,
        )
    )


def test_audit_classifies_historic_rows_and_flags_violations(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path, project="aud")
    repo = Repository(cfg.db_path)
    secret = repo.register_workload(name="secret", project="aud", privacy_class=PrivacyClass.PRIVATE)
    fine = repo.register_workload(name="fine", project="aud", data_boundary="local")
    open_wl = repo.register_workload(name="open", project="aud")
    _historic(repo, "aud", secret, "openai", "gpt-x")  # pre-enforcement leak
    _historic(repo, "aud", secret, "ollama", "q")
    _historic(repo, "aud", secret, "", "", error_kind="SommPrivacyViolation")
    _historic(repo, "aud", fine, "ollama", "q")
    _historic(repo, "aud", open_wl, "deepseek", "chat", tier="unvetted")
    seat = repo.register_workload(name="seat", project="aud", data_boundary="subscribed")
    _historic(repo, "aud", seat, "claude-cli", "")
    with repo._open() as conn:
        rows = {r["workload"]: r for r in audit(
            conn, "aud", since_days=30, base_urls={"ollama": "http://localhost:11434"}
        )}
    assert rows["secret"]["violation"] and rows["secret"]["worst_tier"] == "subscribed"
    assert rows["secret"]["refusals"] == 1 and rows["secret"]["calls"] == 2
    assert not rows["secret"]["routes"][0]["recorded"]
    assert rows["fine"]["worst_tier"] == "local" and not rows["fine"]["violation"]
    assert rows["open"]["effective_boundary"] == "unvetted" and not rows["open"]["violation"]
    assert rows["open"]["routes"][0]["recorded"]
    assert rows["seat"]["assumed"] and not rows["seat"]["violation"]
    assert rows["seat"]["routes"][0]["origin"] == "assumed"

    from somm import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "aud")
    monkeypatch.setenv("SOMM_OLLAMA_URL", "http://localhost:11434")
    rc = cli.main(["boundary", "audit", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1 and payload["violations"] == 1
    rc = cli.main(["boundary", "set", "open", "local"])
    assert rc == 0 and repo.workload_by_name("open", "aud").data_boundary == "local"
    rc = cli.main(["boundary", "routes", "--json"])
    routes = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert any(r["match"] == "anthropic" for r in routes["rules"])


def test_doctor_has_a_boundary_section(tmp_path, monkeypatch, capsys):
    from somm import cli

    cfg = _cfg(tmp_path)
    Repository(cfg.db_path).register_workload(name="w", project=cfg.project, data_boundary="zdr")
    monkeypatch.setattr("somm.cli.load_config", lambda project=None: cfg)
    monkeypatch.setattr(
        "somm.cli.OllamaProvider.health", lambda self: ProviderHealth(available=True, detail="ok")
    )
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "boundary:" in out and "workloads with a boundary: 1/1" in out

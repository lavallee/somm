"""OpenRouter tiers from live evidence: the ZDR endpoint list + provider HQs.

No network: the two lists below are trimmed copies of the real responses
(2026-11-09), and the repo conftest makes any unpatched fetch fail.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
import pytest
from somm import SommLLM
from somm.errors import SommPrivacyViolation
from somm.providers import openrouter_evidence as ev
from somm.providers.base import SommRequest
from somm.providers.openrouter import OpenRouterProvider
from somm_core.boundary import load_policy
from somm_core.config import Config

PAID = "google/gemma-4-31b-it"
FREE = "google/gemma-4-31b-it:free"
GEMINI = "google/gemini-3.8-flash"

# GET /api/v1/endpoints/zdr, trimmed to the fields somm keeps.
ZDR = [
    {"model_id": PAID, "tag": "modal", "provider_name": "Modal", "safety_retention": False},
    {"model_id": PAID, "tag": "deepinfra/fp8", "provider_name": "DeepInfra", "safety_retention": False},
    {"model_id": PAID, "tag": "siliconflow/fp8", "provider_name": "SiliconFlow", "safety_retention": False},
    {"model_id": PAID, "tag": "nextbit/fp4", "provider_name": "NextBit", "safety_retention": False},
    {"model_id": PAID, "tag": "primeintellect/fp4", "provider_name": "PrimeIntellect",
     "safety_retention": False},
    {"model_id": GEMINI, "tag": "google-vertex/us", "provider_name": "Google Vertex",
     "safety_retention": True},
    {"model_id": GEMINI, "tag": "google-vertex/global", "provider_name": "Google Vertex",
     "safety_retention": True},
]
# GET /api/v1/providers, trimmed.
PROVIDERS = [
    {"slug": "modal", "name": "Modal", "headquarters": "US"},
    {"slug": "deepinfra", "name": "DeepInfra", "headquarters": "US"},
    {"slug": "siliconflow", "name": "SiliconFlow", "headquarters": "SG"},
    {"slug": "nextbit", "name": "NextBit", "headquarters": "ES"},
    {"slug": "primeintellect", "name": "PrimeIntellect", "headquarters": None},
    {"slug": "google-vertex", "name": "Google Vertex", "headquarters": "US"},
    {"slug": "google-ai-studio", "name": "Google AI Studio", "headquarters": "US"},
]
US_ONLY = ["deepinfra/fp8", "modal"]


def _owner(text: str) -> None:
    path = Path(os.environ["SOMM_BOUNDARY_PATH"])
    path.write_text(text, encoding="utf-8")
    stamp = time.time() + 1
    os.utime(path, (stamp, stamp))


@pytest.fixture
def fetches(monkeypatch):
    """Serve the fixture lists as a successful fetch; count the fetches."""
    count = {"n": 0}

    def fetch():
        count["n"] += 1
        return {"zdr": [dict(e) for e in ZDR], "providers": [dict(p) for p in PROVIDERS]}

    monkeypatch.setattr(ev, "_fetch", fetch)
    return count


def _mock_openrouter(monkeypatch, bodies: list[dict], *, served_by: str | None = "DeepInfra"):
    """Patch httpx.Client so chat completions are answered locally; record bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        body = {
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
        if served_by:
            body["provider"] = served_by
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)

    class _Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", _Client)


def _llm(tmp_path, provider) -> SommLLM:
    cfg = Config()
    cfg.project = "evidence"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(providers=[provider], config=cfg, on_error=lambda _: None)


def _last_call(llm: SommLLM) -> dict:
    llm._writer.flush(timeout=5.0)
    with llm.repo._open() as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM calls ORDER BY ts DESC LIMIT 1").fetchone())


# ---------------------------------------------------------------------------
# The pure function


def test_qualifying_upstreams_rules():
    q = ev.qualifying_upstreams
    # Exact model id: the paid model has US ZDR endpoints; foreign (SG, ES)
    # and unknown (null) headquarters are excluded.
    assert q(PAID, ZDR, PROVIDERS, ["US"]) == US_ONLY
    # :free is its own id with its own endpoints; none of them is on the list.
    assert q(FREE, ZDR, PROVIDERS, ["US"]) == []
    # A dated or different id is not the same model.
    assert q(PAID + "-20260402", ZDR, PROVIDERS, ["US"]) == []
    # On the ZDR list but retained for safety review: not strict ZDR.
    assert q(GEMINI, ZDR, PROVIDERS, ["US"]) == []
    # Widening the allowed list admits SG; null headquarters never qualifies.
    assert q(PAID, ZDR, PROVIDERS, ["us", "SG"]) == ["deepinfra/fp8", "modal", "siliconflow/fp8"]
    assert q(PAID, ZDR, PROVIDERS, []) == []


def test_one_safety_retaining_endpoint_disqualifies_its_upstream():
    zdr = ZDR + [{"model_id": PAID, "tag": "deepinfra/turbo", "safety_retention": True}]
    assert ev.qualifying_upstreams(PAID, zdr, PROVIDERS, ["US"]) == ["modal"]


def test_zdr_headquarters_setting():
    assert load_policy().zdr_headquarters == ("US",)
    _owner('[defaults]\nzdr_headquarters = ["us", "SG"]\n')
    assert load_policy().zdr_headquarters == ("US", "SG")
    _owner('[defaults]\nzdr_headquarters = "US"\n')  # not a list: fails closed
    assert load_policy().zdr_headquarters == ()
    _owner("[defaults\nzdr_headquarters = [")  # unreadable file: fails closed
    assert load_policy().zdr_headquarters == ()


# ---------------------------------------------------------------------------
# Fetch and cache


def test_fetched_at_most_daily_and_cached_beside_boundary_toml(fetches):
    first = ev.load_evidence()
    assert first and first["zdr"] and fetches["n"] == 1
    assert ev.load_evidence() is first and fetches["n"] == 1
    assert ev.cache_path().parent == Path(os.environ["SOMM_BOUNDARY_PATH"]).parent
    # A new process finds the fresh copy on disk and does not fetch.
    ev._memo.update(evidence=None, next_check=0.0)
    assert ev.load_evidence()["zdr"] == first["zdr"] and fetches["n"] == 1


def test_fetch_failure_uses_the_last_good_copy():
    stale = {"fetched_at": time.time() - 3 * 86400, "zdr": ZDR, "providers": PROVIDERS}
    ev.cache_path().write_text(json.dumps(stale), encoding="utf-8")
    got = ev.load_evidence()  # the conftest's fetch fails
    assert got is not None and got["zdr"] == ZDR
    provider = OpenRouterProvider(api_key="k")
    assert provider.zdr_only(PAID) == US_ONLY


def test_fetch_failure_with_no_copy_is_no_evidence():
    assert ev.load_evidence() is None
    assert OpenRouterProvider(api_key="k").zdr_only(PAID) == []


# ---------------------------------------------------------------------------
# On the call path


def test_default_boundary_leaves_the_request_unchanged_and_fetches_nothing(
    tmp_path, monkeypatch, fetches
):
    bodies: list[dict] = []
    _mock_openrouter(monkeypatch, bodies)
    llm = _llm(tmp_path, OpenRouterProvider(api_key="k", roster=[PAID]))
    assert llm.generate("hi", workload="plain").text == "pong"
    assert llm.generate("hi", workload="plain", provider="openrouter", model=PAID).text == "pong"
    assert fetches["n"] == 0 and not ev.cache_path().exists()
    for body in bodies:
        assert "provider" not in body
        assert set(body) == {"model", "messages", "temperature", "max_tokens", "stream", "usage"}
    row = _last_call(llm)
    # Every row carries its route; an unpinned-upstream call carries no
    # "upstream" record.
    assert row["data_tier"] == "unvetted"
    assert "upstream" not in json.loads(row["routing_json"])
    llm.close()


def test_strict_boundary_pins_the_request_and_records_zdr(tmp_path, monkeypatch, fetches):
    bodies: list[dict] = []
    _mock_openrouter(monkeypatch, bodies)
    llm = _llm(tmp_path, OpenRouterProvider(api_key="k", roster=[PAID]))
    llm.register_workload(name="w", data_boundary="zdr")
    assert llm.generate("hi", workload="w", provider="openrouter", model=PAID).text == "pong"
    assert bodies[-1]["provider"] == {
        "only": US_ONLY, "allow_fallbacks": False, "zdr": True, "data_collection": "deny",
    }
    row = _last_call(llm)
    assert row["data_tier"] == "zdr"
    assert json.loads(row["routing_json"])["upstream"] == {
        "data_tier": "zdr", "only": US_ONLY, "served_by": "DeepInfra",
    }
    # Routed (no pin) under a subscribed boundary: same pin, same tier.
    assert llm.generate("hi", workload="w", data_boundary="subscribed").text == "pong"
    assert bodies[-1]["provider"]["only"] == US_ONLY
    assert _last_call(llm)["data_tier"] == "zdr"
    assert fetches["n"] == 1
    llm.close()


def test_served_by_is_optional(tmp_path, monkeypatch, fetches):
    bodies: list[dict] = []
    _mock_openrouter(monkeypatch, bodies, served_by=None)
    resp = OpenRouterProvider(api_key="k").generate(
        SommRequest(prompt="hi", model=PAID, data_boundary="zdr")
    )
    assert resp.raw["somm_route"]["served_by"] is None and bodies[-1]["provider"]


def test_no_qualifying_upstream_is_refused_and_the_roster_moves_on(tmp_path, monkeypatch, fetches):
    bodies: list[dict] = []
    _mock_openrouter(monkeypatch, bodies)
    provider = OpenRouterProvider(api_key="k", roster=[FREE, PAID])
    resp = provider.generate(SommRequest(prompt="hi", data_boundary="zdr"))
    assert resp.model == PAID and [b["model"] for b in bodies] == [PAID]
    # Pinned to a model with no qualifying upstream: refused before sending.
    llm = _llm(tmp_path, provider)
    llm.register_workload(name="w", data_boundary="zdr")
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="w", provider="openrouter", model=FREE)
    # A roster of only such models is refused outright.
    with pytest.raises(SommPrivacyViolation):
        OpenRouterProvider(api_key="k", roster=[FREE, GEMINI]).generate(
            SommRequest(prompt="hi", data_boundary="zdr")
        )
    # `local` does not admit zdr: no promotion is even considered.
    with pytest.raises(SommPrivacyViolation):
        provider.generate(SommRequest(prompt="hi", model=PAID, data_boundary="local"))
    assert len(bodies) == 1
    llm.close()


def test_no_evidence_means_refused_not_an_error(tmp_path, monkeypatch):
    bodies: list[dict] = []
    _mock_openrouter(monkeypatch, bodies)
    llm = _llm(tmp_path, OpenRouterProvider(api_key="k", roster=[PAID]))
    llm.register_workload(name="w", data_boundary="zdr")
    with pytest.raises(SommPrivacyViolation):
        llm.generate("hi", workload="w")
    assert bodies == []
    llm.close()


# ---------------------------------------------------------------------------
# The evidence view


def test_cli_routes_shows_the_evidence(monkeypatch, capsys, fetches):
    from somm import cli

    endpoints = {"data": {"endpoints": [{"tag": t} for t in
                                        ("modal", "deepinfra/fp8", "deepinfra/turbo", "nextbit/fp4")]}}
    seen: list[str] = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return httpx.Response(200, json=endpoints, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert cli.main(["boundary", "routes", f"openrouter/{PAID}"]) == 0
    out = capsys.readouterr().out
    assert seen == [f"https://openrouter.ai/api/v1/models/{PAID}/endpoints"]
    assert "evidence fetched: 20" in out and "allowed headquarters: US" in out
    lines = {line.split()[0]: line.split()[1:] for line in out.splitlines() if line[:1].islower()}
    assert lines["modal"] == ["US", "yes", "no", "yes"]
    assert lines["deepinfra/turbo"] == ["US", "no", "-", "no"]  # not on the ZDR list
    assert lines["nextbit/fp4"] == ["ES", "yes", "no", "no"]
    assert lines["siliconflow/fp8"] == ["SG", "yes", "no", "no"]  # ZDR-listed, not in /endpoints
    assert "admitted as zdr, pinned to 2 endpoint(s)" in out
    assert cli.main(["boundary", "routes", f"openrouter/{FREE}", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["route"] == f"openrouter/{FREE}"
    assert not any(e["qualifies"] for e in payload["endpoints"])

"""Tests for the Perplexity provider — search-grounded sonar models.

Perplexity is OpenAI-compatible on the wire, so the base class handles
status classification and usage parsing. These tests pin the provider-
specific behaviour: the search-extra payload flags, the deep-research
quirks (no system message, no temperature), and citation passthrough on
SommResponse.raw.
"""

from __future__ import annotations

import httpx
from somm.providers.base import SommRequest
from somm.providers.perplexity import PerplexityProvider


def _patch_client(handler):
    transport = httpx.MockTransport(handler)

    class _MockedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    return _MockedClient


def _capture_handler(captured: dict, *, citations=None):
    def handler(request):
        import json
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "grounded answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                "citations": citations or ["https://a.example", "https://b.example"],
                "related_questions": ["follow up?"],
            },
        )

    return handler


def test_sonar_sets_search_extras_and_returns_text(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(httpx, "Client", _patch_client(_capture_handler(captured)))
    p = PerplexityProvider(api_key="pplx-fake")

    resp = p.generate(SommRequest(prompt="what's new in X?", system="be terse"))

    assert resp.text == "grounded answer"
    assert resp.model == "sonar"
    # Search-grounding flags are always set.
    assert captured["payload"]["return_citations"] is True
    assert captured["payload"]["return_related_questions"] is True
    # System message preserved for the regular sonar models.
    roles = [m["role"] for m in captured["payload"]["messages"]]
    assert roles == ["system", "user"]


def test_citations_survive_on_raw(monkeypatch):
    monkeypatch.setattr(
        httpx, "Client",
        _patch_client(_capture_handler({}, citations=["https://cited.example"])),
    )
    p = PerplexityProvider(api_key="pplx-fake")

    resp = p.generate(SommRequest(prompt="q"))

    # The whole body lands on raw — that's how callers read citations, since
    # SommResponse has no dedicated citation field.
    assert resp.raw is not None
    assert resp.raw["citations"] == ["https://cited.example"]
    assert resp.raw["related_questions"] == ["follow up?"]


def test_deep_research_strips_system_and_temperature(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(httpx, "Client", _patch_client(_capture_handler(captured)))
    p = PerplexityProvider(api_key="pplx-fake", default_model="sonar-deep-research")

    p.generate(SommRequest(prompt="deep q", system="ignored", temperature=0.7))

    payload = captured["payload"]
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["user"]  # system stripped
    assert "temperature" not in payload
    assert "max_tokens" not in payload

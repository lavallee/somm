"""A model the provider does not serve cools that (provider, model) slot.

Before the fix a 404 was a plain ``SommBadRequest``: fatal, never cooled. The
router aborted the whole chain on it, and OpenRouter's roster walk (which only
caught transient errors) did the same, so a withdrawn model was retried on
every call: one fleet ledger held 458 ``nemotron:free`` 404s over 18 days.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from somm.client import _PolicyProvider
from somm.errors import (
    SommBadRequest,
    SommModelNotFound,
    classify_body_error,
    classify_http_status,
)
from somm.providers.base import SommRequest, SommResponse
from somm.providers.openrouter import OpenRouterProvider
from somm.routing import ESCALATION_STEP, ProviderHealthTracker, Router
from somm_core.repository import Repository


def _tracker(tmp_path: Path) -> ProviderHealthTracker:
    return ProviderHealthTracker(Repository(tmp_path / "calls.sqlite"))


def _ok(text: str = "ok", model: str = "m") -> SommResponse:
    return SommResponse(text=text, model=model, tokens_in=1, tokens_out=1, latency_ms=1, raw=None)


class _Provider:
    """Serves some models, 404s on the rest; counts calls per model."""

    def __init__(self, name: str, default_model: str, serves: set[str]) -> None:
        self.name = name
        self.default_model = default_model
        self.serves = serves
        self.calls: list[str] = []

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model
        self.calls.append(model)
        if model not in self.serves:
            classify_http_status(404, '{"error":"gone"}', provider=self.name, model=model)
        return _ok(model=model)

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):  # pragma: no cover
        return None

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


# --- classification ---------------------------------------------------------


def test_404_is_a_model_not_found_bad_request_carrying_the_model():
    with pytest.raises(SommModelNotFound) as caught:
        classify_http_status(404, "no endpoints", provider="openrouter", model="x:free")
    assert isinstance(caught.value, SommBadRequest)  # pinned callers see what they did
    assert caught.value.model == "x:free"
    assert caught.value.cooldown_s > 0


@pytest.mark.parametrize(
    "body",
    [
        '{"error":{"message":"foo/bar is not a valid model ID","code":400}}',
        '{"error":{"code":"model_not_found","message":"The model `gpt-x` does not exist"}}',
    ],
)
def test_400_naming_an_unknown_model_is_model_not_found(body):
    with pytest.raises(SommModelNotFound):
        classify_http_status(400, body, provider="p", model="m")


def test_ordinary_400_stays_a_plain_bad_request():
    with pytest.raises(SommBadRequest) as caught:
        classify_http_status(400, '{"error":"max_tokens too large"}', provider="p", model="m")
    assert not isinstance(caught.value, SommModelNotFound)


def test_body_error_404_is_model_not_found():
    with pytest.raises(SommModelNotFound):
        classify_body_error(
            {"code": 404, "message": "No endpoints found for x:free."}, provider="p", model="x:free"
        )


# --- router -----------------------------------------------------------------


def test_router_cools_the_slot_and_continues_the_chain(tmp_path):
    tr = _tracker(tmp_path)
    a = _Provider("a", "withdrawn", serves=set())
    b = _Provider("b", "good", serves={"good"})
    router = Router([a, b], tr)

    first = router.dispatch(SommRequest(prompt="hi"))
    assert first.provider == "b"
    assert tr.get("a", "withdrawn").is_cooling()
    assert not tr.get("a").is_cooling()  # only the slot, not the provider

    second = router.dispatch(SommRequest(prompt="hi"))
    assert second.provider == "b"
    assert a.calls == ["withdrawn"]  # not retried while cooling


def test_model_not_found_cooldown_escalates(tmp_path):
    tr = _tracker(tmp_path)
    for _ in range(ESCALATION_STEP):
        tr.mark_failure("a", "withdrawn", cooldown_s=SommModelNotFound().cooldown_s)
    rec = tr.mark_failure("a", "withdrawn", cooldown_s=SommModelNotFound().cooldown_s)
    assert rec.consecutive_failures == ESCALATION_STEP + 1
    from datetime import UTC, datetime

    remaining = (rec.cooldown_until - datetime.now(UTC)).total_seconds()
    assert remaining > 1.9 * SommModelNotFound().cooldown_s


def test_policy_fallback_moves_to_another_model_of_the_same_provider(tmp_path):
    tr = _tracker(tmp_path)
    inner = _Provider("openrouter", "", serves={"model-b"})
    chain = [
        _PolicyProvider(inner, model="model-a", explicit_model=False, timeout_s=None),
        _PolicyProvider(inner, model="model-b", explicit_model=False, timeout_s=None),
    ]
    result = Router(chain, tr).dispatch(SommRequest(prompt="hi"))
    assert result.response.model == "model-b"
    assert inner.calls == ["model-a", "model-b"]
    assert tr.get("openrouter", "model-a").is_cooling()


def test_a_lone_missing_model_still_raises_the_not_found_error(tmp_path):
    tr = _tracker(tmp_path)
    a = _Provider("a", "typo", serves=set())
    with pytest.raises(SommModelNotFound, match="not found"):
        Router([a], tr).dispatch(SommRequest(prompt="hi"))


def test_success_on_a_slot_clears_its_failures(tmp_path):
    tr = _tracker(tmp_path)
    tr.mark_failure("a", "m", cooldown_s=0.0)
    a = _Provider("a", "m", serves={"m"})
    Router([a], tr).dispatch(SommRequest(prompt="hi"))
    assert tr.get("a", "m").consecutive_failures == 0


# --- OpenRouter roster -------------------------------------------------------


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    class _Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", _Client)


def test_openrouter_roster_cools_a_withdrawn_model_and_moves_on(tmp_path, monkeypatch):
    tr = _tracker(tmp_path)
    seen: list[str] = []

    def handler(request):
        model = json.loads(request.read())["model"]
        seen.append(model)
        if model == "gone:free":
            return httpx.Response(404, json={"error": {"message": "No endpoints found"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    _patch_client(monkeypatch, handler)
    p = OpenRouterProvider(api_key="k", roster=["gone:free", "live:free"], tracker=tr)
    assert p.generate(SommRequest(prompt="hi")).text == "ok"
    assert p.generate(SommRequest(prompt="hi")).text == "ok"
    assert seen == ["gone:free", "live:free", "live:free"]
    assert tr.get("openrouter", "gone:free").is_cooling()

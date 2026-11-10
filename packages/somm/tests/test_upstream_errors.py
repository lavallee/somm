"""One HTTP classifier with the provider's own detail, and pooled clients.

Error bodies here are synthetic but follow each provider's documented error
object (cited in somm.errors._upstream_fields).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommError,
    SommInsufficientCredit,
    SommModelNotFound,
    SommRateLimited,
    SommRefused,
    SommTimeout,
    classify_body_error,
    classify_http_status,
    describe_error,
    http_error,
)
from somm.providers import _http
from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import SommRequest
from somm.providers.ollama import OllamaProvider
from somm.providers.openai import OpenAIProvider

_OPENAI_OK = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def _raised(status, body="", headers=None, **kw) -> SommError:
    with pytest.raises(SommError) as info:
        classify_http_status(status, body, provider="p", model="m", headers=headers, **kw)
    return info.value


# --- fields on every error ---------------------------------------------------


def test_fields_default_to_none_on_every_error():
    for exc in (SommError("x"), SommBadRequest("x"), SommRefused("x")):
        assert exc.status_code is None and exc.provider_code is None
        assert exc.provider_type is None and exc.provider_param is None
        assert exc.provider_request_id is None and exc.retry_after_s is None


def test_openai_error_object_and_request_id_are_read():
    body = json.dumps({"error": {"message": "echo of the prompt", "type": "invalid_request_error",
                                 "param": "messages", "code": "context_length_exceeded"}})
    exc = _raised(400, body, {"x-request-id": "req_abc"})
    assert isinstance(exc, SommBadRequest)
    assert (exc.status_code, exc.provider_code, exc.provider_type, exc.provider_param) == (
        400, "context_length_exceeded", "invalid_request_error", "messages")
    assert exc.provider_request_id == "req_abc"
    described = describe_error(exc)
    assert described["provider_param"] == "messages" and described["status_code"] == 400
    # The provider's message never lands in a machine field.
    assert all("echo" not in str(v) for k, v in described.items() if k != "message")


def test_anthropic_request_id_from_header_or_body():
    body = json.dumps({"type": "error", "error": {"type": "permission_error", "message": "no"},
                       "request_id": "req_body"})
    assert _raised(403, body).provider_request_id == "req_body"
    assert _raised(403, body, {"request-id": "req_hdr"}).provider_request_id == "req_hdr"


def test_403_is_preserved_as_403():
    exc = _raised(403, '{"error":{"type":"permission_error","message":"x"}}')
    assert isinstance(exc, SommAuthError) and exc.status_code == 403


def test_openrouter_numeric_code_is_a_string():
    exc = _raised(404, '{"error":{"code":404,"message":"No endpoints found"}}')
    assert isinstance(exc, SommModelNotFound) and exc.provider_code == "404"


def test_body_error_inside_a_200_carries_the_error_object_fields():
    with pytest.raises(SommRateLimited) as info:
        classify_body_error({"code": 429, "type": "rate_limit_error", "message": "slow"},
                            provider="p", model="m")
    assert info.value.provider_code == "429" and info.value.status_code is None


def test_long_or_structured_values_are_bounded():
    body = json.dumps({"error": {"code": "x" * 1000, "type": {"nested": 1}, "param": None}})
    exc = _raised(400, body)
    assert len(exc.provider_code) == 256 and exc.provider_type is None


def test_a_fake_without_headers_or_json_still_classifies():
    exc = _raised(500, "not json", None)
    assert exc.status_code == 500 and exc.provider_code is None


# --- credit, waits and timeouts ---------------------------------------------


def test_402_is_insufficient_credit_and_not_retryable_for_a_caller():
    exc = _raised(402, '{"error":{"code":402,"message":"add credits"}}')
    assert isinstance(exc, SommInsufficientCredit) and exc.status_code == 402
    described = describe_error(exc)
    assert described["retryable"] is False and "retry_after_s" not in described


@pytest.mark.parametrize("code", ["credit_balance_exhausted", "project_spend_limit_exceeded"])
def test_openai_billing_codes_are_insufficient_credit(code):
    body = json.dumps({"error": {"type": "insufficient_quota", "code": code, "message": "m"}})
    exc = _raised(429, body)
    assert isinstance(exc, SommInsufficientCredit) and exc.provider_code == code


def test_anthropic_spend_cap_429_is_insufficient_credit():
    body = json.dumps({"type": "error", "error": {
        "type": "rate_limit_error", "message": "You have reached your API usage limits",
        "details": {"error_code": "enforced_spend_limit_reached"}}})
    exc = _raised(429, body)
    assert isinstance(exc, SommInsufficientCredit)
    assert exc.provider_code == "enforced_spend_limit_reached"


def test_minimax_plan_exhaustion_is_still_insufficient_credit():
    body = json.dumps({"error": {"type": "rate_limit_error",
                                 "message": "Token Plan usage limit reached (2056)"}})
    assert isinstance(_raised(429, body), SommInsufficientCredit)


def test_retry_after_zero_is_zero_not_two_minutes():
    exc = _raised(429, "", {"Retry-After": "0"})
    assert isinstance(exc, SommRateLimited) and exc.retry_after_s == 0.0


def test_rate_limit_reset_headers_when_retry_after_is_absent():
    soon = (datetime.now(UTC) + timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    exc = _raised(429, "", {"anthropic-ratelimit-tokens-reset": soon})
    assert 25 <= exc.retry_after_s <= 30
    exc = _raised(429, "", {"x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s"})
    assert exc.retry_after_s == 360.0
    assert _raised(429, "").retry_after_s == 120.0


def test_retry_after_on_a_503_is_reported():
    assert _raised(503, "", {"retry-after": "7"}).retry_after_s == 7.0


def test_408_is_a_timeout():
    assert isinstance(_raised(408), SommTimeout)


def test_detail_replaces_the_body_quote():
    exc = http_error(400, '{"error":{"message":"secret prompt"}}', provider="p", model="m",
                     detail="p failed: bad")
    assert str(exc) == "p failed: bad" and exc.status_code == 400


def test_refused_carries_the_partial_result():
    exc = SommRefused("no", stop_reason="refusal", text="partial", tokens_in=3, tokens_out=1,
                      model="served", call_id="c")
    assert (exc.text, exc.tokens_in, exc.tokens_out, exc.model, exc.call_id) == (
        "partial", 3, 1, "served", "c")
    assert describe_error(exc)["stop_reason"] == "refusal"
    assert "text" not in describe_error(exc)


# --- adapters fill the fields -----------------------------------------------


def _mock(monkeypatch, handler):
    made: list = []
    real = httpx.Client

    class Client(real):
        def __init__(self, *args, **kwargs):
            made.append(self)
            kwargs.setdefault("transport", httpx.MockTransport(handler))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)
    return made


@pytest.mark.parametrize("make", [
    lambda: OpenAIProvider(api_key="k", default_model="gpt-x"),
    lambda: AnthropicProvider(api_key="k"),
    lambda: OllamaProvider(base_url="http://127.0.0.1:1"),
])
def test_every_http_adapter_raises_with_the_upstream_fields(monkeypatch, make):
    body = {"error": {"type": "permission_error", "message": "no", "code": "denied"}}
    _mock(monkeypatch, lambda r: httpx.Response(403, json=body, headers={"x-request-id": "r1"}))
    with pytest.raises(SommAuthError) as info:
        make().generate(SommRequest(prompt="hi"))
    assert info.value.status_code == 403
    assert info.value.provider_type == "permission_error"
    assert info.value.provider_request_id == "r1"


def test_ollama_404_keeps_its_pull_hint_and_the_status(monkeypatch):
    _mock(monkeypatch, lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(SommModelNotFound) as info:
        OllamaProvider(base_url="http://127.0.0.1:1").generate(SommRequest(prompt="hi", model="q"))
    assert "ollama pull q" in str(info.value) and info.value.status_code == 404


# --- connection reuse ---------------------------------------------------------


def test_one_client_serves_every_call_and_close_reopens(monkeypatch):
    made = _mock(monkeypatch, lambda r: httpx.Response(200, json=_OPENAI_OK))
    provider = OpenAIProvider(api_key="k", default_model="gpt-x")
    for _ in range(3):
        provider.generate(SommRequest(prompt="hi"))
    assert len(made) == 1
    provider.close()
    assert made[0].is_closed
    provider.generate(SommRequest(prompt="hi"))
    assert len(made) == 2


def test_per_call_timeout_rides_on_the_request(monkeypatch):
    seen: list = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json=_OPENAI_OK)

    _mock(monkeypatch, handler)
    provider = OpenAIProvider(api_key="k", default_model="gpt-x", timeout=50)
    provider.generate(SommRequest(prompt="hi", timeout=3))
    provider.generate(SommRequest(prompt="hi"))
    assert [t["read"] for t in seen] == [3, 50]
    assert [t["connect"] for t in seen] == [3, 10.0]


def test_a_copied_provider_shares_the_client(monkeypatch):
    import copy

    made = _mock(monkeypatch, lambda r: httpx.Response(200, json=_OPENAI_OK))
    provider = OpenAIProvider(api_key="k", default_model="gpt-x")
    copy.copy(provider).generate(SommRequest(prompt="hi"))
    provider.generate(SommRequest(prompt="hi"))
    assert len(made) == 1


def test_a_forked_child_does_not_reuse_the_parents_client(monkeypatch):
    made = _mock(monkeypatch, lambda r: httpx.Response(200, json=_OPENAI_OK))
    provider = OpenAIProvider(api_key="k", default_model="gpt-x")
    provider.generate(SommRequest(prompt="hi"))
    parent_pid = _http.os.getpid()
    monkeypatch.setattr(_http.os, "getpid", lambda: parent_pid + 1)
    provider.generate(SommRequest(prompt="hi"))
    assert len(made) == 2
    # The parent's client is abandoned, not closed: its sockets are the parent's.
    assert not made[0].is_closed

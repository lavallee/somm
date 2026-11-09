"""Subscription transport must preserve tools and never accept a partial SSE answer."""
import json
from unittest.mock import Mock

import httpx
import pytest
from somm.errors import SommRateLimited, SommUpstream5xx
from somm.providers.chatgpt import ChatGPTProvider, response_input


def test_tool_reasoning_round_trip():
    from somm.providers.chatgpt import chat_completion

    item = {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"}
    result = chat_completion({"id": "resp1", "model": "actual-model", "output": [item,
        {"type": "function_call", "call_id": "c1", "name": "groceries", "arguments": '{"item":"milk"}'}]})
    message = result["choices"][0]["message"]
    replay = response_input([message, {"role": "tool", "tool_call_id": "c1", "content": "added"}])
    assert replay == [item, {"type": "function_call", "call_id": "c1", "name": "groceries",
                            "arguments": '{"item":"milk"}'},
                      {"type": "function_call_output", "call_id": "c1", "output": "added"}]
    assert result["model"] == "actual-model"


def transport(monkeypatch, text, status=200):
    original = httpx.Client
    def handler(request):
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer synthetic"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(status, text=text)
    monkeypatch.setattr("somm.providers.chatgpt.httpx.Client", lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw))
    return ChatGPTProvider(auth=Mock(access_token=Mock(return_value="synthetic")))


def test_only_completed_event_is_success_even_without_trailing_blank_line(monkeypatch):
    value = {"type": "response.completed", "response": {"status": "completed", "id": "r1",
             "model": "served-model", "output": [{"type": "message", "content": [
                 {"type": "output_text", "text": "hello"}]}], "usage": {"input_tokens": 2, "output_tokens": 1}}}
    provider = transport(monkeypatch, "data: " + json.dumps(value))
    result = provider.complete_chat({"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
    assert result["model"] == "served-model"
    assert result["choices"][0]["message"]["content"] == "hello"


@pytest.mark.parametrize("event", ["response.failed", "response.incomplete", "response.output_text.delta"])
def test_http_200_does_not_make_failed_or_interrupted_response_success(monkeypatch, event):
    provider = transport(monkeypatch, 'data: ' + json.dumps({"type": event}) + '\n\n')
    with pytest.raises(SommUpstream5xx):
        provider.complete_chat({"model": "x", "messages": []})


def test_rate_limit_is_not_a_completed_answer(monkeypatch):
    provider = transport(monkeypatch, "", 429)
    with pytest.raises(SommRateLimited):
        provider.complete_chat({"model": "x", "messages": []})

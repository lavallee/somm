"""Tests for SommChatModel — LangChain ⇄ somm translation, bind_tools, error path."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm.providers.base import ToolCall as ProviderToolCall
from somm_core.config import Config
from somm_langchain import SommChatModel

# ---------------------------------------------------------------------------
# Recording fake provider — captures every SommRequest it sees


class _RecordingProvider:
    name = "recorder"

    def __init__(
        self,
        text: str = "",
        tool_calls: list[ProviderToolCall] | None = None,
        stop_reason: str = "end_turn",
    ):
        self._text = text
        self._tool_calls = tool_calls or []
        self._stop_reason = stop_reason
        self.received: list = []

    def generate(self, request):
        self.received.append(request)
        return SommResponse(
            text=self._text,
            model=request.model or "fake-model",
            tokens_in=3,
            tokens_out=2,
            latency_ms=5,
            tool_calls=self._tool_calls,
            stop_reason=self._stop_reason,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_llm(tmp_path: Path, provider: _RecordingProvider) -> SommLLM:
    cfg = Config()
    cfg.project = "somm-lc-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return SommLLM(config=cfg, providers=[provider], on_error=lambda _e: None)


# ---------------------------------------------------------------------------
# Message translation


def test_system_message_extracted_to_system_field(tmp_path):
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        chat.invoke([SystemMessage("be brief"), HumanMessage("hi")])
        req = p.received[0]
        assert req.system == "be brief"
        assert req.messages == [{"role": "user", "content": "hi"}]
    finally:
        llm.close()


def test_multiple_system_messages_concatenate(tmp_path):
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        chat.invoke([SystemMessage("rule one"), SystemMessage("rule two"), HumanMessage("hi")])
        assert p.received[0].system == "rule one\n\nrule two"
    finally:
        llm.close()


def test_ai_message_with_tool_calls_becomes_tool_use_blocks(tmp_path):
    p = _RecordingProvider(text="next")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        chat.invoke(
            [
                HumanMessage("weather?"),
                AIMessage(
                    content="checking",
                    tool_calls=[{"name": "get_weather", "args": {"location": "SF"}, "id": "tu_01"}],
                ),
                ToolMessage(content="62F sunny", tool_call_id="tu_01"),
            ]
        )
        msgs = p.received[0].messages
        # User text
        assert msgs[0] == {"role": "user", "content": "weather?"}
        # Assistant with text + tool_use blocks (mixed → list form)
        assistant_blocks = msgs[1]["content"]
        assert msgs[1]["role"] == "assistant"
        assert {"type": "text", "text": "checking"} in assistant_blocks
        tool_use_block = next(b for b in assistant_blocks if b.get("type") == "tool_use")
        assert tool_use_block == {
            "type": "tool_use",
            "id": "tu_01",
            "name": "get_weather",
            "input": {"location": "SF"},
        }
        # Tool result becomes user message with tool_result block
        assert msgs[2] == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_01", "content": "62F sunny"}
            ],
        }
    finally:
        llm.close()


def test_ai_message_text_only_collapses_to_string(tmp_path):
    """Text-only assistant turns use plain-string content (interoperable shape)."""
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        chat.invoke([HumanMessage("hi"), AIMessage(content="hello there")])
        msgs = p.received[0].messages
        assert msgs[1] == {"role": "assistant", "content": "hello there"}
    finally:
        llm.close()


def test_ai_message_tool_calls_only_no_content(tmp_path):
    """Assistant with only tool_calls (no text) — blocks list contains only tool_use."""
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        chat.invoke(
            [
                HumanMessage("x"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "f", "args": {"a": 1}, "id": "tu_2"}],
                ),
                ToolMessage(content="r", tool_call_id="tu_2"),
            ]
        )
        msgs = p.received[0].messages
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == [
            {"type": "tool_use", "id": "tu_2", "name": "f", "input": {"a": 1}}
        ]
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# bind_tools — LangChain tools → somm-neutral; tool_choice routes correctly


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny in {location}"


def test_bind_tools_routes_through_to_somm_neutral_shape(tmp_path):
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        bound = chat.bind_tools([get_weather])
        bound.invoke([HumanMessage("weather?")])

        sent_tools = p.received[0].tools
        assert len(sent_tools) == 1
        # Neutral shape (post-unwrap), NOT OpenAI's {type:function,...}
        sent = sent_tools[0]
        assert sent["name"] == "get_weather"
        assert "parameters" in sent
        assert "type" not in sent  # would be present if OpenAI wrapping leaked
    finally:
        llm.close()


def test_bind_tools_with_specific_tool_choice(tmp_path):
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        bound = chat.bind_tools([get_weather], tool_choice="get_weather")
        bound.invoke([HumanMessage("weather?")])
        assert p.received[0].tool_choice == {"type": "tool", "name": "get_weather"}
    finally:
        llm.close()


def test_bind_tools_with_required(tmp_path):
    """OpenAI alias 'required' maps to somm 'any'."""
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        bound = chat.bind_tools([get_weather], tool_choice="required")
        bound.invoke([HumanMessage("weather?")])
        assert p.received[0].tool_choice == "any"
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Response translation — SommResult → AIMessage


def test_response_with_tool_calls_returns_aimessage_with_tool_calls(tmp_path):
    tool_calls = [ProviderToolCall(id="tu_5", name="get_weather", arguments={"location": "NYC"})]
    p = _RecordingProvider(text="checking", tool_calls=tool_calls, stop_reason="tool_use")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        result = chat.invoke([HumanMessage("weather?")])
        assert isinstance(result, AIMessage)
        assert result.content == "checking"
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call["id"] == "tu_5"
        assert call["name"] == "get_weather"
        assert call["args"] == {"location": "NYC"}
        # Metadata exposes somm provenance
        assert result.response_metadata["provider"] == "recorder"
        assert result.response_metadata["stop_reason"] == "tool_use"
        assert result.usage_metadata["input_tokens"] == 3
        assert result.usage_metadata["output_tokens"] == 2
    finally:
        llm.close()


def test_text_only_response_returns_aimessage_with_text(tmp_path):
    p = _RecordingProvider(text="hello there")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        result = chat.invoke([HumanMessage("hi")])
        assert result.content == "hello there"
        assert result.tool_calls == []
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Error semantics


class _FailingProvider(_RecordingProvider):
    name = "broken"

    def generate(self, request):
        self.received.append(request)
        from somm.errors import SommTransientError

        raise SommTransientError("upstream down", cooldown_s=3600)


def test_raise_on_failure_true_raises(tmp_path):
    p = _FailingProvider()
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        with pytest.raises(RuntimeError, match="somm call failed"):
            chat.invoke([HumanMessage("hi")])
    finally:
        llm.close()


def test_raise_on_failure_false_returns_error_in_metadata(tmp_path):
    p = _FailingProvider()
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t", raise_on_failure=False)
        result = chat.invoke([HumanMessage("hi")])
        # outcome is not OK; we still return a message rather than raising
        assert result.response_metadata["outcome"] != "ok"
        assert result.content == ""
    finally:
        llm.close()


def test_model_and_provider_pinning(tmp_path):
    """somm_model + somm_provider on the adapter pin the underlying SommLLM call."""
    p = _RecordingProvider(text="ok")
    p.name = "pinprovider"
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(
            somm_llm=llm, workload="t",
            somm_model="my-model-id", somm_provider="pinprovider",
        )
        chat.invoke([HumanMessage("hi")])
        req = p.received[0]
        assert req.model == "my-model-id"
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# reasoning_content round-trip (DeepSeek v4 thinking models)


def test_reasoning_content_surfaced_to_aimessage(tmp_path):
    """A provider response with reasoning_content lands in AIMessage.additional_kwargs."""
    p = _RecordingProvider(text="answer")
    # SommResponse carries reasoning_content; RecordingProvider builds it,
    # so set it via a subclass-free monkey: wrap generate.
    orig = p.generate
    def gen(req):
        r = orig(req)
        r.reasoning_content = "let me think... 2+2=4"
        return r
    p.generate = gen
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        result = chat.invoke([HumanMessage("2+2?")])
        assert result.additional_kwargs.get("reasoning_content") == "let me think... 2+2=4"
    finally:
        llm.close()


def test_reasoning_content_echoed_on_assistant_turn(tmp_path):
    """An AIMessage carrying reasoning_content (with tool_calls) re-serializes it
    onto the somm-neutral assistant message so the provider can echo it."""
    p = _RecordingProvider(text="ok")
    llm = _tmp_llm(tmp_path, p)
    try:
        chat = SommChatModel(somm_llm=llm, workload="t")
        prior = AIMessage(
            content="checking",
            tool_calls=[{"name": "get_weather", "args": {"location": "SF"}, "id": "tu_1"}],
            additional_kwargs={"reasoning_content": "I should call get_weather"},
        )
        chat.invoke([
            HumanMessage("weather?"),
            prior,
            ToolMessage(content="62F", tool_call_id="tu_1"),
        ])
        msgs = p.received[0].messages
        assistant = msgs[1]
        assert assistant["role"] == "assistant"
        assert assistant.get("reasoning_content") == "I should call get_weather"
    finally:
        llm.close()

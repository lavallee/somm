"""SommChatModel — LangChain BaseChatModel implementation that routes through somm.

Lets LangChain / LangGraph / deepagents apps treat somm as their LLM substrate.
Thin adapter: message translation in, somm.llm().generate() through, ChatGeneration out.

Tool-calling routes through somm's neutral tool shape (see docs/tool-calling.md),
which is supported across all somm chat providers.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict
from somm.client import SommLLM
from somm_core import Outcome


class SommChatModel(BaseChatModel):
    """LangChain chat model that delegates to a `SommLLM`.

    Usage:
        ```python
        import somm
        from somm_langchain import SommChatModel

        llm = somm.llm(project="my_agent")
        chat = SommChatModel(somm_llm=llm, workload="agent_thinking")
        ```

    Pass to `deepagents.create_deep_agent(model=chat, ...)` or any LangChain
    runnable expecting a chat model. `bind_tools()` works — tool schemas are
    converted to somm-neutral form and routed via `SommLLM.generate(tools=...)`.

    All telemetry, provider routing, cost tracking, and fallback semantics are
    somm's — this adapter adds no behavior of its own.
    """

    # Allow non-pydantic types (SommLLM) as fields.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    somm_llm: SommLLM
    workload: str = "agent"
    # Pin a specific (model, provider) when set; otherwise somm's router picks.
    somm_model: str | None = None
    somm_provider: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096

    # Whether SommLLM.generate() failures (outcome != OK) should raise a
    # RuntimeError or return as an AIMessage with the error in
    # response_metadata. Default: raise — matches LangChain convention so
    # retry / circuit-breaker middleware in the agent stack engages.
    raise_on_failure: bool = True

    # ClassVar so pydantic doesn't treat it as a field
    LC_ROLE_BLOCK_KEYS: ClassVar[set[str]] = {"text", "tool_use", "tool_result", "image"}

    # ------------------------------------------------------------------
    # LangChain BaseChatModel interface

    @property
    def _llm_type(self) -> str:
        return "somm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, somm_messages = self._translate_messages(messages)

        # Tools come from bind_tools() via Runnable.bind kwargs. They were
        # converted to OpenAI shape there; unwrap to somm-neutral here.
        raw_tools = kwargs.get("tools") or []
        somm_tools = [self._unwrap_oai_tool(t) for t in raw_tools]

        tool_choice = kwargs.get("tool_choice")
        somm_tool_choice = self._translate_tool_choice_lc_to_somm(tool_choice)

        result = self.somm_llm.generate(
            # `prompt` is required by somm's signature but ignored when
            # `messages` is set — we pass an empty string as a marker.
            prompt="",
            messages=somm_messages or None,
            system=system,
            workload=self.workload,
            model=kwargs.get("model") or self.somm_model,
            provider=self.somm_provider,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            tools=somm_tools if somm_tools else None,
            tool_choice=somm_tool_choice,
        )

        if result.outcome != Outcome.OK and self.raise_on_failure:
            raise RuntimeError(
                f"somm call failed: workload={self.workload} "
                f"outcome={result.outcome.value} kind={result.error_kind} — "
                f"{result.error_detail}"
            )

        lc_tool_calls = [
            {"name": tc.name, "args": tc.arguments, "id": tc.id}
            for tc in result.tool_calls
        ]
        # Preserve reasoning_content (DeepSeek v4 thinking models) in
        # additional_kwargs so it survives in LangGraph's message history and
        # can be echoed back on the next turn (_translate_ai_message). DeepSeek
        # 400s on the 2nd turn of a tool-calling loop without it.
        additional_kwargs: dict[str, Any] = {}
        if result.reasoning_content:
            additional_kwargs["reasoning_content"] = result.reasoning_content
        ai_message = AIMessage(
            content=result.text,
            tool_calls=lc_tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata={
                "provider": result.provider,
                "model": result.model,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "stop_reason": result.stop_reason,
                "outcome": result.outcome.value,
                "call_id": result.call_id,
            },
            usage_metadata={
                "input_tokens": result.tokens_in,
                "output_tokens": result.tokens_out,
                "total_tokens": result.tokens_in + result.tokens_out,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(
        self,
        tools: list[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools to this model. Returns a Runnable that injects them on call.

        Tools are converted to OpenAI shape (the most common cross-LangChain
        format) here; `_generate` unwraps them to somm-neutral before calling
        SommLLM. This keeps interop wide: any LangChain-shaped tool spec works.
        """
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted_tools, **kwargs)

    # ------------------------------------------------------------------
    # Translation helpers — LangChain ⇄ somm-neutral

    def _translate_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[str, list[dict[str, Any]]]:
        """LangChain messages → (system, somm-neutral messages).

        somm takes system separately, so SystemMessage(s) are extracted and
        joined into the `system` field. Remaining messages go into the
        Anthropic-shaped neutral format. The OpenAI-compat provider adapter
        translates the neutral shape back into OpenAI's tool_calls + role:tool
        message form at the provider layer.
        """
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(self._stringify_content(m.content))
                continue
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": m.content})
                continue
            if isinstance(m, AIMessage):
                out.append(self._translate_ai_message(m))
                continue
            if isinstance(m, ToolMessage):
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": self._stringify_content(m.content),
                            }
                        ],
                    }
                )
                continue
            # Unknown message subclass — best-effort string forward
            out.append({"role": "user", "content": self._stringify_content(m.content)})
        return "\n\n".join(p for p in system_parts if p), out

    def _translate_ai_message(self, m: AIMessage) -> dict[str, Any]:
        """AIMessage → somm-neutral assistant message with optional tool_use blocks."""
        blocks: list[dict[str, Any]] = []
        text = self._stringify_content(m.content)
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in m.tool_calls or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or "",
                    "name": tc.get("name") or "",
                    "input": tc.get("args") or {},
                }
            )
        # Echo reasoning_content (DeepSeek v4 thinking models) back on the
        # assistant turn — required or DeepSeek 400s on the next turn. Carried
        # as a top-level key the OpenAI-compat provider emits and the Anthropic
        # provider strips. Only matters on tool-calling turns (which have
        # tool_use blocks → list content, so collapse-to-string never drops it).
        reasoning = (m.additional_kwargs or {}).get("reasoning_content")

        # Collapse to plain-string content when the only block is text.
        if len(blocks) == 1 and blocks[0].get("type") == "text" and not reasoning:
            return {"role": "assistant", "content": cast(str, blocks[0]["text"])}
        msg: dict[str, Any] = {"role": "assistant", "content": blocks}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return msg

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _unwrap_oai_tool(t: dict[str, Any]) -> dict[str, Any]:
        """OpenAI `{type:function, function:{...}}` → somm-neutral
        `{name, description, parameters}`. Pass through if already neutral."""
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            return dict(t["function"])
        return t

    @staticmethod
    def _translate_tool_choice_lc_to_somm(
        choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        """LangChain tool_choice (per bind_tools convention) → somm-neutral.

        LangChain accepts: None, "auto", "any", "none", "required" (OpenAI
        alias for any), a tool name string, or a dict like
        `{"type":"function","function":{"name":"X"}}` / `{"type":"tool","name":"X"}`.
        """
        if choice is None:
            return None
        if isinstance(choice, str):
            if choice in ("auto", "any", "none"):
                return choice
            if choice == "required":
                return "any"  # somm uses "any" for "must call SOME tool"
            # Anything else: assume it's a specific tool name
            return {"type": "tool", "name": choice}
        if isinstance(choice, dict):
            # Already somm-neutral?
            if choice.get("type") == "tool" and "name" in choice:
                return choice
            # OpenAI-shaped — extract the function name
            if choice.get("type") == "function":
                fn = choice.get("function") or {}
                name = fn.get("name")
                if name:
                    return {"type": "tool", "name": name}
        return choice

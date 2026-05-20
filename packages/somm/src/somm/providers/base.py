"""SommProvider protocol — public, stable, entry-point registerable.

Third parties implement this to ship new providers without forking somm.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SommRequest:
    prompt: str | list[dict]
    system: str = ""
    max_tokens: int = 256
    temperature: float = 0.2
    model: str | None = None  # None = provider's default
    metadata: dict = field(default_factory=dict)
    # Capabilities required of the (provider, model) serving this request.
    # Auto-inferred from image blocks, merged with workload defaults, and
    # filtered against model_intel.capabilities_json by the router.
    capabilities_required: list[str] = field(default_factory=list)
    # When False (default), the router treats an empty response as a
    # transient failure and tries the next provider. Set True only if the
    # caller genuinely expects some prompts to produce no output.
    allow_empty: bool = False
    # ------------------------------------------------------------------
    # Tool-calling (see docs/tool-calling.md)
    #
    # `tools`: somm-neutral JSON-Schema-shaped tool declarations. Empty
    # list (default) means no tool calling. Provider adapters translate
    # to their native format. Providers that don't yet support tool
    # calling raise SommBadRequest if `tools` is non-empty.
    #
    # `messages`: multi-turn conversation history. When set, overrides
    # `prompt`. Format mirrors Anthropic Messages API (the closest to a
    # cross-provider common denominator); OpenAI-compat adapter
    # translates assistant/tool_use and user/tool_result blocks into
    # OpenAI's tool_calls / tool messages.
    #
    # `tool_choice`: None (default), "auto", "any", "none", or
    # {"type":"tool","name":"<tool_name>"} to force a specific tool.
    tools: list[dict] = field(default_factory=list)
    messages: list[dict] | None = None
    tool_choice: str | dict | None = None


@dataclass(slots=True)
class ToolCall:
    """Provider-level tool invocation. Mirrors somm_core.models.ToolCall
    so providers don't need a somm-core import — client.py converts."""

    id: str
    name: str
    arguments: dict
    arguments_raw: str = ""


@dataclass(slots=True)
class SommResponse:
    text: str
    model: str  # actual model used (after routing)
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: dict | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    # Chain-of-thought from thinking models (DeepSeek v4). Must be echoed back
    # on the assistant turn in multi-turn calls or DeepSeek 400s on turn 2.
    reasoning_content: str = ""


@dataclass(slots=True)
class SommChunk:
    text: str
    done: bool = False


@dataclass(slots=True)
class SommEmbedRequest:
    text: str
    model: str | None = None  # None = provider's default embed model


@dataclass(slots=True)
class SommEmbedResponse:
    embedding: list[float]
    model: str
    tokens_in: int
    latency_ms: int
    raw: dict | None = None


@dataclass(slots=True)
class ProviderHealth:
    available: bool
    detail: str = ""


@dataclass(slots=True)
class SommModel:
    name: str
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)


@runtime_checkable
class SommProvider(Protocol):
    """Every provider adapter implements this protocol."""

    name: str

    def generate(self, request: SommRequest) -> SommResponse: ...

    def stream(self, request: SommRequest) -> Iterator[SommChunk]: ...

    def health(self) -> ProviderHealth: ...

    def models(self) -> list[SommModel]: ...

    def estimate_tokens(self, text: str | list[dict], model: str) -> int: ...

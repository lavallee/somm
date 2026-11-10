"""SommProvider protocol — public, stable, entry-point compatible.

Third-party providers return objects satisfying this protocol from a
``somm.providers.registry.ProviderSpec`` factory. Packages expose that spec
with the ``somm.providers`` entry-point group, for example:

    [project.entry-points."somm.providers"]
    acme = "acme_somm:provider_spec"

The entry point resolves to either a ``ProviderSpec`` instance or a zero-arg
callable returning one. The spec factory is called with ``(config, tracker)``
and returns a provider instance, or ``None`` when the provider is not
configured on this machine.
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
    # Reasoning budget for models that spend tokens thinking before they
    # answer. Provider-neutral: adapters translate or drop it. DeepSeek v4
    # accepts none|minimal|low|medium|high|xhigh|max on both flash and pro, so
    # "thinking" is a dial on a request rather than a property of a model.
    reasoning_effort: str | None = None
    tools: list[dict] = field(default_factory=list)
    messages: list[dict] | None = None
    tool_choice: str | dict | None = None
    # Provider-native structured-output hint. OpenAI-compatible adapters pass
    # this through as response_format; Ollama maps it to format. Validation
    # still belongs to SommLLM.generate_structured().
    response_format: dict | None = None
    claude_cli_text_only: bool = False
    # The loosest data tier (somm_core.boundary) this request may reach, set
    # by SommLLM from the workload. The router drops providers outside it, and
    # a provider that picks its own model (OpenRouter's roster walk) must skip
    # models outside it. None = no boundary (every route admissible).
    data_boundary: str | None = None
    # Seconds this one call may take, overriding the adapter's configured
    # timeout (SOMM_HTTP_TIMEOUT). None = use the adapter's own.
    timeout: float | None = None


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
    # Cost the PROVIDER reported for this call, when it reports one. A CLI
    # seat knows what it charged; computing a price from a token count would
    # invent a number for a call that already has a real one. None means the
    # provider said nothing and the client should compute as usual.
    cost_usd: float | None = None
    # Tokens the model spent thinking before it answered, when the provider
    # reports them separately (OpenAI-compatible
    # ``completion_tokens_details.reasoning_tokens``). None when unknown.
    reasoning_tokens: int | None = None
    # The output cap the adapter actually put on the wire, when it differs
    # from the request's (several adapters multiply max_tokens to leave room
    # for reasoning). None means it sent ``request.max_tokens`` unchanged.
    max_tokens_sent: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One fragment of a streamed tool call.

    ``index`` is the tool call's position in the turn (0 for the first);
    ``id`` and ``name`` arrive on its first fragment and are "" after;
    ``arguments`` is the next piece of the JSON argument string. Joining the
    fragments of one index gives the whole call.
    """

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass(slots=True)
class SommChunk:
    text: str
    done: bool = False
    # The model that served the stream, on the final chunk, as
    # SommResponse.model does for generate. "" when the adapter does not say.
    model: str = ""
    # Chain-of-thought delta, for adapters whose wire format streams it, and
    # the provider's signature closing a reasoning block (Anthropic).
    reasoning: str = ""
    reasoning_signature: str = ""
    # Tool-call fragments carried by this chunk.
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    # On the final chunk: the whole answer as generate would have returned
    # it (usage, tool calls, stop reason, raw body). The client records from
    # it; None means the adapter has no usage and the client estimates.
    response: SommResponse | None = None


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

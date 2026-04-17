"""OpenAI SDK compatibility shim — swap in for `openai.OpenAI().chat.completions.create()`.

Signature matches OpenAI's Python SDK's `chat.completions.create(model,
messages, ...)`. Returns a response object with the same `.choices[0]
.message.content` + `.usage` + `.model` attributes legacy call sites read.

Under the hood: routes through somm's provider chain (ollama / openrouter
/ anthropic / openai / minimax / self-hosted gateways). Writes full
telemetry + provenance.

Usage:

    # Before:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    print(resp.choices[0].message.content)

    # After:
    from somm.compat import openai_chat_completions as create
    resp = create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        project="myproject",
        workload="greet",
    )
    print(resp.choices[0].message.content)

Provider selection: the model string's prefix (before "/") picks the
somm provider. "openai/..." / "ollama/..." / "anthropic/..." /
"openrouter/..." / "minimax/..." or just "openai_compat/..." for any
OpenAI-compatible gateway. If no prefix: falls back to SommLLM's routed
chain (first non-cooled provider wins).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from somm.client import SommLLM


@dataclass(slots=True)
class OpenAIMessage:
    role: str
    content: str


@dataclass(slots=True)
class OpenAIChoice:
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"


@dataclass(slots=True)
class OpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class OpenAIChatCompletion:
    """Mimics openai.types.chat.ChatCompletion. Exposes the attributes
    legacy call sites reach for.
    """

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage
    # somm-native extras:
    somm_call_id: str = ""
    somm_cost_usd: float = 0.0
    somm_latency_ms: int = 0
    somm_outcome: str = "ok"
    somm_provider: str = ""
    _somm_raw: dict[str, Any] = field(default_factory=dict)


_llm_singleton: SommLLM | None = None


def _get_llm(project: str | None = None) -> SommLLM:
    """Lazy-singleton SommLLM. Reuses one handle across calls to avoid
    writer-thread churn in long-running processes."""
    global _llm_singleton
    from somm.client import SommLLM

    if _llm_singleton is None:
        _llm_singleton = SommLLM(project=project)
    return _llm_singleton


def openai_chat_completions(
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.2,
    project: str | None = None,
    workload: str = "default",
    **_unused,
) -> OpenAIChatCompletion:
    """OpenAI-SDK-compatible chat completion. Routes through somm.

    Args:
        model: model identifier. "provider/model" picks a provider
          explicitly; unprefixed routes via the default chain.
        messages: OpenAI-style messages [{"role": "user"|"system", "content": "..."}].
        max_tokens, temperature: standard.
        project: somm project (env var fallback).
        workload: workload tag (default "default"; strict mode requires registration).

    Returns:
        OpenAIChatCompletion (attributes-compatible with openai SDK).
    """
    llm = _get_llm(project=project)

    provider: str | None = None
    if "/" in model:
        provider, _, model = model.partition("/")

    system = ""
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system = content
        else:
            user_parts.append(content)
    prompt = "\n\n".join(user_parts)

    created = int(time.time())
    r = llm.generate(
        prompt=prompt,
        system=system,
        workload=workload,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        provider=provider,
    )

    return OpenAIChatCompletion(
        id=f"somm-{r.call_id}",
        object="chat.completion",
        created=created,
        model=r.model,
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=r.text),
                finish_reason="stop" if r.outcome.value == "ok" else r.outcome.value,
            ),
        ],
        usage=OpenAIUsage(
            prompt_tokens=r.tokens_in,
            completion_tokens=r.tokens_out,
            total_tokens=r.tokens_in + r.tokens_out,
        ),
        somm_call_id=r.call_id,
        somm_cost_usd=r.cost_usd,
        somm_latency_ms=r.latency_ms,
        somm_outcome=r.outcome.value,
        somm_provider=r.provider,
        _somm_raw=r.raw or {},
    )

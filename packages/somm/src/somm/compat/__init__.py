"""Drop-in compatibility shims for grafting somm into existing codebases.

The most common migration patterns:

1. **Custom-wrapper codebase.** Project has `FooLLM.generate(prompt, system,
   max_tokens, provider=None) -> LLMResult(text, provider, model)`.
   Swap in `GenericLLMCompat` — it takes the same args, returns a dict-
   like object with the same attributes, and writes full telemetry +
   provenance underneath.

2. **Raw OpenAI SDK.** Project calls `openai.OpenAI().chat.completions
   .create(model=..., messages=...)`. Use `openai_chat_completions` —
   matches the method signature and return shape, routes through somm's
   provider chain. Works with any OpenAI-compatible model (OpenAI,
   Together, Groq, Fireworks, LM Studio, custom gateways) by passing
   provider="openai" + a base_url override via config.

3. **Raw Anthropic SDK.** Project calls `anthropic.Anthropic().messages
   .create(...)`. Use `anthropic_messages_create`. Same idea.

All shims:
- Write telemetry to the local SQLite store on every call.
- Honor privacy_class gates (workloads marked PRIVATE never egress to
  providers that aren't themselves local).
- Return objects/dicts with the LEGACY shape, so call-site code doesn't
  need changes. The somm-native fields (call_id, cost_usd, tokens_in,
  outcome) are added as extra attributes/keys when it's safe to do so.

Usage:

    # Drop-in replacement — single line change in your project:
    # before:
    #   from mylib.llm import FooLLM
    # after:
    from somm.compat import GenericLLMCompat as FooLLM

    llm = FooLLM(project="myproject")
    result = llm.generate("hello", workload="greet")
    print(result.text, result.provider, result.model)
"""

from somm.compat.generic import GenericLLMCompat, LegacyLLMResult
from somm.compat.openai_compat import (
    OpenAIChatCompletion,
    openai_chat_completions,
)

__all__ = [
    "GenericLLMCompat",
    "LegacyLLMResult",
    "OpenAIChatCompletion",
    "openai_chat_completions",
]

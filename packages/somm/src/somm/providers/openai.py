"""OpenAI provider — uses the OpenAI-compatible base.

Works for api.openai.com out of the box. Can be repurposed for any
OpenAI-compatible gateway (Groq, Together, Fireworks, Azure OpenAI,
vLLM, LM Studio, a local lite-llm gateway, custom internal proxies)
by passing `base_url`.
"""

from __future__ import annotations

from somm.providers._openai_compat import OpenAICompatProvider


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

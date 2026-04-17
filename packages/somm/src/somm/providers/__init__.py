"""Provider adapters. Built-ins: ollama, openrouter (v0.2). Next: minimax, anthropic, openai.

Third parties register via entry-points group `somm.providers` using the
SommProvider Protocol (see base.py).
"""

from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommProvider,
    SommRequest,
    SommResponse,
)
from somm.providers.ollama import OllamaProvider
from somm.providers.openrouter import DEFAULT_FREE_ROSTER, OpenRouterProvider

__all__ = [
    "SommProvider",
    "SommRequest",
    "SommResponse",
    "SommChunk",
    "SommModel",
    "ProviderHealth",
    "OllamaProvider",
    "OpenRouterProvider",
    "DEFAULT_FREE_ROSTER",
]

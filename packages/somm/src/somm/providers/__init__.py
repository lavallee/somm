"""Provider adapters. Built-ins: ollama (v0.1). Future: openrouter, minimax, anthropic, openai.

Third parties register via entry-points group `somm.providers` using the
SommProvider Protocol (see base.py).
"""

from somm.providers.base import ProviderHealth, SommProvider, SommRequest, SommResponse
from somm.providers.ollama import OllamaProvider

__all__ = [
    "SommProvider",
    "SommRequest",
    "SommResponse",
    "ProviderHealth",
    "OllamaProvider",
]

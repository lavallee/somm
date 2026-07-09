"""Provider adapters and public provider protocol.

Built-ins are registered in ``somm.providers.registry.BUILTIN_PROVIDER_SPECS``.
Third parties register via the ``somm.providers`` entry-point group with a
``ProviderSpec``:

    [project.entry-points."somm.providers"]
    acme = "acme_somm:provider_spec"

The entry point resolves to either a ``ProviderSpec`` instance or a zero-arg
callable returning one. The spec factory receives ``(config, tracker)`` and
returns a ``SommProvider`` instance, or ``None`` when unavailable.
"""

from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommProvider,
    SommRequest,
    SommResponse,
)
from somm.providers.claude_cli import ClaudeCLIProvider
from somm.providers.codex_cli import CodexCLIProvider
from somm.providers.deepseek import DeepSeekProvider
from somm.providers.gemini import GeminiProvider
from somm.providers.minimax import MinimaxProvider
from somm.providers.ollama import OllamaProvider
from somm.providers.openai import OpenAIProvider
from somm.providers.openrouter import DEFAULT_FREE_ROSTER, OpenRouterProvider
from somm.providers.perplexity import PerplexityProvider

__all__ = [
    "SommProvider",
    "SommRequest",
    "SommResponse",
    "SommChunk",
    "SommModel",
    "ProviderHealth",
    "OpenAICompatProvider",
    "ClaudeCLIProvider",
    "CodexCLIProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    "OpenAIProvider",
    "MinimaxProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "DeepSeekProvider",
    "PerplexityProvider",
    "DEFAULT_FREE_ROSTER",
]

"""Provider registry for built-in and entry-point provider construction.

Third-party packages register provider specs through the ``somm.providers``
entry-point group. The entry point must resolve to either a ``ProviderSpec``
instance or a zero-argument callable returning one, for example:

    [project.entry-points."somm.providers"]
    acme = "acme_somm:provider_spec"

The spec factory is called as ``factory(config, tracker)`` and should return a
``SommProvider`` instance when configured, or ``None`` when unavailable.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.metadata import entry_points
from typing import Any

from somm_core.config import Config

from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import SommProvider
from somm.providers.claude_cli import ClaudeCLIProvider
from somm.providers.codex_cli import CodexCLIProvider
from somm.providers.deepseek import DeepSeekProvider
from somm.providers.gemini import GeminiProvider
from somm.providers.minimax import MinimaxProvider
from somm.providers.ollama import OllamaProvider
from somm.providers.openai import OpenAIProvider
from somm.providers.openrouter import OpenRouterProvider
from somm.providers.perplexity import PerplexityProvider

LOGGER = logging.getLogger("somm.providers")

type ProviderFactory = Callable[[Config, Any], SommProvider | None]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Provider construction contract used by built-ins and plugins.

    ``default_order_rank`` controls participation in the default routing
    chain. Lower ranks are earlier. ``None`` means the provider is configured
    and reachable through explicit order/full-provider surfaces, but is not
    added to the default routing order.
    """

    name: str
    factory: ProviderFactory
    default_order_rank: int | None


def _ollama(config: Config, tracker: Any) -> SommProvider | None:
    return OllamaProvider(
        base_url=config.ollama_url,
        default_model=config.ollama_model,
        enable_think=config.ollama_think,
        keep_alive=config.ollama_keep_alive,
    )


def _claude_cli(config: Config, tracker: Any) -> SommProvider | None:
    if not shutil.which("claude"):
        return None
    return ClaudeCLIProvider(
        default_model=config.claude_cli_model,
        timeout=max(config.http_timeout, 600.0),
        extra_args=config.claude_cli_extra_args,
    )


def _codex_cli(config: Config, tracker: Any) -> SommProvider | None:
    if not shutil.which("codex"):
        return None
    return CodexCLIProvider(timeout=max(config.http_timeout, 600.0))


def _openrouter(config: Config, tracker: Any) -> SommProvider | None:
    if not config.openrouter_api_key:
        return None
    return OpenRouterProvider(
        api_key=config.openrouter_api_key,
        roster=config.openrouter_roster,
        tracker=tracker,
    )


def _minimax(config: Config, tracker: Any) -> SommProvider | None:
    if not config.minimax_api_key:
        return None
    return MinimaxProvider(
        api_key=config.minimax_api_key,
        default_model=config.minimax_model,
        timeout=config.http_timeout,
    )


def _deepseek(config: Config, tracker: Any) -> SommProvider | None:
    if not config.deepseek_api_key:
        return None
    return DeepSeekProvider(
        api_key=config.deepseek_api_key,
        default_model=config.deepseek_model,
        timeout=config.http_timeout,
    )


def _anthropic(config: Config, tracker: Any) -> SommProvider | None:
    if not config.anthropic_api_key:
        return None
    return AnthropicProvider(
        api_key=config.anthropic_api_key,
        default_model=config.anthropic_model,
    )


def _openai(config: Config, tracker: Any) -> SommProvider | None:
    if not config.openai_api_key:
        return None
    return OpenAIProvider(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        default_model=config.openai_model,
        timeout=config.http_timeout,
    )


def _gemini(config: Config, tracker: Any) -> SommProvider | None:
    if not config.gemini_api_key:
        return None
    return GeminiProvider(
        api_key=config.gemini_api_key,
        default_model=config.gemini_model,
    )


def _perplexity(config: Config, tracker: Any) -> SommProvider | None:
    if not config.perplexity_api_key:
        return None
    return PerplexityProvider(
        api_key=config.perplexity_api_key,
        default_model=config.perplexity_model,
        timeout=max(config.http_timeout, 300.0),
    )


BUILTIN_PROVIDER_SPECS: list[ProviderSpec] = [
    # Hosted MiniMax is the first configured default. Ollama remains available
    # as a final local fallback or explicit pin, but no longer leads the chain.
    ProviderSpec("ollama", _ollama, 90),
    ProviderSpec("claude-cli", _claude_cli, None),
    ProviderSpec("codex-cli", _codex_cli, None),
    ProviderSpec("openrouter", _openrouter, 20),
    ProviderSpec("minimax", _minimax, 10),
    ProviderSpec("deepseek", _deepseek, 30),
    ProviderSpec("anthropic", _anthropic, 50),
    ProviderSpec("openai", _openai, 70),
    ProviderSpec("gemini", _gemini, 60),
    ProviderSpec("perplexity", _perplexity, 80),
]


def _coerce_entrypoint_spec(entry_point_name: str, loaded: object) -> ProviderSpec | None:
    candidate = loaded
    if not isinstance(candidate, ProviderSpec) and callable(candidate):
        candidate = candidate()
    if not isinstance(candidate, ProviderSpec):
        LOGGER.warning(
            "skipping somm.providers entry point %r: expected ProviderSpec, got %s",
            entry_point_name,
            type(candidate).__name__,
        )
        return None
    if not candidate.name or not callable(candidate.factory):
        LOGGER.warning(
            "skipping somm.providers entry point %r: invalid ProviderSpec %r",
            entry_point_name,
            candidate,
        )
        return None
    return candidate


@cache
def load_entrypoint_provider_specs() -> list[ProviderSpec]:
    """Load third-party ``somm.providers`` entry points.

    Broken entry points are logged and skipped so a plugin import error never
    prevents ``somm.llm()`` from building the core provider chain. Built-in
    provider names always win over third-party specs with the same name.
    """

    builtin_names = {spec.name for spec in BUILTIN_PROVIDER_SPECS}
    seen_plugin_names: set[str] = set()
    specs: list[ProviderSpec] = []
    for entry_point in entry_points(group="somm.providers"):
        try:
            spec = _coerce_entrypoint_spec(entry_point.name, entry_point.load())
        except Exception as exc:  # noqa: BLE001 - third-party imports must not break routing
            LOGGER.warning(
                "skipping somm.providers entry point %r: %s",
                entry_point.name,
                exc,
            )
            continue
        if spec is None:
            continue
        if spec.name in builtin_names:
            LOGGER.warning(
                "skipping somm.providers entry point %r: provider name %r is built in",
                entry_point.name,
                spec.name,
            )
            continue
        if spec.name in seen_plugin_names:
            LOGGER.warning(
                "skipping somm.providers entry point %r: provider name %r is already registered",
                entry_point.name,
                spec.name,
            )
            continue
        seen_plugin_names.add(spec.name)
        specs.append(spec)
    return specs


# Executors that never join the routing chain (``default_order_rank is None``)
# but must still be reachable by name. The seat providers are the case that
# motivated this: a caller pins ``provider="claude-cli"`` precisely to move a
# workload OFF the metered chain, so putting them IN the chain would defeat
# the point — yet leaving them unbuilt made the pin raise "not configured".
PINNED_ONLY_PROVIDER_NAMES: frozenset[str] = frozenset({"claude-cli", "codex-cli"})


def build_pinned_only_providers(
    config: Config,
    tracker: Any = None,
    *,
    exclude: set[str] | frozenset[str] | None = None,
) -> dict[str, SommProvider]:
    """Build the out-of-chain executors that are available on this machine.

    ``exclude`` drops names the caller already has in its chain, so a chain
    member always wins over a pinned-only copy — one provider instance per
    name, and the configured one is the one that answers.

    A factory returning ``None`` (binary not on PATH) is simply absent from
    the result: an unavailable executor should read as "not configured" at
    the pin, not fail at build time for every caller.
    """
    skip = set(exclude or ())
    built: dict[str, SommProvider] = {}
    for spec in BUILTIN_PROVIDER_SPECS:
        if spec.name not in PINNED_ONLY_PROVIDER_NAMES or spec.name in skip:
            continue
        try:
            provider = spec.factory(config, tracker)
        except Exception as exc:  # noqa: BLE001 - one bad executor must not break the rest
            LOGGER.warning("skipping pinned-only provider %r: %s", spec.name, exc)
            continue
        if provider is not None:
            built[spec.name] = provider
    return built

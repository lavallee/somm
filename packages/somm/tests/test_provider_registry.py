from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from somm.client import build_default_providers
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)
from somm.providers.registry import ProviderSpec
from somm_core.config import Config


class DummyProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, request: SommRequest) -> SommResponse:
        return SommResponse(
            text="ok",
            model="dummy",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        yield SommChunk(text="ok", done=True)

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def models(self) -> list[SommModel]:
        return []

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        return 1


def _names(providers) -> list[str]:
    return [provider.name for provider in providers]


def _configured() -> Config:
    config = Config()
    config.openrouter_api_key = "openrouter-key"
    config.minimax_api_key = "minimax-key"
    config.deepseek_api_key = "deepseek-key"
    config.anthropic_api_key = "anthropic-key"
    config.openai_api_key = "openai-key"
    config.gemini_api_key = "gemini-key"
    config.perplexity_api_key = "perplexity-key"
    return config


@pytest.fixture(autouse=True)
def no_entrypoint_providers(monkeypatch):
    monkeypatch.setattr("somm.client.load_entrypoint_provider_specs", lambda: [])


@pytest.fixture
def no_cli(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.shutil.which", lambda _name: None)


def test_default_config_is_ollama_only(no_cli):
    assert _names(build_default_providers(Config())) == ["ollama"]


def test_configured_default_chain_prefers_minimax_and_keeps_ollama_last(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.shutil.which", lambda _name: "/usr/bin/tool")

    assert _names(build_default_providers(_configured())) == [
        "minimax",
        "openrouter",
        "deepseek",
        "anthropic",
        "gemini",
        "openai",
        "perplexity",
        "ollama",
    ]


def test_provider_order_is_exclusive_and_keeps_requested_order(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.shutil.which", lambda _name: "/usr/bin/tool")
    config = _configured()
    config.provider_order = ["codex-cli", "minimax", "ollama", "missing", "claude-cli"]

    assert _names(build_default_providers(config)) == [
        "codex-cli",
        "minimax",
        "ollama",
        "claude-cli",
    ]


def test_full_includes_providers_outside_default_order(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.shutil.which", lambda _name: "/usr/bin/tool")

    assert _names(build_default_providers(_configured(), full=True)) == [
        "ollama",
        "claude-cli",
        "codex-cli",
        "openrouter",
        "minimax",
        "deepseek",
        "anthropic",
        "openai",
        "gemini",
        "perplexity",
    ]


def test_entrypoint_specs_are_full_and_order_available(monkeypatch, no_cli):
    specs = [
        ProviderSpec("ranked-plugin", lambda _config, _tracker: DummyProvider("ranked-plugin"), 10),
        ProviderSpec("explicit-plugin", lambda _config, _tracker: DummyProvider("explicit-plugin"), None),
    ]
    monkeypatch.setattr("somm.client.load_entrypoint_provider_specs", lambda: specs)

    assert _names(build_default_providers(Config())) == ["ollama", "ranked-plugin"]
    assert _names(build_default_providers(Config(), full=True)) == [
        "ollama",
        "ranked-plugin",
        "explicit-plugin",
    ]

    config = Config()
    config.provider_order = ["explicit-plugin"]
    assert _names(build_default_providers(config)) == ["explicit-plugin"]


def test_entrypoint_name_collision_cannot_replace_builtin(monkeypatch, caplog, no_cli):
    monkeypatch.setattr(
        "somm.client.load_entrypoint_provider_specs",
        lambda: [ProviderSpec("ollama", lambda _config, _tracker: DummyProvider("plugin-ollama"), 1)],
    )

    with caplog.at_level("WARNING", logger="somm.providers"):
        providers = build_default_providers(Config(), full=True)

    assert _names(providers) == ["ollama"]
    assert providers[0].__class__.__name__ == "OllamaProvider"
    assert "provider name is built in" in caplog.text


def test_broken_entrypoint_factories_are_skipped(monkeypatch, caplog, no_cli):
    def raises(_config: Config, _tracker: Any):
        raise RuntimeError("boom")

    specs = [
        ProviderSpec("raising-plugin", raises, 10),
        ProviderSpec("bad-shape-plugin", lambda _config, _tracker: object(), 20),
    ]
    monkeypatch.setattr("somm.client.load_entrypoint_provider_specs", lambda: specs)

    with caplog.at_level("WARNING", logger="somm.providers"):
        assert _names(build_default_providers(Config())) == ["ollama"]

    assert "factory failed" in caplog.text
    assert "non-SommProvider" in caplog.text


def test_entrypoint_provider_name_must_match_spec(monkeypatch, caplog, no_cli):
    """A spec named 'acme' whose factory returns a provider named 'ollama'
    must be skipped — otherwise it corrupts health/telemetry attribution and
    generate(provider='acme') lookup."""
    specs = [ProviderSpec("acme", lambda _c, _t: DummyProvider("ollama"), 90)]
    monkeypatch.setattr("somm.client.load_entrypoint_provider_specs", lambda: specs)

    with caplog.at_level("WARNING", logger="somm.providers"):
        providers = build_default_providers(Config(), full=True)

    names = [p.name for p in providers]
    assert names.count("ollama") == 1  # only the real built-in ollama
    assert "acme" not in names
    assert "named 'ollama'" in caplog.text or "named \"ollama\"" in caplog.text

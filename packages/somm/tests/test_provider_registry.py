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
    monkeypatch.setattr("somm.providers._cli_bin.shutil.which", lambda _name: None)
    monkeypatch.setattr("somm.providers._cli_bin.WELL_KNOWN_BIN_DIRS", ())


def test_default_config_is_ollama_only(no_cli):
    assert _names(build_default_providers(Config())) == ["ollama"]


def test_configured_default_chain_prefers_minimax_and_keeps_ollama_last(monkeypatch):
    monkeypatch.setattr("somm.providers._cli_bin.shutil.which", lambda _name: "/usr/bin/tool")

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
    monkeypatch.setattr("somm.providers._cli_bin.shutil.which", lambda _name: "/usr/bin/tool")
    config = _configured()
    config.provider_order = ["codex-cli", "minimax", "ollama", "missing", "claude-cli"]

    assert _names(build_default_providers(config)) == [
        "codex-cli",
        "minimax",
        "ollama",
        "claude-cli",
    ]


def test_full_includes_providers_outside_default_order(monkeypatch):
    monkeypatch.setattr("somm.providers._cli_bin.shutil.which", lambda _name: "/usr/bin/tool")

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


# ---------------------------------------------------------------------------
# Named OpenAI-compatible endpoints from configuration (SOMM_OPENAI_ENDPOINTS)


def test_endpoints_parse_and_bad_entries_are_skipped(capsys):
    from somm_core.config import parse_openai_endpoints

    parsed = parse_openai_endpoints(
        "box=http://10.0.0.5:8000/v1 model=served-m key_env=BOX_KEY, "
        "Bad Name=http://x, nourl=10.0.0.6, box=http://dup/v1, two=http://h:1/v1"
    )
    assert parsed == [
        {"name": "box", "base_url": "http://10.0.0.5:8000/v1", "model": "served-m",
         "key_env": "BOX_KEY"},
        {"name": "two", "base_url": "http://h:1/v1", "model": "", "key_env": ""},
    ]
    assert "skipping" in capsys.readouterr().err


def test_endpoints_load_from_the_environment(monkeypatch, tmp_path):
    from somm_core.config import load

    monkeypatch.setenv("SOMM_OPENAI_ENDPOINTS", "box=http://127.0.0.1:8000/v1 model=m")
    cfg = load(cwd=tmp_path)
    assert cfg.openai_endpoints[0]["name"] == "box"


def _with_endpoint(url: str = "http://10.0.0.5:8000/v1", **extra) -> Config:
    config = Config()
    config.openai_endpoints = [
        {"name": "box", "base_url": url, "model": "served-m", "key_env": "", **extra}
    ]
    return config


def test_endpoint_is_pin_only_unless_ordered():
    config = _with_endpoint()
    assert "box" not in _names(build_default_providers(config))
    full = {p.name: p for p in build_default_providers(config, full=True)}
    assert full["box"].base_url == "http://10.0.0.5:8000/v1"
    assert full["box"].default_model == "served-m"
    config.provider_order = ["box", "ollama"]
    assert _names(build_default_providers(config)) == ["box", "ollama"]


def test_endpoint_needs_its_named_key_and_cannot_take_a_builtin_name(monkeypatch):
    from somm.providers.registry import build_pinned_only_providers

    monkeypatch.delenv("BOX_KEY", raising=False)
    assert "box" not in build_pinned_only_providers(_with_endpoint(key_env="BOX_KEY"))
    monkeypatch.setenv("BOX_KEY", "k")
    assert build_pinned_only_providers(_with_endpoint(key_env="BOX_KEY"))["box"].api_key == "k"
    config = _with_endpoint()
    config.openai_endpoints[0]["name"] = "openai"
    assert "openai" not in _names(build_default_providers(config, full=True))


def test_endpoint_tier_follows_the_address():
    from somm.boundary import provider_tier
    from somm.providers.registry import build_pinned_only_providers

    local = build_pinned_only_providers(_with_endpoint())["box"]
    remote = build_pinned_only_providers(_with_endpoint("http://gpu.example.net/v1"))["box"]
    assert provider_tier(local).tier.value == "local"
    assert provider_tier(remote).tier.value == "unvetted"


def test_a_pinned_endpoint_call_is_recorded_unpriced_and_listed_in_health(
    tmp_path, monkeypatch
):
    from somm.client import SommLLM
    from somm.providers._openai_compat import OpenAICompatProvider

    def fake_generate(self, request):
        return SommResponse(text="hi", model=request.model or self.default_model,
                            tokens_in=5, tokens_out=2, latency_ms=1)

    monkeypatch.setattr(OpenAICompatProvider, "generate", fake_generate)
    config = _with_endpoint()
    config.project = "endpoint_test"
    config.db_dir = tmp_path / ".somm"
    config.spool_dir = config.db_dir / "spool"
    llm = SommLLM(config=config)
    try:
        result = llm.generate("ping", workload="w", provider="box")
        health = llm.health()
    finally:
        llm.close()
    assert result.text == "hi"
    with llm.repo._open() as conn:
        row = conn.execute(
            "SELECT provider, model, cost_source, data_tier FROM calls WHERE id = ?",
            (result.call_id,),
        ).fetchone()
    assert row == ("box", "served-m", "unknown:no-price", "local")
    assert health["box"]["configured"] and not health["box"]["in_chain"]

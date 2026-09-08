"""Pinned-only executors, reported cost, and Anthropic-family prompt tokens.

The CLI-seat executors (claude-cli / codex-cli) never join the routing
chain, but a caller that names one must still reach it; a seat call must
carry the cost the seat reported instead of $0; and an Anthropic-shaped
usage block must count the cached prompt, not just the uncached slice.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest
from somm import SommLLM, hooks
from somm.providers.base import ProviderHealth, SommRequest, SommResponse
from somm.providers.registry import build_pinned_only_providers
from somm_core.config import Config, load
from somm_core.parse import anthropic_prompt_tokens


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="ok", model=request.model or "fake-m", tokens_in=3, tokens_out=2, latency_ms=5
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "cli_pin_test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


@pytest.fixture(autouse=True)
def _no_entrypoint_providers(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.load_entrypoint_provider_specs", lambda: [])
    hooks.set_call_site_provider(None)
    yield
    hooks.set_call_site_provider(None)


@pytest.fixture
def cli_on_path(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.shutil.which", lambda name: f"/usr/bin/{name}")


# -- registry ------------------------------------------------------------------


def test_pinned_only_build_returns_the_executors_when_binaries_exist(cli_on_path):
    built = build_pinned_only_providers(Config())
    assert set(built) == {"claude-cli", "codex-cli"}


def test_pinned_only_build_skips_names_already_in_the_chain(cli_on_path):
    built = build_pinned_only_providers(Config(), exclude={"claude-cli"})
    assert set(built) == {"codex-cli"}


def test_pinned_only_build_is_empty_without_binaries(monkeypatch):
    monkeypatch.setattr("somm.providers.registry.shutil.which", lambda _name: None)
    assert build_pinned_only_providers(Config()) == {}


def test_registry_passes_claude_cli_config_through(cli_on_path):
    cfg = Config()
    cfg.claude_cli_model = "claude-opus-4-7"
    cfg.claude_cli_extra_args = ["--tools", ""]
    provider = build_pinned_only_providers(cfg)["claude-cli"]
    assert provider.default_model == "claude-opus-4-7"
    assert provider.extra_args == ["--tools", ""]
    assert [m.name for m in provider.models()] == ["claude-opus-4-7"]


# -- client pin resolution -----------------------------------------------------


def test_pin_reaches_an_executor_outside_the_chain(tmp_path, cli_on_path):
    llm = SommLLM(providers=[FakeProvider()], config=_tmp_config(tmp_path))
    assert [p.name for p in llm.providers] == ["fake"]  # chain untouched
    assert llm._pick_provider("claude-cli").name == "claude-cli"
    assert [p.name for p in llm.all_providers()] == ["fake", "claude-cli", "codex-cli"]


def test_unknown_pin_still_fails_loudly(tmp_path, cli_on_path):
    llm = SommLLM(providers=[FakeProvider()], config=_tmp_config(tmp_path))
    with pytest.raises(ValueError, match="not configured"):
        llm._pick_provider("nope")


def test_chain_member_wins_over_pinned_only_copy(tmp_path, cli_on_path):
    class ChainClaude(FakeProvider):
        name = "claude-cli"

    chain = ChainClaude()
    llm = SommLLM(providers=[chain], config=_tmp_config(tmp_path))
    assert llm._pick_provider("claude-cli") is chain
    assert "claude-cli" not in llm._pinned_only_providers()


# -- reported cost + call site on generate() ----------------------------------


class ReportingProvider(FakeProvider):
    name = "seat"

    def generate(self, request):
        return SommResponse(
            text="ok",
            model="seat-m",
            tokens_in=10,
            tokens_out=2,
            latency_ms=5,
            raw={"total_cost_usd": 0.0421},
            cost_usd=0.0421,
        )


def test_generate_records_a_provider_reported_cost(tmp_path):
    llm = SommLLM(providers=[ReportingProvider()], config=_tmp_config(tmp_path))
    result = llm.generate("hi", workload="w", provider="seat")
    llm.close()
    assert result.cost_usd == pytest.approx(0.0421)
    row = llm.repo.get_call(result.call_id)
    assert row is not None
    assert row.cost_usd == pytest.approx(0.0421)
    assert row.cost_basis == "reported"
    assert row.cost_source == "provider:seat"


def test_generate_records_the_call_site(tmp_path):
    llm = SommLLM(providers=[FakeProvider()], config=_tmp_config(tmp_path))
    result, line = llm.generate("hi", workload="w"), inspect.currentframe().f_lineno
    llm.close()
    row = llm.repo.get_call(result.call_id)
    assert row is not None
    assert row.call_site is not None
    assert row.call_site.endswith(f"test_cli_executor_pins.py:{line}")


# -- adapters ------------------------------------------------------------------


def test_anthropic_prompt_tokens_counts_the_cached_prompt():
    usage = {
        "input_tokens": 3,
        "cache_read_input_tokens": 2303,
        "cache_creation_input_tokens": 4123,
        "output_tokens": 5,
    }
    assert anthropic_prompt_tokens(usage) == 6429
    assert anthropic_prompt_tokens({"input_tokens": 7}) == 7
    assert anthropic_prompt_tokens(None) == 0
    assert anthropic_prompt_tokens({"input_tokens": "junk"}) == 0


def test_claude_cli_reports_whole_prompt_and_seat_cost(monkeypatch):
    from somm.providers.claude_cli import ClaudeCLIProvider

    envelope = (
        '{"result":"pong","duration_ms":900,"total_cost_usd":0.0265,'
        '"usage":{"input_tokens":3,"cache_read_input_tokens":2303,'
        '"cache_creation_input_tokens":4123,"output_tokens":5},'
        '"modelUsage":{"claude-sonnet-4-6":{"outputTokens":5}}}'
    )

    def fake_run(cmd, *, input, capture_output, text, timeout, cwd, env):
        assert cmd[-2:] == ["--tools", ""]
        return subprocess.CompletedProcess(cmd, 0, stdout=envelope, stderr="")

    monkeypatch.setattr("somm.providers.claude_cli.subprocess.run", fake_run)
    provider = ClaudeCLIProvider(binary="claude", extra_args=["--tools", ""])
    resp = provider.generate(SommRequest(prompt="ping"))
    assert resp.tokens_in == 6429
    assert resp.tokens_out == 5
    assert resp.cost_usd == pytest.approx(0.0265)
    assert resp.model == "claude-sonnet-4-6"


def test_claude_cli_without_cost_in_envelope_leaves_cost_unset(monkeypatch):
    from somm.providers.claude_cli import ClaudeCLIProvider

    def fake_run(cmd, *, input, capture_output, text, timeout, cwd, env):
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"result":"ok","usage":{"input_tokens":1,"output_tokens":1}}', stderr=""
        )

    monkeypatch.setattr("somm.providers.claude_cli.subprocess.run", fake_run)
    resp = ClaudeCLIProvider(binary="claude").generate(SommRequest(prompt="hi"))
    assert resp.cost_usd is None
    assert resp.tokens_in == 1


# -- config ---------------------------------------------------------------------


def test_claude_cli_config_from_pyproject(tmp_path, monkeypatch):
    monkeypatch.delenv("SOMM_CLAUDE_CLI_MODEL", raising=False)
    monkeypatch.delenv("SOMM_CLAUDE_CLI_ARGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "pyproject.toml").write_text(
        "[tool.somm]\nproject = 'p'\nclaude_cli_model = 'claude-opus-4-7'\n"
        "claude_cli_extra_args = ['--tools', '', '--no-session-persistence']\n",
        encoding="utf-8",
    )
    (tmp_path / ".somm").mkdir()
    cfg = load(cwd=tmp_path)
    assert cfg.claude_cli_model == "claude-opus-4-7"
    assert cfg.claude_cli_extra_args == ["--tools", "", "--no-session-persistence"]


def test_claude_cli_config_from_env_is_shell_split(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SOMM_CLAUDE_CLI_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("SOMM_CLAUDE_CLI_ARGS", "--tools '' --setting-sources project")
    cfg = load(cwd=tmp_path)
    assert cfg.claude_cli_model == "claude-sonnet-4-6"
    assert cfg.claude_cli_extra_args == ["--tools", "", "--setting-sources", "project"]

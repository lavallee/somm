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


# -- cross-provider consistency -------------------------------------------------
#
# The reason the API provider gets the same treatment as the CLI seat: somm
# exists to compare providers, and it cannot if the same prompt reports a
# different size depending on who served it.


def test_anthropic_api_reports_the_whole_prompt_like_the_openai_family(monkeypatch):
    """Anthropic splits the prompt three ways; OpenAI reports one inclusive number.

    `prompt_tokens` on an OpenAI-compatible response already includes the
    cached portion (`prompt_tokens_details.cached_tokens` is a subset of it).
    Reading Anthropic's `input_tokens` alone reported only the uncached slice,
    so a cached call looked ~2000x smaller on one provider than the other.
    """
    from somm.providers.anthropic import AnthropicProvider

    anthropic_usage = {
        "input_tokens": 3,
        "cache_read_input_tokens": 2303,
        "cache_creation_input_tokens": 4123,
        "output_tokens": 5,
    }
    # What an OpenAI-compatible provider would report for the same prompt.
    openai_prompt_tokens = 3 + 2303 + 4123

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-x",
                "usage": anthropic_usage,
            }

        @staticmethod
        def raise_for_status():
            return None

        text = ""

    monkeypatch.setattr(
        "somm.providers.anthropic.httpx.Client.post",
        lambda *a, **kw: FakeResponse(),
    )
    provider = AnthropicProvider(api_key="k", default_model="claude-x")
    resp = provider.generate(SommRequest(prompt="hi"))

    assert resp.tokens_in == openai_prompt_tokens
    assert resp.tokens_out == 5


def test_the_cache_split_is_still_recorded_separately():
    """Folding cache tokens into tokens_in must not lose the breakdown.

    cost accuracy work needs the split; extract_cache_tokens is where it
    lives, and it reads the raw payload, not tokens_in.
    """
    from somm_core.parse import extract_cache_tokens

    raw = {"usage": {
        "input_tokens": 3,
        "cache_read_input_tokens": 2303,
        "cache_creation_input_tokens": 4123,
        "output_tokens": 5,
    }}
    assert extract_cache_tokens(raw) == (2303, 4123)


# -- gaps the original spec did not cover --------------------------------------
#
# Found auditing the reconstruction rather than from a failing test. Recorded
# here so they cannot come back silently.


def test_reported_cost_is_captured_when_a_seat_is_reached_through_the_chain(tmp_path):
    """The README documents SOMM_PROVIDER_ORDER as a way to reach the seats.

    The original spec only pinned `provider="seat"`, so the chain path could
    have dropped a provider-reported cost and no test would have noticed.
    """
    llm = SommLLM(providers=[ReportingProvider()], config=_tmp_config(tmp_path))
    result = llm.generate("hi", workload="w")  # no pin — routed through the chain
    llm.close()

    assert result.cost_usd == pytest.approx(0.0421)
    row = llm.repo.get_call(result.call_id)
    assert row is not None
    assert row.cost_basis == "reported"
    assert row.cost_source == "provider:seat"


def test_a_failed_pin_does_not_record_a_cost_it_never_incurred(tmp_path, cli_on_path):
    """cost_usd_out must stay unset when generate() raises."""

    class ExplodingProvider(FakeProvider):
        name = "boom"

        def generate(self, request):
            raise RuntimeError("upstream is down")

    llm = SommLLM(providers=[ExplodingProvider()], config=_tmp_config(tmp_path))
    result = None
    try:
        result = llm.generate("hi", workload="w", provider="boom")
    except Exception:
        pass
    llm.close()

    assert result is not None, "a failed pinned call is still recorded"
    row = llm.repo.get_call(result.call_id)
    assert row is not None
    assert row.cost_basis != "reported"
    assert row.cost_source != "provider:boom"


def test_pinned_only_providers_get_the_real_health_tracker(tmp_path, cli_on_path):
    """A pinned-only build must receive the same tracker the chain uses.

    The reconstruction passed `getattr(self, "tracker", None)` — an attribute
    that does not exist — so every pinned-only factory silently got None.
    Harmless for the CLI seats, which ignore it, and exactly the kind of thing
    that bites the first provider that does not.
    """
    seen: list[object] = []

    def spy(config, tracker):
        seen.append(tracker)
        return FakeProvider()

    import somm.providers.registry as reg

    original = reg.BUILTIN_PROVIDER_SPECS
    monkey = [reg.ProviderSpec("claude-cli", spy, None)]
    reg.BUILTIN_PROVIDER_SPECS = monkey
    try:
        llm = SommLLM(providers=[FakeProvider()], config=_tmp_config(tmp_path))
        llm._pinned_only_providers()
        llm.close()
    finally:
        reg.BUILTIN_PROVIDER_SPECS = original

    assert seen, "the pinned-only factory should have been called"
    assert seen[0] is not None, "pinned-only providers were built with no tracker"


def test_pinned_only_construction_happens_once(tmp_path, cli_on_path):
    """_pick_provider runs on every call; building executors shells out."""
    llm = SommLLM(providers=[FakeProvider()], config=_tmp_config(tmp_path))
    first = llm._pinned_only_providers()
    second = llm._pinned_only_providers()
    llm.close()
    assert first is second

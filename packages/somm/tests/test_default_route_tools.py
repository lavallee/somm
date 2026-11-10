"""Regression coverage for new-project defaults and explicit CLI text generation."""
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest
from somm.cli import main
from somm.errors import SommBadRequest
from somm.providers.base import SommRequest
from somm.providers.claude_cli import ClaudeCLIProvider
from somm.providers.ollama import OllamaProvider
from somm_core.config import Config
from somm_core.models import Outcome, SommResult


def test_missing_default_model_names_fix():
    provider = OllamaProvider()
    transport = httpx.MockTransport(lambda req: httpx.Response(404, json={"error": "not found"}))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(SommBadRequest) as caught:
            provider._post(client, "/api/chat", {}, Config().ollama_model)
    detail = str(caught.value)
    assert "qwen3:8b" in detail
    assert "ollama pull qwen3:8b" in detail
    assert "SOMM_OLLAMA_MODEL" in detail
    assert "[tool.somm].ollama_model" in detail


@pytest.mark.parametrize("text_only", [False, True])
def test_claude_text_only_argv(monkeypatch, text_only):
    def run(cmd, **kwargs):
        assert cmd == ["claude", "-p", "--output-format", "json", "--model", "claude-haiku-5-5",
                       "--tools", "", "--strict-mcp-config", "--setting-sources", "",
                       "--no-session-persistence"]
        assert kwargs["input"] == "untrusted input"
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": "answer"}), "")
    monkeypatch.setattr("somm.providers.claude_cli.subprocess.run", run)
    ClaudeCLIProvider().generate(SommRequest(prompt="untrusted input", claude_cli_text_only=text_only))


def test_text_only_rejects_extra_args():
    with pytest.raises(SommBadRequest, match="extra_args"):
        ClaudeCLIProvider(extra_args=["--tools", "Bash"]).generate(
            SommRequest(prompt="hi", claude_cli_text_only=True)
        )


@pytest.mark.parametrize("outcome", list(Outcome))
@pytest.mark.parametrize("as_json", [False, True])
def test_generate_exit_status(monkeypatch, capsys, outcome, as_json):
    result = SommResult(text="answer" if outcome == Outcome.OK else "", provider="fake",
                        model="fake", tokens_in=1, tokens_out=1, latency_ms=1,
                        cost_usd=0, call_id="test", outcome=outcome)
    monkeypatch.setattr("somm.cli._common.load_config", lambda **kw: Config())
    monkeypatch.setattr("somm.client.SommLLM", lambda **kw: SimpleNamespace(
        generate=lambda *a, **kw: result, close=lambda: None))
    rc = main(["generate", "hello"] + (["--json"] if as_json else []))
    assert (rc == 0) == (outcome == Outcome.OK)
    if as_json:
        assert json.loads(capsys.readouterr().out)["ok"] == (rc == 0)


def test_cli_text_only_forwarding(monkeypatch):
    def generate(*a, **kwargs):
        assert kwargs["claude_cli_text_only"] is True
        assert kwargs["allow_fallback"] is False
        raise ValueError("sentinel")
    monkeypatch.setattr("somm.cli._common.load_config", lambda **kw: Config())
    monkeypatch.setattr("somm.client.SommLLM", lambda **kw: SimpleNamespace(
        generate=generate, close=lambda: None))
    assert main(["generate", "hi", "--provider", "claude-cli",
                 "--claude-cli-text-only", "--json"]) == 2


def test_api_text_only_reaches_provider(tmp_path, monkeypatch):
    from somm import SommLLM
    from somm.providers.base import SommResponse

    cfg = Config(db_dir=tmp_path, spool_dir=tmp_path / "spool")
    with SommLLM(config=cfg) as llm:
        def generate(request):
            assert request.claude_cli_text_only is True
            return SommResponse(text="ok", model="fake", tokens_in=1, tokens_out=1, latency_ms=1)
        monkeypatch.setattr(llm, "_pinned_only_cache",
                            {"claude-cli": SimpleNamespace(name="claude-cli", generate=generate)})
        result = llm.generate("hi", provider="claude-cli",
                              claude_cli_text_only=True, allow_fallback=False)
        assert result.text == "ok"
        for provider, fallback in [(None, False), ("ollama", False), ("claude-cli", True)]:
            with pytest.raises(ValueError, match="requires"):
                llm.generate("hi", provider=provider, allow_fallback=fallback,
                             claude_cli_text_only=True)

"""Public agent-harness execution contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from somm import harnesses
from somm.harnesses import HarnessCapabilityError, HarnessOutcome, HarnessRequest
from somm.harnesses import codex as codex_module
from somm.harnesses import pi as pi_module


def _request(tmp_path: Path, **overrides) -> HarnessRequest:
    values = {
        "prompt": "do it",
        "cwd": tmp_path,
        "capture_dir": tmp_path / "capture",
        "allow_unsafe": True,
    }
    values.update(overrides)
    return HarnessRequest(**values)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_registry_and_unknown_harness() -> None:
    assert harnesses.get("codex") is harnesses.get("codex-cli")
    assert harnesses.get("claude-cli").capabilities.max_turns is True
    assert harnesses.get("codex").capabilities.reasoning_effort is True
    assert harnesses.get("opencode").capabilities.agent_selection is True
    assert harnesses.get("pi").capabilities.reasoning_effort is True
    with pytest.raises(ValueError, match="unknown harness"):
        harnesses.get("missing")


@pytest.mark.parametrize(
    ("harness", "model"),
    [
        ("claude-cli", None),
        ("claude-cli", "sonnet"),
        ("claude-cli", "claude-sonnet-4-5"),
        ("claude-cli", "anthropic/claude-opus-4-1"),
        ("codex", None),
        ("codex", "gpt-5.6"),
        ("codex", "openai/gpt-5.6"),
        ("codex", "o3"),
        ("codex", "future-model"),
        ("opencode", "anthropic/claude-sonnet-4-5"),
        ("opencode", "gpt-5.6"),
        ("pi", "anthropic/claude-sonnet-4-5"),
        ("pi", "gpt-5.6"),
    ],
)
def test_model_family_validation_accepts_compatible_or_unknown_models(
    harness: str, model: str | None
) -> None:
    harnesses.validate_model(harness, model)


@pytest.mark.parametrize(
    ("harness", "model", "family"),
    [
        ("codex", "sonnet", "anthropic"),
        ("codex", "claude-sonnet-4-5", "anthropic"),
        ("claude-cli", "gpt-5.6", "openai"),
        ("claude-cli", "o3", "openai"),
    ],
)
def test_model_family_validation_rejects_cross_provider_models(
    harness: str, model: str, family: str
) -> None:
    with pytest.raises(HarnessCapabilityError, match=family):
        harnesses.validate_model(harness, model)


@pytest.mark.parametrize(
    ("model", "family"),
    [
        (None, None),
        ("", None),
        ("sonnet-4-5", "anthropic"),
        ("claude-opus-4-1", "anthropic"),
        ("anthropic/claude-haiku-4-5", "anthropic"),
        ("gpt-5.6", "openai"),
        ("chatgpt-5.6", "openai"),
        ("codex-mini", "openai"),
        ("openai/gpt-5.6", "openai"),
        ("o3-mini", "openai"),
        ("future-model", None),
    ],
)
def test_infer_model_family(model: str | None, family: str | None) -> None:
    assert harnesses.infer_model_family(model) == family


def test_start_rejects_incompatible_model_before_adapter_launch(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = harnesses.get("codex")
    launched = False

    def fake_start(request):
        nonlocal launched
        launched = True
        return object()

    monkeypatch.setattr(adapter, "start", fake_start)

    with pytest.raises(HarnessCapabilityError, match="anthropic"):
        harnesses.start("codex", _request(tmp_path, model="sonnet"))

    assert launched is False


def test_safe_mode_does_not_bypass_permissions(tmp_path: Path) -> None:
    request = _request(tmp_path, allow_unsafe=False)
    assert "--permission-mode" not in harnesses.get("claude-cli").build_argv(request)
    assert "--dangerously-bypass-approvals-and-sandbox" not in (
        harnesses.get("codex").build_argv(request)
    )
    assert "--dangerously-skip-permissions" not in (
        harnesses.get("opencode").build_argv(request)
    )
    pi_argv = harnesses.get("pi").build_argv(request)
    assert "--no-tools" in pi_argv
    assert "--no-context-files" in pi_argv
    assert "--no-approve" in pi_argv


def test_claude_argv_and_result(tmp_path: Path) -> None:
    adapter = harnesses.get("claude-cli")
    request = _request(
        tmp_path,
        model="sonnet",
        reasoning_effort="high",
        max_turns=20,
        session_id="sess-1",
        correlation_id="job-1",
        executable="/opt/claude",
    )
    argv = adapter.build_argv(request)
    assert argv[:3] == ["/opt/claude", "-p", "do it"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--max-turns") + 1] == "20"
    assert argv[argv.index("--resume") + 1] == "sess-1"
    assert "--permission-mode" in argv

    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "system", "subtype": "init", "session_id": "sess-1",
         "rateLimitInfo": {"status": "allowed"}},
        {"type": "result", "subtype": "success", "is_error": False,
         "session_id": "sess-1", "result": "Done.",
         "usage": {"input_tokens": 10, "output_tokens": 2}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    result = adapter.inspect(stdout, stderr, exit_code=0, correlation_id="job-1")
    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.final_text == "Done."
    assert result.session_id == "sess-1"
    assert result.usage["input_tokens"] == 10
    assert result.correlation_id == "job-1"


def test_claude_max_turns_and_real_rate_limit(tmp_path: Path) -> None:
    adapter = harnesses.get("claude-cli")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "result", "subtype": "error_max_turns", "is_error": True},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.TURN_LIMIT

    stdout.write_text(_stream(
        {"type": "result", "subtype": "error_during_execution", "is_error": True},
    ))
    stderr.write_text("HTTPStatusError: http_status=429 retry-after: 12")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.RATE_LIMIT


def test_codex_argv_resume_and_result(tmp_path: Path) -> None:
    adapter = harnesses.get("codex")
    request = _request(
        tmp_path,
        model="gpt-5.6",
        reasoning_effort="high",
        session_id="thread-1",
        executable="/opt/codex",
    )
    argv = adapter.build_argv(request)
    assert argv[:3] == ["/opt/codex", "exec", "resume"]
    assert argv[-2:] == ["thread-1", "do it"]
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert argv[argv.index("--config") + 1] == 'model_reasoning_effort="high"'

    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": "Shipped.",
        }},
        {"type": "turn.completed", "usage": {"input_tokens": 11}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    result = adapter.inspect(stdout, stderr, exit_code=0)
    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.final_text == "Shipped."
    assert result.session_id == "thread-1"
    assert result.usage["input_tokens"] == 11


def test_codex_argv_can_read_prompt_from_stdin(tmp_path: Path) -> None:
    adapter = harnesses.get("codex")
    request = _request(tmp_path, prompt="a large evidence packet", prompt_via_stdin=True)

    argv = adapter.build_argv(request)

    assert argv[-1] == "-"
    assert "a large evidence packet" not in argv


def test_codex_start_passes_prompt_as_stdin(tmp_path: Path, monkeypatch) -> None:
    adapter = harnesses.get("codex")
    request = _request(tmp_path, prompt="large packet", prompt_via_stdin=True)
    seen = {}

    def fake_launch(argv, req, *, stdin_data=None):
        seen.update(argv=argv, request=req, stdin_data=stdin_data)
        return object()

    monkeypatch.setattr(codex_module, "launch_process", fake_launch)

    adapter.start(request)

    assert seen["argv"][-1] == "-"
    assert seen["request"] is request
    assert seen["stdin_data"] == "large packet"


def test_codex_failed_event_normalizes_auth(tmp_path: Path) -> None:
    adapter = harnesses.get("codex")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "turn.failed", "error": {"message": "authentication failed http_status=401"}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.AUTH_ERROR


def test_codex_failed_event_normalizes_exact_cyber_safety_notice(tmp_path: Path) -> None:
    adapter = harnesses.get("codex")
    message = (
        "This content was flagged for possible cybersecurity risk. "
        "If this seems wrong, try rephrasing your request."
    )
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "turn.failed", "error": {"message": message}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.REFUSED


def test_codex_completed_event_may_quote_cyber_safety_notice(tmp_path: Path) -> None:
    adapter = harnesses.get("codex")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "item.completed", "item": {
            "type": "agent_message",
            "text": "The log said content was flagged for possible cybersecurity risk.",
        }},
        {"type": "turn.completed", "usage": {"input_tokens": 12}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.COMPLETED


def test_opencode_argv_and_result(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    request = _request(
        tmp_path,
        model="anthropic/claude-sonnet-4-5",
        agent="build",
        session_id="ses-1",
    )
    argv = adapter.build_argv(request)
    assert argv[:2] == ["opencode", "run"]
    assert argv[argv.index("--session") + 1] == "ses-1"
    assert argv[argv.index("--agent") + 1] == "build"
    assert argv[-1] == "do it"

    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_start", "sessionID": "ses-1"},
        {"type": "text", "sessionID": "ses-1", "part": {"text": "Hello "}},
        {"type": "text", "sessionID": "ses-1", "part": {"text": "world."}},
        {"type": "step_finish", "sessionID": "ses-1", "part": {
            "reason": "stop", "tokens": {"input": 6, "output": 15},
        }},
    ))
    stderr = _write(tmp_path, "stderr", "")
    result = adapter.inspect(stdout, stderr, exit_code=0)
    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.final_text == "Hello world."
    assert result.session_id == "ses-1"
    assert result.usage == {"input": 6, "output": 15}


def test_opencode_length_and_context_failure(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {"reason": "length"}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.TURN_LIMIT

    stdout.write_text("")
    stderr.write_text("request rejected: context window exceeded")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.CONTEXT_LIMIT


def test_pi_safe_argv_uses_stdin_and_blocks_ambient_resources(tmp_path: Path) -> None:
    adapter = harnesses.get("pi")
    request = _request(
        tmp_path,
        allow_unsafe=False,
        prompt_via_stdin=True,
        model="anthropic/claude-sonnet-4-5",
        reasoning_effort="high",
        executable="/opt/pi",
    )

    argv = adapter.build_argv(request)

    assert argv[:3] == ["/opt/pi", "--mode", "json"]
    for flag in (
        "--no-tools",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-approve",
        "--no-session",
    ):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet-4-5"
    assert argv[argv.index("--thinking") + 1] == "high"
    assert "do it" not in argv


def test_pi_safe_mode_rejects_extra_cli_arguments(tmp_path: Path) -> None:
    request = _request(tmp_path, allow_unsafe=False, extra=("--tools", "bash"))

    with pytest.raises(HarnessCapabilityError, match="does not accept extra"):
        harnesses.get("pi").build_argv(request)


def test_pi_start_transports_prompt_over_stdin(tmp_path: Path, monkeypatch) -> None:
    adapter = harnesses.get("pi")
    request = _request(
        tmp_path,
        allow_unsafe=False,
        prompt="private workload prompt",
        prompt_via_stdin=True,
    )
    seen = {}

    def fake_launch(argv, req, *, stdin_data=None):
        seen.update(argv=argv, request=req, stdin_data=stdin_data)
        return object()

    monkeypatch.setattr(pi_module, "launch_process", fake_launch)

    adapter.start(request)

    assert seen["request"] is request
    assert seen["stdin_data"] == "private workload prompt"
    assert "private workload prompt" not in seen["argv"]


def test_pi_result_uses_authoritative_assistant_message(tmp_path: Path) -> None:
    adapter = harnesses.get("pi")
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Probe passed."}],
        "stopReason": "stop",
        "usage": {"input": 11, "output": 3, "cost": {"total": 0.01}},
    }
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "session", "version": 3, "id": "pi-session-1"},
        {"type": "message_end", "message": message},
        {"type": "agent_end", "messages": [message]},
    ))
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0, correlation_id="attempt-1")

    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.final_text == "Probe passed."
    assert result.session_id == "pi-session-1"
    assert result.usage == {"input": 11, "output": 3}
    assert result.correlation_id == "attempt-1"


def test_pi_error_and_length_are_not_false_successes(tmp_path: Path) -> None:
    adapter = harnesses.get("pi")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "session", "id": "pi-session-2"},
        {"type": "agent_end", "messages": [{
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": "authentication failed http_status=401",
        }]},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.AUTH_ERROR

    stdout.write_text(_stream(
        {"type": "agent_end", "messages": [{
            "role": "assistant",
            "content": [{"type": "text", "text": "truncated"}],
            "stopReason": "length",
        }]},
    ))
    assert adapter.inspect(stdout, stderr).outcome is HarnessOutcome.CONTEXT_LIMIT

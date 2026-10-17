"""Public agent-harness execution contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from somm import harnesses
from somm.harnesses import HarnessCapabilityError, HarnessOutcome, HarnessRequest
from somm.harnesses import base as base_module


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


def test_opencode_insufficient_credits_is_budget_exhausted(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(
        tmp_path,
        "stdout",
        _stream(
            {
                "type": "error",
                "sessionID": "ses_credit",
                "error": {
                    "name": "APIError",
                    "data": {
                        "message": "Insufficient credits.",
                        "statusCode": 402,
                        "isRetryable": False,
                    },
                },
            }
        ),
    )
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=1)

    assert result.outcome is HarnessOutcome.BUDGET_EXHAUSTED
    assert result.session_id == "ses_credit"


def test_codex_argv_resume_and_result(tmp_path: Path) -> None:
    adapter = harnesses.get("codex")
    request = _request(
        tmp_path,
        model="gpt-5.6",
        reasoning_effort="high",
        session_id="thread-1",
        executable="/opt/codex",
        allow_unsafe=False,
    )
    argv = adapter.build_argv(request)
    assert argv[:2] == ["/opt/codex", "exec"]
    assert argv.index("--cd") < argv.index("resume")
    assert argv[argv.index("--cd") + 1] == str(tmp_path.resolve())
    assert argv[-2:] == ["thread-1", "do it"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
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
    request = _request(tmp_path, prompt="a large evidence packet", prompt_via_stdin=True, allow_unsafe=False)

    argv = adapter.build_argv(request)

    assert argv[-1] == "-"
    assert "a large evidence packet" not in argv


def test_codex_start_passes_prompt_as_stdin(tmp_path: Path, monkeypatch) -> None:
    adapter = harnesses.get("codex")
    request = _request(tmp_path, prompt="large packet", prompt_via_stdin=True, allow_unsafe=False)
    seen = {}

    def fake_launch(argv, req, *, stdin_data=None, **kwargs):
        seen.update(argv=argv, request=req, stdin_data=stdin_data, **kwargs)
        return object()

    monkeypatch.setattr(base_module, "launch_process", fake_launch)

    adapter.start(request)

    assert seen["argv"][-1] == "-"
    assert seen["request"].cwd == str(tmp_path.resolve())
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


def test_opencode_interrupted_tool_call_reason_is_failure(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "text", "sessionID": "ses-1", "part": {"text": "partial"}},
        {"type": "step_finish", "sessionID": "ses-1", "part": {
            "reason": "tool-calls", "tokens": {"input": 5, "output": 1},
        }},
    ))
    stderr = _write(
        tmp_path,
        "stderr",
        "permission request auto-rejected for external directory\n",
    )

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.outcome is HarnessOutcome.FAILED
    assert "tool-calls" in result.detail


def test_opencode_tool_calls_then_stop_completes(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {"reason": "tool-calls"}},
        {"type": "step_finish", "part": {"reason": "tool-calls"}},
        {"type": "text", "sessionID": "ses-1", "part": {"text": "Final answer."}},
        {"type": "step_finish", "sessionID": "ses-1", "part": {"reason": "stop"}},
    ))
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.final_text == "Final answer."


def test_opencode_sums_step_costs_but_keeps_terminal_usage(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {
            "reason": "tool", "tokens": {
                "total": 9, "input": 3, "output": 1, "reasoning": 1,
                "cache": {"read": 4, "write": 0},
            }, "cost": 0.01,
        }},
        {"type": "step_finish", "part": {
            "reason": "tool", "tokens": {
                "total": 13, "input": 4, "output": 2, "reasoning": 1,
                "cache": {"read": 6, "write": 0},
            }, "cost": 0.02,
        }},
        {"type": "step_finish", "sessionID": "ses-1", "part": {
            "reason": "stop", "tokens": {
                "total": 30, "input": 6, "output": 15, "reasoning": 2,
                "cache": {"read": 7, "write": 0},
            }, "cost": 0.03,
        }},
    ))
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.cost_usd == pytest.approx(0.06)
    assert result.usage == {
        "total": 30,
        "input": 6,
        "output": 15,
        "reasoning": 2,
        "cache": {"read": 7, "write": 0},
    }
    assert result.billing_usage == {
        "steps": 3,
        "total": 52,
        "input": 13,
        "output": 18,
        "reasoning": 4,
        "cache": {"read": 17, "write": 0},
    }


def test_opencode_billing_usage_ignores_invalid_values(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {"reason": "tool", "tokens": {
            "input": -1, "output": True, "reasoning": "2",
            "cache": {"read": float("inf"), "write": 3},
        }}},
        {"type": "step_finish", "part": {"reason": "stop", "tokens": {
            "input": 5, "output": 2, "cache": {"read": 7},
        }}},
    ))
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.billing_usage == {
        "steps": 2,
        "total": 0,
        "input": 5,
        "output": 2,
        "reasoning": 0,
        "cache": {"read": 7, "write": 3},
    }


def test_opencode_ignores_invalid_step_costs(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {"cost": 0.5}},
        {"type": "step_finish", "part": {"cost": -1}},
        {"type": "step_finish", "part": {"cost": True}},
        {"type": "step_finish", "part": {"cost": "0.25"}},
        {"type": "step_finish", "part": {"cost": None}},
        {"type": "step_finish", "part": {"cost": float("nan")}},
        {"type": "step_finish", "part": {"cost": float("inf")}},
        {"type": "step_finish", "part": {"reason": "tool"}},
        {"type": "step_finish"},
        {"type": "text", "part": {"text": "ignored"}},
        {"type": "step_finish", "part": {"reason": "stop", "cost": 0.5}},
    ))
    stdout.write_text(stdout.read_text() + "{not valid json\n")
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.outcome is HarnessOutcome.COMPLETED
    assert result.cost_usd == pytest.approx(1.0)


def test_opencode_absent_cost_is_none(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {"reason": "stop", "tokens": {"input": 1}}},
    ))
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.cost_usd is None


def test_opencode_reported_zero_cost_is_not_absent(tmp_path: Path) -> None:
    adapter = harnesses.get("opencode")
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "step_finish", "part": {"reason": "stop", "cost": 0}},
    ))
    stderr = _write(tmp_path, "stderr", "")

    result = adapter.inspect(stdout, stderr, exit_code=0)

    assert result.cost_usd == 0.0


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

    def fake_launch(argv, req, *, stdin_data=None, **kwargs):
        seen.update(argv=argv, request=req, stdin_data=stdin_data, **kwargs)
        return object()

    monkeypatch.setattr(base_module, "launch_process", fake_launch)

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


# ---------------------------------------------------------------------------
# Envelope 2: the harness says what served the attempt, what it cost, and how
# long it took — and the attempt can land in the ledger.


def _claude_capture(tmp_path: Path, **result_overrides) -> tuple[Path, Path]:
    result = {
        "type": "result", "subtype": "success", "is_error": False,
        "session_id": "sess-9", "result": "Done.", "num_turns": 15,
        "duration_ms": 100328, "total_cost_usd": 0.3447,
        "usage": {
            "input_tokens": 30, "cache_creation_input_tokens": 21069,
            "cache_read_input_tokens": 232599, "output_tokens": 6479,
            "output_tokens_details": {"thinking_tokens": 845},
        },
        "modelUsage": {"claude-opus-5-5": {"outputTokens": 6479, "costUSD": 0.3447}},
    }
    result.update(result_overrides)
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "system", "subtype": "init", "session_id": "sess-9",
         "model": "claude-opus-5-5", "claude_code_version": "2.1.286"},
        result,
    ))
    return stdout, _write(tmp_path, "stderr", "")


def test_claude_envelope_reports_model_version_cost_duration(tmp_path: Path) -> None:
    stdout, stderr = _claude_capture(tmp_path)
    result = harnesses.get("claude-cli").inspect(stdout, stderr, exit_code=0)
    assert result.envelope_version == harnesses.ENVELOPE_VERSION == 2
    assert result.model == "claude-opus-5-5"
    assert result.harness_version == "2.1.286"
    assert result.cost_usd == pytest.approx(0.3447)
    assert result.duration_ms == 100328
    assert result.turns == 15
    # The whole prompt, not the uncached slice; thinking tokens broken out.
    assert result.usage["prompt_tokens_total"] == 30 + 21069 + 232599
    assert result.usage["reasoning_tokens"] == 845
    assert result.as_dict()["outcome"] == "completed"


def test_claude_seat_limit_with_reset_is_rate_limit_with_retry_after(tmp_path: Path) -> None:
    stdout, stderr = _claude_capture(
        tmp_path,
        subtype="error_during_execution",
        is_error=True,
        terminal_reason="api_error",
        result="You've hit your usage limit · resets 3pm (America/New_York)",
    )
    result = harnesses.get("claude-cli").inspect(stdout, stderr, exit_code=1)
    assert result.outcome is HarnessOutcome.RATE_LIMIT
    assert result.retry_after_s is not None and result.retry_after_s > 60


def test_claude_seat_limit_without_reset_is_budget_exhausted(tmp_path: Path) -> None:
    stdout, stderr = _claude_capture(
        tmp_path,
        subtype="error_during_execution",
        is_error=True,
        terminal_reason="api_error",
        result="Monthly spend limit reached for this organization.",
    )
    result = harnesses.get("claude-cli").inspect(stdout, stderr, exit_code=1)
    assert result.outcome is HarnessOutcome.BUDGET_EXHAUSTED


def test_agent_stuck_tag_is_agent_blocked(tmp_path: Path) -> None:
    stdout, stderr = _claude_capture(
        tmp_path, result="<stuck>the repo has no tests to run</stuck>"
    )
    result = harnesses.get("claude-cli").inspect(stdout, stderr, exit_code=0)
    assert result.outcome is HarnessOutcome.AGENT_BLOCKED
    assert "no tests" in result.detail


def test_no_terminal_event_and_signal_exit_is_interrupted(tmp_path: Path) -> None:
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "system", "subtype": "init", "session_id": "s"},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": "working"}]}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    adapter = harnesses.get("claude-cli")
    assert adapter.inspect(stdout, stderr, exit_code=-9).outcome is HarnessOutcome.INTERRUPTED
    # A clean exit with nothing said stays unknown: the supervisor may know more.
    assert adapter.inspect(stdout, stderr, exit_code=0).outcome is HarnessOutcome.UNKNOWN


def test_input_error_is_distinct_from_provider_error(tmp_path: Path) -> None:
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "turn.failed", "error": {"message": "invalid request http_status=400"}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    assert harnesses.get("codex").inspect(stdout, stderr).outcome is HarnessOutcome.INPUT_ERROR


def test_codex_usage_gets_portable_aliases(tmp_path: Path) -> None:
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.completed", "usage": {
            "input_tokens": 3808034, "cached_input_tokens": 3652992,
            "output_tokens": 40357, "reasoning_output_tokens": 9396,
        }},
    ))
    stderr = _write(tmp_path, "stderr", "")
    result = harnesses.get("codex").inspect(stdout, stderr, exit_code=0)
    assert result.usage["reasoning_tokens"] == 9396
    assert result.usage["cache_read_input_tokens"] == 3652992
    assert result.model is None  # codex does not say; nothing is guessed


def test_opencode_reports_served_model(tmp_path: Path) -> None:
    stdout = _write(tmp_path, "stdout", _stream(
        {"type": "text", "sessionID": "s", "part": {
            "text": "hi", "metadata": {"model": {"modelID": "MiniMax-M2.7",
                                                 "providerID": "minimax-coding-plan"}},
        }},
        {"type": "step_finish", "sessionID": "s", "part": {"reason": "stop", "cost": 0}},
    ))
    stderr = _write(tmp_path, "stderr", "")
    result = harnesses.get("opencode").inspect(stdout, stderr, exit_code=0)
    assert result.model == "minimax-coding-plan/MiniMax-M2.7"
    assert result.turns == 1


@pytest.mark.parametrize(
    ("harness", "field", "value"),
    [
        ("codex", "max_turns", 5),
        ("codex", "agent", "build"),
        ("opencode", "reasoning_effort", "high"),
        ("claude-cli", "agent", "build"),
        ("claude-cli", "prompt_via_stdin", True),
    ],
)
def test_validate_request_fails_closed_on_unsupported_fields(
    tmp_path: Path, harness: str, field: str, value
) -> None:
    request = _request(tmp_path, **{field: value})
    with pytest.raises(HarnessCapabilityError, match=field):
        harnesses.validate_request(harness, request)


def test_capability_profile_is_a_versioned_record() -> None:
    profile = harnesses.capability_profile("claude-cli")
    assert profile["envelope_version"] == 2
    assert profile["harness"] == "claude-cli"
    assert profile["max_turns"] is True and profile["reports_cost"] is True
    assert "available" in profile


def test_start_scrubs_the_seat_api_key_and_layers_request_env(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake_launch(argv, req, *, env=None, stdin_data=None, start_new_session=False):
        seen.update(env=env, start_new_session=start_new_session)
        return object()

    monkeypatch.setattr(base_module, "launch_process", fake_launch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-pass-through")
    monkeypatch.setenv("HOME_MARKER", "kept")
    harnesses.get("claude-cli").start(
        _request(tmp_path, env={"FAB_ATTEMPT_DIR": "/x"}, start_new_session=True)
    )
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["env"]["HOME_MARKER"] == "kept"
    assert seen["env"]["FAB_ATTEMPT_DIR"] == "/x"
    assert seen["start_new_session"] is True


def test_record_attempt_lands_in_the_ledger(tmp_path: Path, monkeypatch) -> None:
    import sqlite3

    from somm_core.config import Config

    cfg = Config()
    cfg.project = "harness-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    stdout, stderr = _claude_capture(tmp_path)
    request = _request(tmp_path, correlation_id="job-42")
    result = harnesses.get("claude-cli").inspect(
        stdout, stderr, exit_code=0, correlation_id="job-42"
    )

    call = harnesses.record_attempt(result, request=request, config=cfg)

    with sqlite3.connect(cfg.db_path) as conn:
        row = conn.execute(
            "SELECT provider, model, origin, cost_usd, cost_basis, tokens_in, tokens_out, "
            "reasoning_tokens, cache_tokens_in, session_id, correlation_id, outcome, "
            "latency_ms, client_version FROM calls WHERE id = ?",
            (call.id,),
        ).fetchone()
    assert row[0] == "claude-cli" and row[1] == "claude-opus-5-5" and row[2] == "harness"
    assert row[3] == pytest.approx(0.3447) and row[4] == "reported"
    assert row[5] == 30 + 21069 + 232599 and row[6] == 6479 and row[7] == 845
    assert row[8] == 232599
    assert row[9] == "sess-9" and row[10] == "job-42" and row[11] == "ok"
    assert row[12] == 100328 and row[13]

@pytest.mark.parametrize("session_id", [None, "thread-1"])
def test_codex_cli_and_popen_share_canonical_cwd(tmp_path, monkeypatch, session_id):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    link = tmp_path / "link"
    link.symlink_to(workspace, target_is_directory=True)
    seen = {}
    class Process:
        stdin = None
    def popen(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return Process()
    monkeypatch.setattr(base_module, "scrubbed_env", lambda *_: {})
    monkeypatch.setattr(base_module.subprocess, "Popen", popen)
    handle = harnesses.get("codex").start(_request(
        tmp_path, cwd=link, allow_unsafe=False, session_id=session_id))
    handle.stdout_fh.close()
    handle.stderr_fh.close()
    argv = seen["argv"]
    assert seen["cwd"] == str(workspace.resolve()) == argv[argv.index("--cd") + 1]
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert "--ignore-user-config" in argv
    if session_id:
        assert argv.index("--sandbox") < argv.index("resume")

@pytest.mark.parametrize("bad_cwd", ["missing", "file"])
def test_missing_workspace_never_falls_back_home(tmp_path, monkeypatch, bad_cwd):
    (tmp_path / "file").write_text("synthetic")
    monkeypatch.setattr(base_module.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        harnesses.get("codex").start(_request(tmp_path, cwd=tmp_path / bad_cwd, allow_unsafe=False))

@pytest.mark.parametrize("overrides", [
    {"allow_unsafe": True}, {"extra": ["--sandbox", "danger-full-access"]},
    {"extra": ["--cd", "/"]}, {"extra": ["--config", "x=y"]},
    {"session_id": "--last"},
])
def test_codex_rejects_untyped_scope_overrides(tmp_path, overrides):
    values = {"allow_unsafe": False, **overrides}
    with pytest.raises(HarnessCapabilityError):
        harnesses.get("codex").build_argv(_request(tmp_path, **values))

@pytest.mark.parametrize("session_id", [None, "thread-1"])
def test_codex_option_looking_prompt_is_positional(tmp_path, session_id):
    argv = harnesses.get("codex").build_argv(_request(
        tmp_path, allow_unsafe=False, session_id=session_id,
        prompt="--dangerously-bypass-approvals-and-sandbox"))
    separator = argv.index("--")
    assert argv[-1] == "--dangerously-bypass-approvals-and-sandbox"
    assert argv[-1] not in argv[:separator]
    assert argv[separator + 1:] == ([session_id] if session_id else []) + [argv[-1]]


@pytest.mark.parametrize("mode", ["read-only", "workspace-write"])
def test_codex_typed_sandbox_and_ephemeral(tmp_path, mode):
    argv = harnesses.get("codex").build_argv(_request(tmp_path, allow_unsafe=False, sandbox_mode=mode, ephemeral=True))
    assert argv[argv.index("--sandbox") + 1] == mode
    assert "--ephemeral" in argv

@pytest.mark.parametrize("values", [{"sandbox_mode":"danger-full-access"}, {"ephemeral":True,"session_id":"thread-1"}])
def test_codex_rejects_unsafe_typed_mode_or_ephemeral_resume(tmp_path, values):
    with pytest.raises(HarnessCapabilityError):
        harnesses.get("codex").build_argv(_request(tmp_path, allow_unsafe=False, **values))


@pytest.mark.parametrize("network,expected", [(None, "false"), (False, "false"), (True, "true")])
def test_codex_workspace_write_network_is_always_explicit(tmp_path, network, expected):
    argv = harnesses.get("codex").build_argv(_request(
        tmp_path, allow_unsafe=False, network_access=network))
    configs = [argv[i + 1] for i, a in enumerate(argv) if a == "--config"]
    assert f"sandbox_workspace_write.network_access={expected}" in configs
    assert argv.index("sandbox_workspace_write.network_access=" + expected) < argv.index("--")


def test_codex_read_only_sends_no_network_override(tmp_path):
    argv = harnesses.get("codex").build_argv(_request(
        tmp_path, allow_unsafe=False, sandbox_mode="read-only"))
    assert not any(a.startswith("sandbox_workspace_write.") for a in argv)


def test_codex_network_requires_workspace_write(tmp_path):
    with pytest.raises(HarnessCapabilityError, match="workspace-write"):
        harnesses.get("codex").build_argv(_request(
            tmp_path, allow_unsafe=False, sandbox_mode="read-only", network_access=True))


def test_codex_network_override_survives_resume(tmp_path):
    argv = harnesses.get("codex").build_argv(_request(
        tmp_path, allow_unsafe=False, session_id="thread-1", network_access=True))
    assert "sandbox_workspace_write.network_access=true" in argv
    assert argv[-2:] == ["thread-1", "do it"]


@pytest.mark.parametrize("name", ["claude-cli", "opencode"])
@pytest.mark.parametrize("field,value", [
    ("sandbox_mode", "read-only"), ("ephemeral", True),
    ("network_access", False), ("network_access", True),
])
def test_unsandboxed_harnesses_reject_scope_they_cannot_enforce(tmp_path, name, field, value):
    request = _request(tmp_path, allow_unsafe=False, **{field: value})
    with pytest.raises(HarnessCapabilityError, match=field):
        harnesses.validate_request(name, request)


def test_codex_declares_its_enforceable_scope():
    caps = harnesses.get("codex").capabilities.as_dict()
    assert caps["sandbox_modes"] == ["read-only", "workspace-write"]
    assert caps["ephemeral"] is True and caps["network_sandbox"] is True
    assert harnesses.get("claude-cli").capabilities.network_sandbox is False

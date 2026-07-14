"""Reusable execution adapters for autonomous coding-agent harnesses.

The public API runs one attempt. Queueing, retry policy, inactivity watchdogs,
backend failover, verification, and release remain the caller's responsibility.
"""

from __future__ import annotations

import subprocess

from .base import (
    AgentHarness,
    HarnessCapabilities,
    HarnessHandle,
    HarnessOutcome,
    HarnessRequest,
    HarnessResult,
    decode_json_events,
)
from .claude_cli import ClaudeCLIHarness
from .codex import CodexHarness
from .opencode import OpenCodeHarness

_CODEX = CodexHarness()

_HARNESSES: dict[str, AgentHarness] = {
    "claude-cli": ClaudeCLIHarness(),
    "codex": _CODEX,
    "codex-cli": _CODEX,
    "opencode": OpenCodeHarness(),
}


def get(name: str) -> AgentHarness:
    """Return a built-in harness adapter by stable name."""

    try:
        return _HARNESSES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown harness {name!r}; expected one of {sorted(_HARNESSES)}"
        ) from error


def available() -> list[str]:
    """List canonical harness names whose executables are on PATH."""

    return [
        name for name in ("claude-cli", "codex", "opencode")
        if _HARNESSES[name].is_available()
    ]


def start(name: str, request: HarnessRequest) -> HarnessHandle:
    """Start one attempt and return immediately with its process handle."""

    return get(name).start(request)


def inspect(
    name: str,
    stdout_path,
    stderr_path,
    *,
    exit_code: int | None = None,
    correlation_id: str | None = None,
) -> HarnessResult:
    """Normalize a completed adapter capture without applying retry policy."""

    return get(name).inspect(
        stdout_path,
        stderr_path,
        exit_code=exit_code,
        correlation_id=correlation_id,
    )


def terminal_event(name: str, stdout_path):
    return get(name).parse_terminal(stdout_path)


def session_id(name: str, stdout_path) -> str | None:
    return get(name).parse_session_id(stdout_path)


def result_text(name: str, stdout_path) -> str:
    return get(name).extract_final_text(stdout_path)


def extract_final_text(capture: str) -> str | None:
    """Extract a final agent message from any built-in JSON event stream."""

    claude_result: str | None = None
    codex_result: str | None = None
    opencode_chunks: list[str] = []
    for event in decode_json_events(capture):
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            claude_result = event["result"]
        if event.get("type") == "item.completed":
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                codex_result = item["text"]
        if event.get("type") == "text":
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            value = part.get("text") or event.get("text")
            if isinstance(value, str):
                opencode_chunks.append(value)
    if claude_result is not None:
        return claude_result
    if codex_result is not None:
        return codex_result
    if opencode_chunks:
        return "".join(opencode_chunks)
    return None


def extract_last_assistant_text(capture: str) -> str | None:
    """Return the last assistant text even when no terminal event exists."""

    last: str | None = None
    for event in decode_json_events(capture):
        if event.get("type") == "assistant":
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            content = message.get("content") if isinstance(message.get("content"), list) else []
            text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text:
                last = text
        if event.get("type") == "item.completed":
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                last = item["text"]
    return last


def is_event_stream(capture: str) -> bool:
    """Return whether a bounded capture sample contains typed JSON events."""

    lines = capture.splitlines()
    sample = "\n".join(lines[:5] + lines[-5:])
    return any("type" in event for event in decode_json_events(sample))


def run(
    name: str, request: HarnessRequest, *, timeout: float | None = None
) -> HarnessResult:
    """Run one attempt synchronously; never retries or selects a fallback."""

    adapter = get(name)
    handle = adapter.start(request)
    try:
        try:
            exit_code = handle.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            handle.proc.terminate()
            try:
                handle.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                handle.proc.wait()
            raise
    finally:
        handle.close_captures()
    return adapter.inspect(
        handle.stdout_path,
        handle.stderr_path,
        exit_code=exit_code,
        correlation_id=request.correlation_id,
    )


__all__ = [
    "AgentHarness",
    "ClaudeCLIHarness",
    "CodexHarness",
    "HarnessCapabilities",
    "HarnessHandle",
    "HarnessOutcome",
    "HarnessRequest",
    "HarnessResult",
    "OpenCodeHarness",
    "available",
    "extract_final_text",
    "extract_last_assistant_text",
    "get",
    "inspect",
    "is_event_stream",
    "result_text",
    "run",
    "session_id",
    "start",
    "terminal_event",
]

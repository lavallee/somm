"""Pi CLI adapter for one portable, non-interactive agent attempt."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .base import (
    HarnessCapabilities,
    HarnessCapabilityError,
    HarnessOutcome,
    HarnessRequest,
    HarnessResult,
    classify_error_text,
    iter_json_events,
    launch_process,
    read_capture,
)

_SAFE_FLAGS = (
    "--no-tools",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--no-approve",
)


def _assistant_messages(event: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if event.get("type") == "message_end" and isinstance(event.get("message"), dict):
        messages.append(event["message"])
    if event.get("type") == "agent_end" and isinstance(event.get("messages"), list):
        messages.extend(
            message for message in event["messages"] if isinstance(message, dict)
        )
    return [message for message in messages if message.get("role") == "assistant"]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


class PiHarness:
    """Translate Pi's JSON event stream into Somm's one-attempt contract.

    ``allow_unsafe=False`` is intentionally narrow: Pi receives no tools,
    extensions, skills, templates, themes, project context, or project-local
    configuration. Tool-using Pi runners need a separately reviewed process
    isolation profile; callers cannot weaken safe mode through ``extra``.
    """

    name = "pi"
    capabilities = HarnessCapabilities(resume=True, reasoning_effort=True)

    def is_available(self) -> bool:
        return shutil.which("pi") is not None

    def build_argv(self, request: HarnessRequest) -> list[str]:
        if not request.allow_unsafe and request.extra:
            raise HarnessCapabilityError(
                "Pi safe mode does not accept extra CLI arguments; use a separately "
                "reviewed runner profile for tool or extension access"
            )

        argv = [request.resolved_executable("pi"), "--mode", "json"]
        if request.allow_unsafe:
            # Project-local packages and instruction files remain outside the
            # implicit trust boundary even for an externally sandboxed runner.
            argv.extend(["--no-context-files", "--no-approve"])
        else:
            argv.extend(_SAFE_FLAGS)
        if request.model:
            argv.extend(["--model", request.model])
        if request.reasoning_effort:
            argv.extend(["--thinking", request.reasoning_effort])
        if request.session_id:
            argv.extend(["--session", request.session_id])
        else:
            argv.append("--no-session")
        argv.extend(str(arg) for arg in request.extra)
        if not request.prompt_via_stdin:
            argv.extend(["--", request.prompt])
        return argv

    def start(self, request: HarnessRequest):
        stdin_data = request.prompt if request.prompt_via_stdin else None
        return launch_process(
            self.build_argv(request), request, stdin_data=stdin_data
        )

    @staticmethod
    def parse_terminal(path: Path) -> dict | None:
        terminal = None
        for event in iter_json_events(path):
            if event.get("type") == "agent_end":
                terminal = event
        return terminal

    @staticmethod
    def parse_session_id(path: Path) -> str | None:
        for event in iter_json_events(path):
            if event.get("type") == "session" and event.get("id"):
                return str(event["id"])
        return None

    @staticmethod
    def extract_final_text(path: Path) -> str:
        final = ""
        for event in iter_json_events(path):
            for message in _assistant_messages(event):
                final = _message_text(message)
        return final.strip()

    @staticmethod
    def _last_assistant(path: Path) -> dict[str, Any] | None:
        last = None
        for event in iter_json_events(path):
            for message in _assistant_messages(event):
                last = message
        return last

    def inspect(
        self, stdout_path: Path, stderr_path: Path, *, exit_code=None, correlation_id=None
    ) -> HarnessResult:
        terminal = self.parse_terminal(stdout_path)
        message = self._last_assistant(stdout_path)
        detail = ""
        usage: dict[str, int | float] = {}
        outcome = HarnessOutcome.UNKNOWN

        if message is not None:
            reason = str(message.get("stopReason") or "").lower()
            raw_usage = message.get("usage")
            if isinstance(raw_usage, dict):
                usage = {
                    str(key): value
                    for key, value in raw_usage.items()
                    if isinstance(value, int | float) and not isinstance(value, bool)
                }
            error_message = message.get("errorMessage")
            if isinstance(error_message, str):
                detail = error_message
            if terminal is not None and reason == "stop":
                outcome = HarnessOutcome.COMPLETED
            elif reason == "length":
                # Pi has no native turn cap. "length" is the provider output
                # limit, so Fab must not apply its resume-and-add-turns policy.
                outcome = HarnessOutcome.CONTEXT_LIMIT
            elif reason in {"error", "aborted"}:
                evidence = detail + "\n" + read_capture(stderr_path, tail=4000)
                outcome = classify_error_text(evidence) or HarnessOutcome.FAILED

        if terminal is None:
            evidence = read_capture(stdout_path, tail=4000) + "\n" + read_capture(
                stderr_path, tail=4000
            )
            detail = detail or evidence.strip()
            outcome = classify_error_text(evidence) or (
                HarnessOutcome.FAILED
                if exit_code not in {None, 0}
                else HarnessOutcome.UNKNOWN
            )

        return HarnessResult(
            harness=self.name,
            outcome=outcome,
            final_text=self.extract_final_text(stdout_path),
            session_id=self.parse_session_id(stdout_path),
            exit_code=exit_code,
            detail=detail.strip(),
            usage=usage,
            terminal_event=terminal,
            correlation_id=correlation_id,
        )

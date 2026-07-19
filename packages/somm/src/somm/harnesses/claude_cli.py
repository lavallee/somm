"""Claude Code CLI adapter for autonomous workspace tasks."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import (
    HarnessCapabilities,
    HarnessOutcome,
    HarnessRequest,
    HarnessResult,
    classify_error_text,
    iter_json_events,
    launch_process,
    read_capture,
)


class ClaudeCLIHarness:
    name = "claude-cli"
    capabilities = HarnessCapabilities(
        resume=True, max_turns=True, reasoning_effort=True
    )

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def build_argv(self, request: HarnessRequest) -> list[str]:
        argv = [
            request.resolved_executable("claude"), "-p", request.prompt,
            "--verbose", "--output-format", "stream-json",
        ]
        if request.model:
            argv.extend(["--model", request.model])
        if request.reasoning_effort:
            argv.extend(["--effort", request.reasoning_effort])
        if request.max_turns is not None:
            argv.extend(["--max-turns", str(int(request.max_turns))])
        if request.allow_unsafe:
            argv.extend(["--permission-mode", "bypassPermissions"])
        if request.session_id:
            argv.extend(["--resume", request.session_id])
        argv.extend(str(arg) for arg in request.extra)
        return argv

    def start(self, request: HarnessRequest):
        return launch_process(self.build_argv(request), request)

    @staticmethod
    def parse_terminal(path: Path) -> dict | None:
        terminal = None
        for event in iter_json_events(path):
            if event.get("type") == "result":
                terminal = event
        return terminal

    @staticmethod
    def parse_session_id(path: Path) -> str | None:
        for event in iter_json_events(path):
            value = event.get("session_id") or event.get("conversation_id")
            if value:
                return str(value)
        return None

    @classmethod
    def extract_final_text(cls, path: Path) -> str:
        terminal = cls.parse_terminal(path)
        if not terminal:
            return ""
        value = terminal.get("result") or terminal.get("text") or ""
        return value.strip() if isinstance(value, str) else ""

    def inspect(
        self, stdout_path: Path, stderr_path: Path, *, exit_code=None, correlation_id=None
    ) -> HarnessResult:
        terminal = self.parse_terminal(stdout_path)
        session_id = self.parse_session_id(stdout_path)
        final_text = self.extract_final_text(stdout_path)
        detail = ""
        outcome = HarnessOutcome.UNKNOWN
        if terminal is not None:
            subtype = str(terminal.get("subtype") or "").lower()
            if not terminal.get("is_error") and subtype == "success":
                outcome = HarnessOutcome.COMPLETED
            elif subtype == "error_max_turns":
                outcome = HarnessOutcome.TURN_LIMIT
            elif terminal.get("is_error"):
                detail = read_capture(stderr_path, tail=4000)
                outcome = classify_error_text(detail) or HarnessOutcome.FAILED
        else:
            detail = read_capture(stdout_path, tail=4000) + "\n" + read_capture(
                stderr_path, tail=4000
            )
            outcome = classify_error_text(detail) or HarnessOutcome.UNKNOWN
            lowered = detail.lower()
            if "max turns" in lowered or "turn limit" in lowered:
                outcome = HarnessOutcome.TURN_LIMIT
        usage = terminal.get("usage") if isinstance(terminal, dict) else {}
        return HarnessResult(
            harness=self.name,
            outcome=outcome,
            final_text=final_text,
            session_id=session_id,
            exit_code=exit_code,
            detail=detail.strip(),
            usage=usage if isinstance(usage, dict) else {},
            terminal_event=terminal,
            correlation_id=correlation_id,
        )

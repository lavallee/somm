"""OpenAI Codex CLI adapter for autonomous workspace tasks."""

from __future__ import annotations

import json
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


class CodexHarness:
    name = "codex"
    capabilities = HarnessCapabilities(resume=True, reasoning_effort=True)

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def build_argv(self, request: HarnessRequest) -> list[str]:
        argv = [request.resolved_executable("codex"), "exec"]
        if request.session_id:
            argv.append("resume")
        argv.append("--json")
        if request.allow_unsafe:
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        argv.append("--skip-git-repo-check")
        if request.model:
            argv.extend(["--model", request.model])
        if request.reasoning_effort:
            argv.extend([
                "--config",
                f"model_reasoning_effort={json.dumps(request.reasoning_effort)}",
            ])
        argv.extend(str(arg) for arg in request.extra)
        if request.session_id:
            argv.append(request.session_id)
        argv.append("-" if request.prompt_via_stdin else request.prompt)
        return argv

    def start(self, request: HarnessRequest):
        return launch_process(
            self.build_argv(request),
            request,
            stdin_data=request.prompt if request.prompt_via_stdin else None,
        )

    @staticmethod
    def parse_terminal(path: Path) -> dict | None:
        terminal = None
        for event in iter_json_events(path):
            if event.get("type") in {"turn.completed", "turn.failed"}:
                terminal = event
        return terminal

    @staticmethod
    def parse_session_id(path: Path) -> str | None:
        for event in iter_json_events(path):
            if event.get("type") == "thread.started" and event.get("thread_id"):
                return str(event["thread_id"])
        return None

    @staticmethod
    def extract_final_text(path: Path) -> str:
        final = ""
        for event in iter_json_events(path):
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            if (
                event.get("type") == "item.completed"
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                final = item["text"]
        return final.strip()

    def inspect(
        self, stdout_path: Path, stderr_path: Path, *, exit_code=None, correlation_id=None
    ) -> HarnessResult:
        terminal = self.parse_terminal(stdout_path)
        outcome = HarnessOutcome.UNKNOWN
        detail = ""
        if terminal is not None:
            if terminal.get("type") == "turn.completed":
                outcome = HarnessOutcome.COMPLETED
            else:
                detail = json.dumps(terminal, sort_keys=True) + "\n" + read_capture(
                    stderr_path, tail=4000
                )
                outcome = classify_error_text(detail) or HarnessOutcome.FAILED
        else:
            detail = read_capture(stdout_path, tail=4000) + "\n" + read_capture(
                stderr_path, tail=4000
            )
            outcome = classify_error_text(detail) or HarnessOutcome.UNKNOWN
        usage = terminal.get("usage") if isinstance(terminal, dict) else {}
        return HarnessResult(
            harness=self.name,
            outcome=outcome,
            final_text=self.extract_final_text(stdout_path),
            session_id=self.parse_session_id(stdout_path),
            exit_code=exit_code,
            detail=detail.strip(),
            usage=usage if isinstance(usage, dict) else {},
            terminal_event=terminal,
            correlation_id=correlation_id,
        )

"""Pi CLI adapter for one portable, non-interactive agent attempt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    BaseHarness,
    HarnessCapabilities,
    HarnessCapabilityError,
    HarnessOutcome,
    HarnessRequest,
    classify_error_text,
    iter_json_events,
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


class PiHarness(BaseHarness):
    """Translate Pi's JSON event stream into Somm's one-attempt contract.

    ``allow_unsafe=False`` is intentionally narrow: Pi receives no tools,
    extensions, skills, templates, themes, project context, or project-local
    configuration. Tool-using Pi runners need a separately reviewed process
    isolation profile; callers cannot weaken safe mode through ``extra``.
    """

    name = "pi"
    executable = "pi"
    capabilities = HarnessCapabilities(
        resume=True, reasoning_effort=True, prompt_via_stdin=True, reports_model=True
    )

    def build_argv(self, request: HarnessRequest) -> list[str]:
        if not request.allow_unsafe and request.extra:
            raise HarnessCapabilityError(
                "Pi safe mode does not accept extra CLI arguments; use a separately "
                "reviewed runner profile for tool or extension access"
            )

        argv = [request.resolved_executable(self.executable), "--mode", "json"]
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

    def parse_usage(self, path: Path, terminal: dict | None) -> dict:
        message = self._last_assistant(path)
        raw_usage = message.get("usage") if message else None
        if not isinstance(raw_usage, dict):
            return {}
        return {
            str(key): value
            for key, value in raw_usage.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }

    def parse_model(self, path: Path, terminal: dict | None) -> str | None:
        message = self._last_assistant(path)
        if not message:
            return None
        model = message.get("model")
        provider = message.get("provider")
        if isinstance(model, str) and model:
            return f"{provider}/{model}" if isinstance(provider, str) and provider else model
        return None

    def parse_cost_usd(self, path: Path, terminal: dict | None) -> float | None:
        message = self._last_assistant(path)
        raw_usage = message.get("usage") if message else None
        cost = raw_usage.get("cost") if isinstance(raw_usage, dict) else None
        total = cost.get("total") if isinstance(cost, dict) else None
        if isinstance(total, bool) or not isinstance(total, (int, float)) or total < 0:
            return None
        return float(total)

    def classify_without_terminal(self, stdout_path, stderr_path, exit_code):
        # Pi's authoritative signal is the last assistant message, which can
        # exist without an agent_end (the process was cut off after it).
        message = self._last_assistant(stdout_path)
        if message is not None:
            reason = str(message.get("stopReason") or "").lower()
            error_message = message.get("errorMessage")
            detail = error_message if isinstance(error_message, str) else ""
            if reason == "length":
                return HarnessOutcome.CONTEXT_LIMIT, detail
            if reason in {"error", "aborted"}:
                evidence = detail + "\n" + read_capture(stderr_path, tail=4000)
                return classify_error_text(evidence) or HarnessOutcome.FAILED, detail
        return super().classify_without_terminal(stdout_path, stderr_path, exit_code)

    def classify(self, terminal: dict, stdout_path: Path, stderr_path: Path):
        message = self._last_assistant(stdout_path)
        if message is None:
            return HarnessOutcome.UNKNOWN, ""
        reason = str(message.get("stopReason") or "").lower()
        error_message = message.get("errorMessage")
        detail = error_message if isinstance(error_message, str) else ""
        if reason == "stop":
            return HarnessOutcome.COMPLETED, ""
        if reason == "length":
            # Pi has no native turn cap. "length" is the provider output
            # limit, so a runner must not apply a resume-and-add-turns policy.
            return HarnessOutcome.CONTEXT_LIMIT, detail
        if reason in {"error", "aborted"}:
            evidence = detail + "\n" + read_capture(stderr_path, tail=4000)
            return classify_error_text(evidence) or HarnessOutcome.FAILED, detail
        return HarnessOutcome.UNKNOWN, detail

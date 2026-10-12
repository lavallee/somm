"""OpenAI Codex CLI adapter for autonomous workspace tasks.

``codex exec --json`` emits ``thread.started`` (thread id), ``item.*`` events,
and ``turn.completed`` / ``turn.failed``. ``turn.completed.usage`` carries
``input_tokens``, ``cached_input_tokens``, ``output_tokens`` and
``reasoning_output_tokens``. Codex does not report the served model or a
cost; both stay ``None`` rather than being guessed from the request.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import (
    BaseHarness,
    HarnessCapabilities,
    HarnessOutcome,
    HarnessRequest,
    classify_error_text,
    iter_json_events,
    read_capture,
)


class CodexHarness(BaseHarness):
    name = "codex"
    executable = "codex"
    capabilities = HarnessCapabilities(
        resume=True,
        reasoning_effort=True,
        prompt_via_stdin=True,
        model_families=frozenset({"openai"}),
    )
    scrub_env = ("OPENAI_API_KEY",)

    def build_argv(self, request: HarnessRequest) -> list[str]:
        argv = [request.resolved_executable(self.executable), "exec"]
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

    def parse_usage(self, path: Path, terminal: dict | None) -> dict:
        usage = super().parse_usage(path, terminal)
        # Normalized aliases beside the native keys, so a consumer can read
        # the same name across harnesses.
        if isinstance(usage.get("reasoning_output_tokens"), int):
            usage.setdefault("reasoning_tokens", usage["reasoning_output_tokens"])
        if isinstance(usage.get("cached_input_tokens"), int):
            usage.setdefault("cache_read_input_tokens", usage["cached_input_tokens"])
        return usage

    def classify(self, terminal: dict, stdout_path: Path, stderr_path: Path):
        if terminal.get("type") == "turn.completed":
            return HarnessOutcome.COMPLETED, ""
        detail = json.dumps(terminal, sort_keys=True) + "\n" + read_capture(stderr_path, tail=4000)
        return classify_error_text(detail) or HarnessOutcome.FAILED, detail

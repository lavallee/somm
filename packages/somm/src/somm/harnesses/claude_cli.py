"""Claude Code CLI adapter for autonomous workspace tasks.

``claude -p --output-format stream-json`` emits a ``system/init`` event (model,
``claude_code_version``), per-turn events, and one ``result`` event carrying
``total_cost_usd``, ``duration_ms``, ``num_turns``, inclusive ``usage``, and a
per-model ``modelUsage`` breakdown. All of it is read here; nothing is
guessed.
"""

from __future__ import annotations

from pathlib import Path

from somm_core.parse import anthropic_prompt_tokens

from .base import (
    BaseHarness,
    HarnessCapabilities,
    HarnessOutcome,
    HarnessRequest,
    classify_error_text,
    iter_json_events,
    read_capture,
    retry_after_from_text,
)


class ClaudeCLIHarness(BaseHarness):
    name = "claude-cli"
    executable = "claude"
    capabilities = HarnessCapabilities(
        resume=True,
        max_turns=True,
        reasoning_effort=True,
        reports_cost=True,
        reports_model=True,
        model_families=frozenset({"anthropic"}),
    )
    # The seat bills the subscription; a key in the environment would bill
    # the API account and attribute the spend to the wrong plan.
    scrub_env = ("ANTHROPIC_API_KEY",)

    def build_argv(self, request: HarnessRequest) -> list[str]:
        argv = [
            request.resolved_executable(self.executable), "-p", request.prompt,
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

    # --- parsers -----------------------------------------------------

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

    @staticmethod
    def _init_event(path: Path) -> dict | None:
        for event in iter_json_events(path):
            if event.get("type") == "system" and event.get("subtype") == "init":
                return event
        return None

    def parse_model(self, path: Path, terminal: dict | None) -> str | None:
        # modelUsage is keyed by the model that actually answered, which is
        # the one to record; the init event names the one that was asked for.
        usage = terminal.get("modelUsage") if isinstance(terminal, dict) else None
        if isinstance(usage, dict) and usage:
            served = [k for k, v in usage.items() if isinstance(v, dict)]
            if len(served) == 1:
                return served[0]
            if served:
                # Several models (a subagent, a fallback): record the one that
                # did the most output; the breakdown stays in terminal_event.
                return max(served, key=lambda k: (usage[k].get("outputTokens") or 0))
        init = self._init_event(path)
        model = init.get("model") if init else None
        return str(model) if model else None

    def parse_version(self, path: Path, terminal: dict | None) -> str | None:
        init = self._init_event(path)
        value = init.get("claude_code_version") if init else None
        return str(value) if value else None

    def parse_usage(self, path: Path, terminal: dict | None) -> dict:
        usage = terminal.get("usage") if isinstance(terminal, dict) else None
        if not isinstance(usage, dict):
            return {}
        out = {k: v for k, v in usage.items() if isinstance(v, (int, float))}
        # The whole prompt, not just the uncached slice; see
        # somm_core.parse.anthropic_prompt_tokens.
        out["prompt_tokens_total"] = anthropic_prompt_tokens(usage)
        details = usage.get("output_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("thinking_tokens"), int):
            out["reasoning_tokens"] = details["thinking_tokens"]
        return out

    def parse_cost_usd(self, path: Path, terminal: dict | None) -> float | None:
        value = terminal.get("total_cost_usd") if isinstance(terminal, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        return float(value)

    def parse_duration_ms(self, path: Path, terminal: dict | None) -> int | None:
        value = terminal.get("duration_ms") if isinstance(terminal, dict) else None
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def parse_turns(self, path: Path, terminal: dict | None) -> int | None:
        value = terminal.get("num_turns") if isinstance(terminal, dict) else None
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def parse_retry_after(self, terminal: dict) -> float | None:
        if terminal.get("terminal_reason") == "api_error":
            return retry_after_from_text(str(terminal.get("result", "")))
        return None

    # --- classification ----------------------------------------------

    def classify(self, terminal: dict, stdout_path: Path, stderr_path: Path):
        subtype = str(terminal.get("subtype") or "").lower()
        reason = str(terminal.get("terminal_reason") or "").lower()
        if not terminal.get("is_error") and subtype == "success":
            return HarnessOutcome.COMPLETED, ""
        if subtype == "error_max_turns" or reason == "max_turns":
            return HarnessOutcome.TURN_LIMIT, ""
        if subtype == "error_max_budget_usd" or reason == "budget_exhausted":
            # The caller's --max-budget-usd cap, enforced client-side by the CLI.
            return HarnessOutcome.BUDGET_EXHAUSTED, (
                f"claude --max-budget-usd reached (total_cost_usd="
                f"{terminal.get('total_cost_usd')})"
            )
        if reason == "api_error":
            # Only a provider terminal error can establish a seat limit —
            # never quoted tool output in the transcript.
            message = str(terminal.get("result", ""))
            if _looks_like_seat_limit(message):
                if "reset" in message.lower():
                    return HarnessOutcome.RATE_LIMIT, message
                return HarnessOutcome.BUDGET_EXHAUSTED, (
                    message + "\nprovider limit without a reset; no automatic increase"
                )
        detail = read_capture(stderr_path, tail=4000)
        if not detail.strip():
            detail = str(terminal.get("result") or terminal.get("api_error_status") or "")
        return classify_error_text(detail) or HarnessOutcome.FAILED, detail

    def classify_without_terminal(self, stdout_path, stderr_path, exit_code):
        outcome, detail = super().classify_without_terminal(stdout_path, stderr_path, exit_code)
        lowered = detail.lower()
        if "max turns" in lowered or "turn limit" in lowered:
            return HarnessOutcome.TURN_LIMIT, detail
        return outcome, detail


def _looks_like_seat_limit(message: str) -> bool:
    import re

    return bool(re.search(r"(?:spend|usage|rate|session|weekly) limit", message, re.IGNORECASE))

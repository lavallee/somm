"""OpenCode CLI adapter for autonomous workspace tasks."""

from __future__ import annotations

import math
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


class OpenCodeHarness:
    name = "opencode"
    capabilities = HarnessCapabilities(resume=True, agent_selection=True)

    def is_available(self) -> bool:
        return shutil.which("opencode") is not None

    def build_argv(self, request: HarnessRequest) -> list[str]:
        argv = [request.resolved_executable("opencode"), "run", "--format", "json"]
        if request.allow_unsafe:
            argv.append("--dangerously-skip-permissions")
        if request.model:
            argv.extend(["--model", request.model])
        if request.agent:
            argv.extend(["--agent", request.agent])
        if request.session_id:
            argv.extend(["--session", request.session_id])
        argv.extend(["--dir", request.resolved_cwd()])
        argv.extend(str(arg) for arg in request.extra)
        argv.append(request.prompt)
        return argv

    def start(self, request: HarnessRequest):
        return launch_process(self.build_argv(request), request)

    @staticmethod
    def parse_terminal(path: Path) -> dict | None:
        terminal = None
        for event in iter_json_events(path):
            if event.get("type") == "step_finish":
                terminal = event
        return terminal

    @staticmethod
    def parse_cost_usd(path: Path) -> float | None:
        """Sum valid reported step costs, or None when none were reported."""

        total: float | None = None
        for event in iter_json_events(path):
            if event.get("type") != "step_finish":
                continue
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            value = part.get("cost")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value) or value < 0:
                continue
            total = float(value) if total is None else total + float(value)
        return total

    @staticmethod
    def parse_billing_usage(path: Path) -> dict:
        """Sum token categories across billable OpenCode steps.

        ``usage`` remains the terminal context snapshot used by callers.  This
        separate rollup is suitable for approximate rate-card comparisons.
        """

        totals = {
            "steps": 0,
            "total": 0,
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache": {"read": 0, "write": 0},
        }
        found = False
        for event in iter_json_events(path):
            if event.get("type") != "step_finish":
                continue
            part = event.get("part")
            if not isinstance(part, dict) or not isinstance(part.get("tokens"), dict):
                continue
            tokens = part["tokens"]
            found = True
            totals["steps"] += 1
            for key in ("total", "input", "output", "reasoning"):
                value = tokens.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ):
                    totals[key] += value
            cache = tokens.get("cache")
            if isinstance(cache, dict):
                for key in ("read", "write"):
                    value = cache.get(key)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value >= 0
                    ):
                        totals["cache"][key] += value
        return totals if found else {}

    @staticmethod
    def parse_session_id(path: Path) -> str | None:
        for event in iter_json_events(path):
            value = event.get("sessionID") or event.get("session_id")
            if not value and isinstance(event.get("part"), dict):
                value = event["part"].get("sessionID") or event["part"].get("session_id")
            if value:
                return str(value)
        return None

    @staticmethod
    def extract_final_text(path: Path) -> str:
        chunks: list[str] = []
        for event in iter_json_events(path):
            if event.get("type") != "text":
                continue
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            value = part.get("text") or event.get("text")
            if isinstance(value, str):
                chunks.append(value)
        return "".join(chunks).strip()

    def inspect(
        self, stdout_path: Path, stderr_path: Path, *, exit_code=None, correlation_id=None
    ) -> HarnessResult:
        terminal = self.parse_terminal(stdout_path)
        outcome = HarnessOutcome.UNKNOWN
        detail = ""
        usage: dict = {}
        if terminal is not None:
            part = terminal.get("part") if isinstance(terminal.get("part"), dict) else {}
            reason = str(part.get("reason") or terminal.get("reason") or "").lower()
            if reason == "stop":
                outcome = HarnessOutcome.COMPLETED
            elif reason == "length":
                outcome = HarnessOutcome.TURN_LIMIT
            elif reason in {"error", "content-filter"}:
                detail = read_capture(stderr_path, tail=4000)
                outcome = classify_error_text(detail) or (
                    HarnessOutcome.REFUSED
                    if reason == "content-filter"
                    else HarnessOutcome.FAILED
                )
            elif reason:
                detail = f"OpenCode ended on a non-terminal step reason: {reason!r}."
                stderr_detail = read_capture(stderr_path, tail=4000).strip()
                if stderr_detail:
                    detail = f"{detail}\n{stderr_detail}"
                outcome = HarnessOutcome.FAILED
            usage = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
        else:
            detail = read_capture(stdout_path, tail=4000) + "\n" + read_capture(
                stderr_path, tail=4000
            )
            outcome = classify_error_text(detail) or HarnessOutcome.UNKNOWN
        return HarnessResult(
            harness=self.name,
            outcome=outcome,
            final_text=self.extract_final_text(stdout_path),
            session_id=self.parse_session_id(stdout_path),
            exit_code=exit_code,
            detail=detail.strip(),
            usage=usage,
            billing_usage=self.parse_billing_usage(stdout_path),
            cost_usd=self.parse_cost_usd(stdout_path),
            terminal_event=terminal,
            correlation_id=correlation_id,
        )

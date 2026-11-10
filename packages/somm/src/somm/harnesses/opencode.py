"""OpenCode CLI adapter for autonomous workspace tasks.

``opencode run --format json`` emits ``step_start`` / ``text`` / ``tool_use`` /
``step_finish`` events. Each ``step_finish.part`` carries a ``reason``,
``tokens`` (input/output/reasoning/cache) and a ``cost``. The served model
appears as ``metadata.model.{providerID,modelID}`` on message parts.
"""

from __future__ import annotations

import functools
import math
import os
import re
import shutil
import subprocess
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

_VERSION = re.compile(r"(\d+)\.\d+")


@functools.lru_cache(maxsize=8)
def _major_version(executable: str, modified_ns: int) -> int | None:
    """The installed major version, or None when it cannot be determined."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION.search(result.stdout) if result.returncode == 0 else None
    return int(match.group(1)) if match else None


def _uses_v2_cli(executable: str) -> bool:
    resolved = shutil.which(executable)
    if resolved is None:
        return False
    try:
        modified_ns = os.stat(resolved).st_mtime_ns
    except OSError:
        return False
    major = _major_version(resolved, modified_ns)
    return major is not None and major >= 2


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


class OpenCodeHarness(BaseHarness):
    name = "opencode"
    executable = "opencode"
    capabilities = HarnessCapabilities(
        resume=True,
        agent_selection=True,
        reports_cost=True,
        reports_model=True,
    )

    def build_argv(self, request: HarnessRequest) -> list[str]:
        executable = request.resolved_executable(self.executable)
        v2 = _uses_v2_cli(executable)
        argv = [executable, "run"]
        if v2:
            # 2.x attaches to a shared background service unless told otherwise.
            # A private server stays inside the caller's process tree.
            argv.append("--standalone")
        argv.extend(["--format", "json"])
        if request.allow_unsafe:
            argv.append("--auto" if v2 else "--dangerously-skip-permissions")
        if request.model:
            argv.extend(["--model", request.model])
        if request.agent:
            argv.extend(["--agent", request.agent])
        if request.session_id:
            argv.extend(["--session", request.session_id])
        if not v2:
            # 2.x removed --dir and runs in the process working directory.
            argv.extend(["--dir", request.resolved_cwd()])
        argv.extend(str(arg) for arg in request.extra)
        # `--` ends option parsing, so a prompt starting with "-" is the
        # message, not a flag (2.x verified: the value after `--` is taken as a
        # positional; 1.x uses yargs, which treats `--` the same way).
        argv.extend(["--", request.prompt])
        return argv

    @staticmethod
    def parse_terminal(path: Path) -> dict | None:
        terminal = None
        for event in iter_json_events(path):
            if event.get("type") == "step_finish":
                terminal = event
        return terminal

    def parse_cost_usd(self, path: Path, terminal: dict | None) -> float | None:
        """Sum valid reported step costs, or None when none were reported."""
        total: float | None = None
        for event in iter_json_events(path):
            if event.get("type") != "step_finish":
                continue
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            value = _number(part.get("cost"))
            if value is None:
                continue
            total = value if total is None else total + value
        return total

    def parse_billing_usage(self, path: Path, terminal: dict | None) -> dict:
        """Sum token categories across billable OpenCode steps.

        ``usage`` remains the terminal context snapshot used by callers. This
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
                value = _number(tokens.get(key))
                if value is not None:
                    totals[key] += value
            cache = tokens.get("cache")
            if isinstance(cache, dict):
                for key in ("read", "write"):
                    value = _number(cache.get(key))
                    if value is not None:
                        totals["cache"][key] += value
        return totals if found else {}

    def parse_usage(self, path: Path, terminal: dict | None) -> dict:
        part = terminal.get("part") if isinstance(terminal, dict) else None
        tokens = part.get("tokens") if isinstance(part, dict) else None
        return dict(tokens) if isinstance(tokens, dict) else {}

    def parse_model(self, path: Path, terminal: dict | None) -> str | None:
        last: str | None = None
        for event in iter_json_events(path):
            for holder in (event, event.get("part")):
                if not isinstance(holder, dict):
                    continue
                meta = holder.get("metadata")
                model = meta.get("model") if isinstance(meta, dict) else None
                if isinstance(model, dict) and model.get("modelID"):
                    provider = model.get("providerID")
                    last = f"{provider}/{model['modelID']}" if provider else str(model["modelID"])
        return last

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

    def parse_turns(self, path: Path, terminal: dict | None) -> int | None:
        steps = sum(1 for e in iter_json_events(path) if e.get("type") == "step_finish")
        return steps or None

    def classify(self, terminal: dict, stdout_path: Path, stderr_path: Path):
        part = terminal.get("part") if isinstance(terminal.get("part"), dict) else {}
        reason = str(part.get("reason") or terminal.get("reason") or "").lower()
        if reason == "stop":
            return HarnessOutcome.COMPLETED, ""
        if reason == "length":
            return HarnessOutcome.TURN_LIMIT, ""
        if reason in {"error", "content-filter"}:
            detail = read_capture(stderr_path, tail=4000)
            fallback = HarnessOutcome.REFUSED if reason == "content-filter" else HarnessOutcome.FAILED
            return classify_error_text(detail) or fallback, detail
        if reason:
            detail = f"OpenCode ended on a non-terminal step reason: {reason!r}."
            stderr_detail = read_capture(stderr_path, tail=4000).strip()
            if stderr_detail:
                detail = f"{detail}\n{stderr_detail}"
            return HarnessOutcome.FAILED, detail
        return HarnessOutcome.UNKNOWN, ""

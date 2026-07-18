"""Provider-neutral contracts for one coding-agent harness attempt.

Harnesses execute a single agent session. They deliberately do not queue,
retry, supervise inactivity, choose a fallback harness, or release work; task
runners such as Fab own those policy decisions.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable


class HarnessOutcome(StrEnum):
    """Portable termination reasons emitted by agent harness adapters."""

    COMPLETED = "completed"
    TURN_LIMIT = "turn_limit"
    CONTEXT_LIMIT = "context_limit"
    REFUSED = "refused"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    PROVIDER_ERROR = "provider_error"
    AUTH_ERROR = "auth_error"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HarnessCapabilities:
    """Features a harness adapter can honor natively."""

    resume: bool = False
    max_turns: bool = False
    agent_selection: bool = False
    reasoning_effort: bool = False
    streaming_events: bool = True


@dataclass(slots=True)
class HarnessRequest:
    """One harness invocation, independent of any queue or task runner."""

    prompt: str
    cwd: str | Path
    capture_dir: str | Path
    model: str | None = None
    reasoning_effort: str | None = None
    max_turns: int | None = None
    session_id: str | None = None
    agent: str | None = None
    extra: Sequence[str] = field(default_factory=tuple)
    allow_unsafe: bool = False
    correlation_id: str | None = None
    executable: str | Path | None = None
    prompt_via_stdin: bool = False

    def resolved_cwd(self) -> str:
        path = Path(self.cwd).expanduser()
        return str(path if path.is_dir() else Path.home())

    def resolved_capture_dir(self) -> Path:
        path = Path(self.capture_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_executable(self, default: str) -> str:
        """Return a caller-selected CLI path or the adapter's standard command."""

        return str(Path(self.executable).expanduser()) if self.executable else default


@dataclass(slots=True)
class HarnessHandle:
    """A running attempt and its durable capture files."""

    proc: subprocess.Popen
    stdout_fh: IO[bytes]
    stderr_fh: IO[bytes]
    stdout_path: Path
    stderr_path: Path

    def close_captures(self) -> None:
        self.stdout_fh.close()
        self.stderr_fh.close()


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Normalized result from a completed or interrupted attempt."""

    harness: str
    outcome: HarnessOutcome
    final_text: str = ""
    session_id: str | None = None
    exit_code: int | None = None
    detail: str = ""
    usage: dict[str, int | float] = field(default_factory=dict)
    terminal_event: dict[str, Any] | None = None
    correlation_id: str | None = None


@runtime_checkable
class AgentHarness(Protocol):
    name: str
    capabilities: HarnessCapabilities

    def is_available(self) -> bool: ...

    def build_argv(self, request: HarnessRequest) -> list[str]: ...

    def start(self, request: HarnessRequest) -> HarnessHandle: ...

    def parse_terminal(self, path: Path) -> dict | None: ...

    def parse_session_id(self, path: Path) -> str | None: ...

    def extract_final_text(self, path: Path) -> str: ...

    def inspect(
        self,
        stdout_path: Path,
        stderr_path: Path,
        *,
        exit_code: int | None = None,
        correlation_id: str | None = None,
    ) -> HarnessResult: ...


def launch_process(
    argv: list[str],
    request: HarnessRequest,
    *,
    env: dict[str, str] | None = None,
    stdin_data: str | bytes | None = None,
) -> HarnessHandle:
    """Launch an adapter command with durable stdout/stderr captures."""

    capture_dir = request.resolved_capture_dir()
    stdout_path = capture_dir / "stdout"
    stderr_path = capture_dir / "stderr"
    # These captures intentionally remain open for the child process lifetime.
    stdout_fh = open(stdout_path, "wb")  # noqa: SIM115
    stderr_fh = open(stderr_path, "wb")  # noqa: SIM115
    try:
        proc = subprocess.Popen(
            argv,
            cwd=request.resolved_cwd(),
            stdout=stdout_fh,
            stderr=stderr_fh,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            env=env,
        )
        if stdin_data is not None and proc.stdin is not None:
            payload = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
            proc.stdin.write(payload)
            proc.stdin.close()
    except Exception:
        stdout_fh.close()
        stderr_fh.close()
        raise
    return HarnessHandle(proc, stdout_fh, stderr_fh, stdout_path, stderr_path)


def decode_json_events(text: str) -> Iterator[dict[str, Any]]:
    """Yield valid object events from JSON or NDJSON text."""

    if not text.strip():
        return
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        value = None
    if isinstance(value, dict):
        yield value
        return
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            yield value


def iter_json_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield valid object events from a JSON object or NDJSON capture."""

    try:
        text = path.read_text(errors="replace")
    except OSError:
        return
    yield from decode_json_events(text)


_SIGNALS: tuple[tuple[HarnessOutcome, tuple[str, ...]], ...] = (
    (HarnessOutcome.AUTH_ERROR, (
        r"\bhttp_status=40[13]\b", r"unauthorized", r"forbidden",
        r"invalid api key", r"authentication failed", r"bad credentials",
    )),
    (HarnessOutcome.CONTEXT_LIMIT, (
        r"prompt too long", r"context (?:length|window)(?: exceeded)?",
    )),
    (HarnessOutcome.REFUSED, (
        r"content (?:filter|policy)", r"refused", r"policy violat",
    )),
    (HarnessOutcome.RATE_LIMIT, (
        r"\bhttp_status=429\b", r"\b429\b", r"rate ?limit",
        r"too many requests", r"quota exceeded", r"throttl",
    )),
    (HarnessOutcome.PROVIDER_ERROR, (
        r"\bhttp_status=5\d\d\b", r"\b5\d\d server error\b",
        r"service unavailable", r"bad gateway", r"gateway timeout",
        r"upstream", r"overloaded", r"capacity",
    )),
    (HarnessOutcome.NETWORK_ERROR, (
        r"connection (?:reset|refused|closed|aborted)",
        r"temporary failure in name resolution", r"network is unreachable",
        r"\bECONN(?:REFUSED|RESET|ABORTED)\b", r"timed? out", r"timeout",
    )),
)


def classify_error_text(text: str) -> HarnessOutcome | None:
    """Normalize common CLI/provider failures without runner-specific policy."""

    for outcome, patterns in _SIGNALS:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return outcome
    return None


def read_capture(path: Path, *, tail: int | None = None) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-tail:] if tail is not None else text

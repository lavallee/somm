"""Provider-neutral contracts for one coding-agent harness attempt.

Harnesses execute a single agent session. They deliberately do not queue,
retry, supervise inactivity, choose a fallback harness, or release work; task
runners such as Fab own those policy decisions.

The result envelope is versioned (:data:`ENVELOPE_VERSION`). Changes are
additive: a field, once published, keeps its name and meaning, and an older
consumer can read a newer result. Native detail stays in ``terminal_event``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ENVELOPE_VERSION = 2
"""Version of :class:`HarnessResult`. 1 was 0.14–0.16; 2 adds model,
harness_version, duration_ms, turns, retry_after_s, and three outcomes."""


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
    BUDGET_EXHAUSTED = "budget_exhausted"
    # The request itself was rejected (400/422, invalid prompt). Retrying the
    # same input is futile; distinct from a provider that is merely unwell.
    INPUT_ERROR = "input_error"
    # The agent said it cannot proceed (`<stuck>reason</stuck>` as its final
    # output). Retrying the same prompt is futile; a human or a different plan
    # is needed. Distinct from an infrastructure stall, which the outer
    # runner's watchdog detects and this layer never sees.
    AGENT_BLOCKED = "agent_blocked"
    # The process ended without a terminal event and did not exit cleanly:
    # killed, signalled, or cut off. The outer supervisor usually knows why.
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    UNKNOWN = "unknown"


class HarnessCapabilityError(ValueError):
    """A request asks a harness to launch a known-incompatible capability."""


@dataclass(frozen=True, slots=True)
class HarnessCapabilities:
    """Features a harness adapter can honor natively.

    ``validate_request`` fails closed on these: a request field the adapter
    cannot translate is rejected before launch rather than silently dropped.
    """

    resume: bool = False
    max_turns: bool = False
    agent_selection: bool = False
    reasoning_effort: bool = False
    streaming_events: bool = True
    prompt_via_stdin: bool = False
    permission_bypass: bool = True
    reports_usage: bool = True
    reports_cost: bool = False
    reports_model: bool = False
    model_families: frozenset[str] = frozenset()
    # Typed scope the adapter can enforce. An adapter without an OS sandbox
    # declares none of these, and validate_request rejects a request that
    # asks for scope it cannot enforce.
    sandbox_modes: frozenset[str] = frozenset()
    ephemeral: bool = False
    network_sandbox: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "resume": self.resume,
            "max_turns": self.max_turns,
            "agent_selection": self.agent_selection,
            "reasoning_effort": self.reasoning_effort,
            "streaming_events": self.streaming_events,
            "prompt_via_stdin": self.prompt_via_stdin,
            "permission_bypass": self.permission_bypass,
            "reports_usage": self.reports_usage,
            "reports_cost": self.reports_cost,
            "reports_model": self.reports_model,
            "model_families": sorted(self.model_families),
            "sandbox_modes": sorted(self.sandbox_modes),
            "ephemeral": self.ephemeral,
            "network_sandbox": self.network_sandbox,
        }


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
    sandbox_mode: str | None = None
    ephemeral: bool = False
    # None: the adapter's default (Codex: no network). True: allow network
    # inside the sandbox. False: require network to be blocked, which only an
    # adapter with ``network_sandbox`` can guarantee.
    network_access: bool | None = None
    allow_unsafe: bool = False
    correlation_id: str | None = None
    executable: str | Path | None = None
    prompt_via_stdin: bool = False
    # Extra environment for the child, layered over the scrubbed process
    # environment. A runner that must hand the agent a workspace path or a
    # feature flag puts it here instead of relaunching the process itself.
    env: dict[str, str] | None = None
    # Put the child in its own session/process group so a supervisor can
    # cancel or reap the whole tree. Off by default; a plain `run()` does not
    # need it.
    start_new_session: bool = False

    def resolved_cwd(self) -> str:
        path = Path(self.cwd).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(str(path))
        return str(path)

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
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def close_captures(self) -> None:
        self.stdout_fh.close()
        self.stderr_fh.close()


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Normalized result from a completed or interrupted attempt.

    Fields are additive across envelope versions. ``None`` means the harness
    did not say, never a guess.
    """

    harness: str
    outcome: HarnessOutcome
    final_text: str = ""
    session_id: str | None = None
    exit_code: int | None = None
    detail: str = ""
    usage: dict[str, int | float] = field(default_factory=dict)
    billing_usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    terminal_event: dict[str, Any] | None = None
    correlation_id: str | None = None
    # --- envelope 2 ---
    envelope_version: int = ENVELOPE_VERSION
    # The model that actually served the attempt, as the harness reported it.
    model: str | None = None
    # The CLI's own version string, when the capture or the binary says.
    harness_version: str | None = None
    # Wall time of the attempt in milliseconds, when known (the harness
    # reported it, or the caller timed the process).
    duration_ms: int | None = None
    # Agent turns the harness reported.
    turns: int | None = None
    # Seconds until a rate/quota limit lifts, when the harness said so.
    retry_after_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view of the envelope (``terminal_event`` passed through)."""
        return {
            "envelope_version": self.envelope_version,
            "harness": self.harness,
            "harness_version": self.harness_version,
            "outcome": self.outcome.value,
            "model": self.model,
            "final_text": self.final_text,
            "session_id": self.session_id,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "usage": dict(self.usage),
            "billing_usage": dict(self.billing_usage),
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "turns": self.turns,
            "retry_after_s": self.retry_after_s,
            "correlation_id": self.correlation_id,
            "terminal_event": self.terminal_event,
        }


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
        duration_ms: int | None = None,
    ) -> HarnessResult: ...


# ---------------------------------------------------------------------------
# Template adapter


class BaseHarness:
    """Shared skeleton for the built-in adapters.

    An adapter sets ``name``, ``executable``, ``capabilities``, and the env
    variables to scrub, implements ``build_argv`` and the four parsers, and
    overrides ``classify`` for its native terminal event. Everything the four
    adapters used to copy — the terminal scan, the no-terminal fallback, the
    exit-code reading, the result assembly — lives here once.
    """

    name: str = ""
    executable: str = ""
    capabilities: HarnessCapabilities = HarnessCapabilities()
    # API keys the seat must not see. A coding-agent seat is billed to the
    # subscription; a key in the environment would quietly bill the API
    # account instead and attribute the spend to the wrong plan.
    scrub_env: tuple[str, ...] = ()

    _version_cache: dict[str, str | None] = {}

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def build_argv(self, request: HarnessRequest) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def stdin_data(self, request: HarnessRequest) -> str | None:
        return request.prompt if request.prompt_via_stdin else None

    def start(self, request: HarnessRequest) -> HarnessHandle:
        env = scrubbed_env(self.scrub_env)
        if request.env:
            env.update(request.env)
        return launch_process(
            self.build_argv(request),
            request,
            env=env,
            stdin_data=self.stdin_data(request),
            start_new_session=request.start_new_session,
        )

    def version(self) -> str | None:
        """The CLI's version string, cached per process. ``None`` when the
        binary is missing or does not answer ``--version``."""
        key = self.executable
        if key in self._version_cache:
            return self._version_cache[key]
        value: str | None = None
        path = shutil.which(key)
        if path:
            try:
                out = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=10
                )
                text = (out.stdout or out.stderr).strip().splitlines()
                value = text[-1].strip() if text else None
            except Exception:
                value = None
        self._version_cache[key] = value
        return value

    # --- parsers an adapter implements --------------------------------

    def parse_terminal(self, path: Path) -> dict | None:  # pragma: no cover
        raise NotImplementedError

    def parse_session_id(self, path: Path) -> str | None:  # pragma: no cover
        raise NotImplementedError

    def extract_final_text(self, path: Path) -> str:  # pragma: no cover
        raise NotImplementedError

    def parse_model(self, path: Path, terminal: dict | None) -> str | None:
        return None

    def parse_version(self, path: Path, terminal: dict | None) -> str | None:
        """Version from the capture itself, when the harness writes one."""
        return None

    def parse_usage(self, path: Path, terminal: dict | None) -> dict:
        usage = terminal.get("usage") if isinstance(terminal, dict) else None
        return dict(usage) if isinstance(usage, dict) else {}

    def parse_billing_usage(self, path: Path, terminal: dict | None) -> dict:
        return {}

    def parse_cost_usd(self, path: Path, terminal: dict | None) -> float | None:
        return None

    def parse_duration_ms(self, path: Path, terminal: dict | None) -> int | None:
        return None

    def parse_turns(self, path: Path, terminal: dict | None) -> int | None:
        return None

    def classify(
        self, terminal: dict, stdout_path: Path, stderr_path: Path
    ) -> tuple[HarnessOutcome, str]:
        """Outcome and detail for a capture that has a terminal event."""
        raise NotImplementedError  # pragma: no cover

    # --- the shared skeleton -------------------------------------------

    def inspect(
        self,
        stdout_path: Path,
        stderr_path: Path,
        *,
        exit_code: int | None = None,
        correlation_id: str | None = None,
        duration_ms: int | None = None,
    ) -> HarnessResult:
        terminal = self.parse_terminal(stdout_path)
        if terminal is not None:
            outcome, detail = self.classify(terminal, stdout_path, stderr_path)
        else:
            outcome, detail = self.classify_without_terminal(stdout_path, stderr_path, exit_code)
        final_text = self.extract_final_text(stdout_path)
        if outcome is HarnessOutcome.COMPLETED:
            blocked = agent_blocked_reason(final_text)
            if blocked is not None:
                outcome = HarnessOutcome.AGENT_BLOCKED
                detail = blocked
        retry_after = retry_after_from_text(detail) if detail else None
        if retry_after is None and terminal is not None:
            retry_after = self.parse_retry_after(terminal)
        return HarnessResult(
            harness=self.name,
            outcome=outcome,
            final_text=final_text,
            session_id=self.parse_session_id(stdout_path),
            exit_code=exit_code,
            detail=detail.strip(),
            usage=self.parse_usage(stdout_path, terminal),
            billing_usage=self.parse_billing_usage(stdout_path, terminal),
            cost_usd=self.parse_cost_usd(stdout_path, terminal),
            terminal_event=terminal,
            correlation_id=correlation_id,
            model=self.parse_model(stdout_path, terminal),
            harness_version=self.parse_version(stdout_path, terminal) or self.version(),
            duration_ms=(
                duration_ms
                if duration_ms is not None
                else self.parse_duration_ms(stdout_path, terminal)
            ),
            turns=self.parse_turns(stdout_path, terminal),
            retry_after_s=retry_after,
        )

    def parse_retry_after(self, terminal: dict) -> float | None:
        return None

    def classify_without_terminal(
        self, stdout_path: Path, stderr_path: Path, exit_code: int | None
    ) -> tuple[HarnessOutcome, str]:
        """No terminal event: read both captures and let the evidence speak.

        A recognizable provider failure in the tail wins. Otherwise a clean
        exit is UNKNOWN (the harness said nothing, and the supervisor may know
        more), a signal or non-zero exit without a terminal event is
        INTERRUPTED.
        """
        detail = read_capture(stdout_path, tail=4000) + "\n" + read_capture(stderr_path, tail=4000)
        classified = classify_error_text(detail)
        if classified is not None:
            return classified, detail
        if exit_code is None or exit_code == 0:
            return HarnessOutcome.UNKNOWN, detail
        return HarnessOutcome.INTERRUPTED, detail


# ---------------------------------------------------------------------------
# Helpers


def infer_model_family(model: str | None) -> str | None:
    """Return a known provider family while leaving future model names open."""

    if not model:
        return None
    value = model.strip().lower()
    if not value:
        return None
    if (
        value.startswith(("anthropic/", "claude-"))
        or re.fullmatch(r"(?:sonnet|opus|haiku)(?:[-._].*)?", value)
    ):
        return "anthropic"
    if (
        value.startswith(("openai/", "gpt-", "chatgpt-", "codex-"))
        or re.fullmatch(r"o\d+(?:[-._].*)?", value)
    ):
        return "openai"
    return None


def scrubbed_env(drop: Sequence[str] = ()) -> dict[str, str]:
    """A copy of the process environment without the named variables."""
    env = os.environ.copy()
    for key in drop:
        env.pop(key, None)
    return env


def launch_process(
    argv: list[str],
    request: HarnessRequest,
    *,
    env: dict[str, str] | None = None,
    stdin_data: str | bytes | None = None,
    start_new_session: bool = False,
) -> HarnessHandle:
    """Launch an adapter command with durable stdout/stderr captures."""

    capture_dir = request.resolved_capture_dir()
    stdout_path = capture_dir / "stdout"
    stderr_path = capture_dir / "stderr"
    # These captures intentionally remain open for the child process lifetime.
    stdout_fh = open(stdout_path, "wb")  # noqa: SIM115
    stderr_fh = open(stderr_path, "wb")  # noqa: SIM115
    try:
        started_at = datetime.now(UTC)
        proc = subprocess.Popen(
            argv,
            cwd=request.resolved_cwd(),
            stdout=stdout_fh,
            stderr=stderr_fh,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            env=env,
            start_new_session=start_new_session,
        )
        if stdin_data is not None and proc.stdin is not None:
            payload = stdin_data.encode() if isinstance(stdin_data, str) else stdin_data
            proc.stdin.write(payload)
            proc.stdin.close()
    except Exception:
        stdout_fh.close()
        stderr_fh.close()
        raise
    return HarnessHandle(proc, stdout_fh, stderr_fh, stdout_path, stderr_path, started_at)


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
    (HarnessOutcome.BUDGET_EXHAUSTED, (
        r"\bstatuscode[\"'\s:=]+402\b", r"\bhttp_status=402\b",
        r"\b402\b[^\n]{0,80}\bpayment required\b",
        r"\binsufficient (?:credits?|balance)\b",
    )),
    (HarnessOutcome.AUTH_ERROR, (
        r"\bhttp_status=40[13]\b", r"unauthorized", r"forbidden",
        r"invalid api key", r"authentication failed", r"bad credentials",
    )),
    (HarnessOutcome.CONTEXT_LIMIT, (
        r"prompt too long", r"context (?:length|window)(?: exceeded)?",
    )),
    (HarnessOutcome.REFUSED, (
        r"content was flagged for possible cybersecurity risk",
        r"content (?:filter|policy)", r"refused", r"policy violat",
    )),
    (HarnessOutcome.RATE_LIMIT, (
        r"\bhttp_status=429\b", r"\b429\b", r"rate ?limit",
        r"too many requests", r"quota exceeded", r"throttl",
    )),
    (HarnessOutcome.INPUT_ERROR, (
        r"\bhttp_status=4(?:00|22)\b", r"\binvalid (?:request|parameter|prompt)\b",
        r"\bunprocessable entity\b",
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


_AGENT_STUCK_RE = re.compile(r"<stuck>(.*?)</stuck>", re.DOTALL | re.IGNORECASE)


def agent_blocked_reason(final_text: str) -> str | None:
    """The agent's own `<stuck>reason</stuck>` escape hatch, if it used it."""
    match = _AGENT_STUCK_RE.search(final_text or "")
    if not match:
        return None
    reason = match.group(1).strip()
    return f"agent reported it cannot proceed: {reason}" if reason else "agent reported it cannot proceed"


_RETRY_AFTER_RE = re.compile(r"retry[\s\-]?after[:=\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)
_RESET_CLOCK_RE = re.compile(
    r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^)]+)\)", re.IGNORECASE
)


def retry_after_from_text(text: str, *, now: datetime | None = None) -> float | None:
    """Seconds until a limit lifts, parsed from provider wording.

    Understands ``retry-after: N`` and the Claude seat's "resets 3pm
    (America/New_York)". A stated reset with an unparseable clock gets a
    conservative hour. ``None`` when the text says nothing about timing.
    """
    match = _RETRY_AFTER_RE.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    if not re.search(r"\bresets?\b", text, re.IGNORECASE):
        return None
    clock = _RESET_CLOCK_RE.search(text)
    if not clock:
        return 3600.0
    hour, minute, half, zone = clock.groups()
    try:
        if not 1 <= int(hour) <= 12 or not 0 <= int(minute or 0) < 60:
            raise ValueError("invalid reset clock")
        local = (now or datetime.now(UTC)).astimezone(ZoneInfo(zone))
        target = local.replace(
            hour=int(hour) % 12 + (12 if half.lower() == "pm" else 0),
            minute=int(minute or 0),
            second=0,
            microsecond=0,
        )
        if target <= local:
            target += timedelta(days=1)
        return (target - local).total_seconds() + 60
    except (ValueError, ZoneInfoNotFoundError):
        return 3600.0


def read_capture(path: Path, *, tail: int | None = None) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-tail:] if tail is not None else text


def elapsed_ms(started_at: datetime | None) -> int | None:
    if started_at is None:
        return None
    return max(0, int((datetime.now(UTC) - started_at).total_seconds() * 1000))


__all__ = [
    "ENVELOPE_VERSION",
    "AgentHarness",
    "BaseHarness",
    "HarnessCapabilities",
    "HarnessCapabilityError",
    "HarnessHandle",
    "HarnessOutcome",
    "HarnessRequest",
    "HarnessResult",
    "agent_blocked_reason",
    "classify_error_text",
    "decode_json_events",
    "elapsed_ms",
    "infer_model_family",
    "iter_json_events",
    "launch_process",
    "read_capture",
    "retry_after_from_text",
    "scrubbed_env",
]

"""Typed data shapes shared across packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    OK = "ok"
    BAD_JSON = "bad_json"
    EMPTY = "empty"
    OFF_TASK = "off_task"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UPSTREAM_ERROR = "upstream_error"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


# Back-compat alias for early code; prefer Outcome going forward.
CallOutcome = Outcome


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"


@dataclass(frozen=True, slots=True)
class Workload:
    id: str  # content-addressed (hash of name+schemas)
    name: str
    description: str = ""
    input_schema: dict | None = None
    output_schema: dict | None = None
    quality_criteria: list[str] = field(default_factory=list)
    budget_cap_usd_daily: float | None = None
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL
    # Capabilities every call for this workload requires of the serving
    # (provider, model). See somm_core.parse.infer_capabilities — these are
    # merged with what the prompt self-advertises at dispatch time.
    capabilities_required: list[str] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str  # content-addressed (hash of body)
    workload_id: str
    version: str  # "v1", "v1.2", etc
    hash: str
    body: str
    created_at: datetime | None = None
    retired_at: datetime | None = None


@dataclass(slots=True)
class Call:
    """A row in `calls`. Immutable after insert — late data goes in `call_updates`."""

    id: str  # UUID4
    ts: datetime
    project: str
    workload_id: str | None  # None in demo mode w/ auto-registered ad_hoc
    prompt_id: str | None
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    outcome: Outcome
    error_kind: str | None
    prompt_hash: str
    response_hash: str


@dataclass(slots=True)
class SommResult:
    """Return shape of SommLLM.generate()."""

    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    call_id: str
    outcome: Outcome = Outcome.OK
    error_kind: str | None = None
    raw: dict[str, Any] | None = None

    def mark(self, outcome: Outcome) -> SommResult:
        """Post-tag a call's outcome. Returns self for chaining."""
        self.outcome = outcome
        return self

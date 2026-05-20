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

    @property
    def failure_class(self) -> "FailureClass":
        """Adequacy-tier classification — see FailureClass docstring."""
        return _OUTCOME_TO_FAILURE_CLASS.get(self, FailureClass.UNKNOWN)

    @property
    def is_capability_signal(self) -> bool:
        """True when the failure is the model's fault (Tier 2/3 in steve's framing).

        Use to ask "is this model unfit for this workload?" — exclude detractors,
        which reflect provider/network state, not model capability.
        """
        return self.failure_class.is_capability_signal

    @property
    def is_detractor(self) -> bool:
        """True when the failure is a provider/network/operational issue.

        Detractors are reasons to try other approaches (or wait), but not
        evidence that the model itself can't do the workload.
        """
        return self.failure_class.is_detractor


class FailureClass(StrEnum):
    """Adequacy tier for a call's outcome.

    Splits the existing :class:`Outcome` enum into two axes that admin
    queries care about distinctly:

    * ``capability_*`` — the model itself produced something inadequate
      (no output, broken JSON, off-task content). Evidence the model is
      unfit for this workload.
    * ``detractor_*`` — the provider/network failed (rate limit, 5xx,
      timeout). Reason to try other approaches, but not capability
      evidence — a model is innocent until proven model-fault.
    * ``meta_*`` / ``none`` / ``unknown`` — neither bucket.

    The split mirrors steve's reporter's-notebook framework (timeliness
    vs. model-traceable error vs. payload error vs. subjective quality).
    Subjective quality is intentionally absent — it lives in
    ``eval_results``, not in this classification.
    """

    NONE = "none"
    CAPABILITY_PAYLOAD = "capability_payload"
    CAPABILITY_EMPTY = "capability_empty"
    DETRACTOR_TIMEOUT = "detractor_timeout"
    DETRACTOR_RATE_LIMIT = "detractor_rate_limit"
    DETRACTOR_UPSTREAM = "detractor_upstream"
    META_EXHAUSTED = "meta_exhausted"
    UNKNOWN = "unknown"

    @property
    def is_capability_signal(self) -> bool:
        return self.value.startswith("capability_")

    @property
    def is_detractor(self) -> bool:
        return self.value.startswith("detractor_")


_OUTCOME_TO_FAILURE_CLASS: dict[Outcome, FailureClass] = {
    Outcome.OK: FailureClass.NONE,
    Outcome.BAD_JSON: FailureClass.CAPABILITY_PAYLOAD,
    Outcome.OFF_TASK: FailureClass.CAPABILITY_PAYLOAD,
    Outcome.EMPTY: FailureClass.CAPABILITY_EMPTY,
    Outcome.TIMEOUT: FailureClass.DETRACTOR_TIMEOUT,
    Outcome.RATE_LIMIT: FailureClass.DETRACTOR_RATE_LIMIT,
    Outcome.UPSTREAM_ERROR: FailureClass.DETRACTOR_UPSTREAM,
    Outcome.EXHAUSTED: FailureClass.META_EXHAUSTED,
    Outcome.UNKNOWN: FailureClass.UNKNOWN,
}


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
    # Adequacy thresholds (schema v6+). Make "is this model performing
    # adequately?" queryable rather than judgment-call. None = unset.
    max_p95_latency_ms: int | None = None              # Tier 1: timeliness
    max_capability_failure_rate: float | None = None   # Tier 2/3: 0–1 (e.g. 0.05 = 5%)
    max_cost_per_call_usd: float | None = None         # cost ceiling per ok call
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
    error_detail: str | None = None
    # 0007: commission_id ties the call to a scribe commission (cross-tool
    # audit spine); the param fields record what the caller asked for.
    commission_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences_json: str | None = None


@dataclass(slots=True)
class Decision:
    """Advisory memory: a question asked, candidates considered, a choice made.

    Mirrored across projects by default — the whole point is to remember
    past reasoning when the same question comes up elsewhere.
    """

    id: str  # UUID4
    ts: datetime
    project: str
    question: str
    question_hash: str
    candidates: list[dict]
    rationale: str
    chosen_provider: str | None = None
    chosen_model: str | None = None
    workload_id: str | None = None
    workload_name: str | None = None
    constraints: dict | None = None
    agent: str | None = None
    superseded_by: str | None = None
    outcome_note: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model.

    `id` is the provider-assigned identifier (Anthropic `tool_use_id`,
    OpenAI `tool_calls[].id`). Callers correlate tool_results back to
    the request with this id when constructing the next turn.

    `arguments` is the parsed JSON object the model produced — callers
    do not need to `json.loads` again. Malformed arguments surface as an
    empty dict with `arguments_raw` populated so the caller can repair.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    arguments_raw: str = ""  # populated only when arguments failed to parse as JSON


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
    # Human-readable error body / stack summary. Truncated to 512 chars to
    # keep telemetry rows bounded. Populated whenever outcome != OK.
    error_detail: str | None = None
    raw: dict[str, Any] | None = None
    # Tool calls the model requested in this turn. Empty when no tools
    # were offered or the model chose to respond with text only.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" | "" when unknown.
    # Callers building an agent loop check `stop_reason == "tool_use"` to
    # know to invoke the tools and submit a follow-up turn.
    stop_reason: str = ""

    def mark(self, outcome: Outcome) -> SommResult:
        """Post-tag a call's outcome. Returns self for chaining."""
        self.outcome = outcome
        return self


@dataclass(slots=True)
class EmbedResult:
    """Return shape of SommLLM.embed().

    Mirrors SommResult so telemetry, provenance, and error semantics are
    consistent with the generate() path. `embedding` is the vector;
    `dim` is its length, surfaced for quick sanity checks at call sites.
    On failure, `embedding` is empty and `outcome != OK`.
    """

    embedding: list[float]
    provider: str
    model: str
    dim: int
    tokens_in: int
    latency_ms: int
    cost_usd: float
    call_id: str
    outcome: Outcome = Outcome.OK
    error_kind: str | None = None
    error_detail: str | None = None
    raw: dict[str, Any] | None = None

    def mark(self, outcome: Outcome) -> EmbedResult:
        self.outcome = outcome
        return self

"""Record a harness attempt in the same ledger as every other call.

A coding-agent attempt is the most expensive kind of call a project makes,
and until now it was the one kind the ledger never saw: harness adapters
normalized the attempt and handed the result to a runner, which kept its own
receipts. Fleet rollups then reported seat spend as $0, and a question such
as "what did this job cost, across the planner call and the agent attempt?"
could not be answered from one place.

``record_attempt`` writes one ``calls`` row per attempt with
``origin="harness"``. The provider is the harness name (``claude-cli``,
``codex``, ``opencode``, ``pi``), which is also the seat the attempt ran on;
the model is the one the harness reported; the cost is reported when the
harness states it and computed from model_intel otherwise; the session id
and the caller's correlation id carry through, so a runner's receipt and the
ledger row name the same attempt.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from somm_core import Outcome, cost_for_call
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.cost import cost_provenance
from somm_core.models import Call
from somm_core.parse import stable_hash
from somm_core.repository import Repository
from somm_core.version import VERSION

from somm import hooks

from .base import HarnessOutcome, HarnessRequest, HarnessResult

DEFAULT_WORKLOAD = "harness_attempt"

# Portable termination → ledger outcome. The harness vocabulary is richer
# than the call vocabulary; error_kind keeps the harness word.
_OUTCOME_MAP: dict[HarnessOutcome, Outcome] = {
    HarnessOutcome.COMPLETED: Outcome.OK,
    HarnessOutcome.TURN_LIMIT: Outcome.OK,
    HarnessOutcome.CONTEXT_LIMIT: Outcome.UPSTREAM_ERROR,
    HarnessOutcome.REFUSED: Outcome.OFF_TASK,
    HarnessOutcome.RATE_LIMIT: Outcome.RATE_LIMIT,
    HarnessOutcome.NETWORK_ERROR: Outcome.UPSTREAM_ERROR,
    HarnessOutcome.PROVIDER_ERROR: Outcome.UPSTREAM_ERROR,
    HarnessOutcome.AUTH_ERROR: Outcome.UPSTREAM_ERROR,
    HarnessOutcome.BUDGET_EXHAUSTED: Outcome.EXHAUSTED,
    HarnessOutcome.INPUT_ERROR: Outcome.UPSTREAM_ERROR,
    HarnessOutcome.AGENT_BLOCKED: Outcome.OFF_TASK,
    HarnessOutcome.INTERRUPTED: Outcome.TIMEOUT,
    HarnessOutcome.FAILED: Outcome.UPSTREAM_ERROR,
    HarnessOutcome.UNKNOWN: Outcome.UNKNOWN,
}


def _tokens(result: HarnessResult) -> tuple[int, int, int | None, int | None, int | None]:
    """(tokens_in, tokens_out, cache_read, cache_write, reasoning) from whichever
    usage shape the harness produced. Billing totals win over the terminal
    snapshot when both exist (OpenCode)."""
    usage = result.billing_usage or result.usage or {}
    cache = usage.get("cache") if isinstance(usage.get("cache"), dict) else {}

    def pick(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    tokens_in = pick("prompt_tokens_total", "input_tokens", "input", "prompt_tokens") or 0
    tokens_out = pick("output_tokens", "output", "completion_tokens") or 0
    cache_read = pick("cache_read_input_tokens", "cached_input_tokens")
    if cache_read is None and isinstance(cache.get("read"), (int, float)):
        cache_read = int(cache["read"])
    cache_write = pick("cache_creation_input_tokens", "cache_write_input_tokens")
    if cache_write is None and isinstance(cache.get("write"), (int, float)):
        cache_write = int(cache["write"])
    reasoning = pick("reasoning_tokens", "reasoning_output_tokens", "reasoning")
    # Codex and OpenCode report input exclusive of cache reads; the ledger's
    # tokens_in is the whole prompt, as for every other provider.
    if result.harness in ("codex", "opencode") and cache_read:
        tokens_in += cache_read
    return tokens_in, tokens_out, cache_read, cache_write, reasoning


def record_attempt(
    result: HarnessResult,
    *,
    request: HarnessRequest | None = None,
    project: str | None = None,
    workload: str = DEFAULT_WORKLOAD,
    repo: Repository | None = None,
    config: Config | None = None,
    started_at: datetime | None = None,
) -> Call:
    """Write one ``calls`` row for a finished harness attempt and return it.

    ``project`` defaults to the configured project (``SOMM_PROJECT`` or the
    cwd's ``[tool.somm]``); pass it explicitly from a runner that serves many
    projects. The workload is auto-registered in observe mode. Never raises
    into the caller's control flow for a ledger problem: a runner's own
    receipt is still written when the ledger is unavailable.
    """
    cfg = config or load_config(
        project=project,
        cwd=Path(request.cwd) if request is not None else None,
    )
    repo = repo or Repository(cfg.db_path)
    wl = repo.workload_by_name(workload, cfg.project)
    if wl is None:
        wl = repo.register_workload(
            name=workload,
            project=cfg.project,
            description="Autonomous coding-agent attempts recorded through somm.harnesses",
        )

    tokens_in, tokens_out, cache_read, cache_write, reasoning = _tokens(result)
    provider = result.harness
    model = result.model or (request.model if request is not None else None) or ""
    reported = result.cost_usd is not None
    cost = (
        float(result.cost_usd)
        if reported
        else cost_for_call(repo, provider, model, tokens_in, tokens_out)
    )
    cost_basis, cost_kind, cost_accuracy, cost_source, pricing_version = cost_provenance(
        repo, provider, model, reported=reported, source=f"harness:{provider}" if reported else None
    )
    outcome = _OUTCOME_MAP.get(result.outcome, Outcome.UNKNOWN)
    error_kind = None if outcome is Outcome.OK else result.outcome.value
    detail = result.detail[:512] if result.detail else None
    ts = started_at or datetime.now(UTC)
    call = Call(
        id=str(uuid.uuid4()),
        ts=ts,
        project=cfg.project,
        workload_id=wl.id,
        prompt_id=None,
        provider=provider,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=int(result.duration_ms or 0),
        cost_usd=cost,
        outcome=outcome,
        error_kind=error_kind,
        error_detail=detail,
        prompt_hash=stable_hash(request.prompt) if request is not None else "",
        response_hash=stable_hash(result.final_text),
        correlation_id=result.correlation_id or hooks.current_correlation_id(),
        session_id=result.session_id,
        cache_tokens_in=cache_read,
        cache_tokens_out=cache_write,
        reasoning_tokens=reasoning,
        cost_basis=cost_basis,
        cost_kind=cost_kind,
        cost_accuracy=cost_accuracy,
        cost_source=cost_source,
        pricing_version=pricing_version,
        origin="harness",
        call_site=hooks.current_call_site(),
        client_version=VERSION,
    )
    repo.write_call(call)
    return call


__all__ = ["DEFAULT_WORKLOAD", "record_attempt"]

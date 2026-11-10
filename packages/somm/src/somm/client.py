"""SommLLM — the library entry point.

``somm.llm()`` returns a :class:`SommLLM`: one object that resolves the
workload, applies pre-call hooks and the budget gate, dispatches to a pinned
provider or through the router chain, and records one ``calls`` row per
attempt. ``generate``, ``stream``, ``embed``, and the structured-output
wrappers share that path; the ``a*`` variants run it off the event loop.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import shlex
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from somm_core import EmbedResult, Outcome, SommResult, cost_for_call
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.cost import cost_provenance as _cost_provenance
from somm_core.models import Call, MarkHook, Prompt
from somm_core.models import ToolCall as CoreToolCall
from somm_core.parse import (
    ThinkStreamStripper,
    extract_cache_tokens,
    extract_citations,
    extract_json,
    extract_reasoning_tokens,
    infer_capabilities,
    stable_hash,
)
from somm_core.parse import (
    prompt_id as _prompt_id,
)
from somm_core.plans import load_plans
from somm_core.pricing import seed_known_pricing, sync_bundled_pricing
from somm_core.registry import fleet_db_paths, register_project
from somm_core.repository import Repository
from somm_core.version import VERSION as _CLIENT_VERSION

from somm import hooks
from somm._redaction import scrub_text
from somm.errors import (
    SommEmptyResponse,
    SommError,
    SommProviderError,
    SommProviderNotConfigured,
    SommProvidersExhausted,
    SommRateLimited,
    SommRefused,
    SommStructuredError,
    SommTimeout,
)
from somm.errors import SommStrictMode as _SommStrictMode
from somm.events import ReasoningDelta, StreamDone, StreamEvent, TextDelta
from somm.prompts import fork_prompt as fork_prompt_version
from somm.prompts import (
    get_prompt,
    prompt_ids_for_workload,
    register_prompt,
    resolve_label,
    set_label,
)
from somm.providers.base import (
    SommEmbedRequest,
    SommProvider,
    SommRequest,
    SommResponse,
    ToolCallDelta,
)
from somm.providers.registry import (
    BUILTIN_PROVIDER_SPECS,
    ProviderSpec,
    load_entrypoint_provider_specs,
)
from somm.routing import (
    ProviderHealthTracker,
    Router,
    _parse_ts,
    reasoning_ate_budget,
)
from somm.slots import parallel_slots as _parallel_slots
from somm.telemetry import WriterQueue

SommStrictMode = _SommStrictMode  # re-export; new canonical lives in somm.errors

# Track which (workload, date) pairs have already emitted a budget-exceeded
# warning so we only warn once per workload per day per process.
_warned_budget_exceeded: set[tuple[str, date]] = set()

# One in-process scheduler per DB per process (SOMM_INPROCESS_WORKERS=1).
_inprocess_schedulers: dict[str, object] = {}

# DBs we've already warned about a configured-but-dormant intelligence loop.
_warned_dormant_loop: set[str] = set()

# Global mirror paths we've already warned about when setup failed.
_warned_mirror_unavailable: set[str] = set()

# Providers we've already emitted a plan-pacing warning for this process.
_warned_plan_pace: set[str] = set()

# Project registry entries already announced by this client process.
_registered_project_keys: set[tuple[str, str]] = set()
_registered_project_keys_lock = threading.Lock()
_WAIT_UNSET: Any = object()
_STRUCTURED_TEMPERATURE_JITTER = 0.05


def _next_stream_item(iterator: Iterator) -> tuple[bool, Any]:
    """Advance a sync stream without leaking StopIteration through a Future."""
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


def _pydantic_base_model():
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    return BaseModel


def _looks_like_pydantic_model(schema: Any) -> bool:
    return (
        isinstance(schema, type)
        and hasattr(schema, "model_validate")
        and hasattr(schema, "model_json_schema")
    )


def _classify_structured_schema(schema: Any) -> tuple[str, Any | None]:
    if schema is None:
        return "any", None
    if _looks_like_pydantic_model(schema):
        base_model = _pydantic_base_model()
        if base_model is None:
            raise ValueError(
                "generate_structured(schema=...) received a Pydantic model, "
                "but pydantic is not installed. Install with: pip install 'somm[pydantic]'"
            )
        if issubclass(schema, base_model):
            return "pydantic", schema.model_json_schema()
    if isinstance(schema, dict):
        return "json_schema", schema
    if callable(schema):
        return "callable", None
    raise TypeError(
        "generate_structured(schema=...) expects a Pydantic BaseModel subclass, "
        "a JSON Schema dict, or a callable validator"
    )


def _structured_system(system: str, schema_doc: Any | None) -> str:
    if schema_doc is None:
        return system
    schema_text = json.dumps(schema_doc, sort_keys=True, separators=(",", ":"))
    suffix = f"Respond with ONLY valid JSON matching this schema: {schema_text}"
    return f"{system.rstrip()}\n\n{suffix}" if system else suffix


def _structured_retry_system(system: str, raw: str, error: str) -> str:
    feedback = (
        "Your previous response failed: "
        f"{error}\n\nPrevious raw output:\n{raw}\n\nReturn corrected JSON."
    )
    return f"{system.rstrip()}\n\n{feedback}" if system else feedback


def _json_schema_response_format(schema_doc: Any | None) -> dict | None:
    if not isinstance(schema_doc, dict):
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "somm_structured_output",
            "schema": schema_doc,
            "strict": True,
        },
    }


def _validate_structured_json_schema(parsed: dict | list, schema: dict) -> dict | list:
    try:
        import jsonschema
    except ImportError:
        _validate_structured_json_schema_light(parsed, schema)
        return parsed

    jsonschema.validate(instance=parsed, schema=schema)
    return parsed


def _validate_structured_json_schema_light(parsed: dict | list, schema: dict) -> None:
    expected_type = schema.get("type")
    if expected_type is None and (
        "properties" in schema or "required" in schema or "additionalProperties" in schema
    ):
        expected_type = "object"
    if isinstance(expected_type, list):
        expected_types = set(expected_type)
    elif isinstance(expected_type, str):
        expected_types = {expected_type}
    else:
        expected_types = set()

    if expected_types & {"object", "array"}:
        type_matches = ("object" in expected_types and isinstance(parsed, dict)) or (
            "array" in expected_types and isinstance(parsed, list)
        )
        if not type_matches:
            expected = " or ".join(sorted(expected_types & {"object", "array"}))
            raise ValueError(
                f"JSON Schema validation failed: expected top-level {expected}. "
                "Install somm[jsonschema] for full JSON Schema validation."
            )
    if expected_types and expected_types.isdisjoint({"object", "array"}):
        raise ValueError(
            "JSON Schema validation failed: extract_json only returns object or array values. "
            "Install somm[jsonschema] for full JSON Schema validation."
        )

    required = schema.get("required", [])
    if required:
        if not isinstance(parsed, dict):
            raise ValueError(
                "JSON Schema validation failed: required keys need a top-level object. "
                "Install somm[jsonschema] for full JSON Schema validation."
            )
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(
                "JSON Schema validation failed: missing required key(s): "
                f"{', '.join(map(str, missing))}. Install somm[jsonschema] for full "
                "JSON Schema validation."
            )


def _format_structured_failure(
    last_raw: str, last_error: str, *, delivery_failure: bool = False
) -> str:
    if delivery_failure:
        return (
            "SOMM_STRUCTURED_OUTPUT_FAILED\n\n"
            "Problem: generate_structured() got no model output to validate.\n"
            f"Cause: {last_error}\n"
            "Fix: This is a delivery failure (throttling, an upstream error, or an "
            "exhausted chain), not a schema failure. Back off and retry, space out "
            "burst calls, or pick another provider; do not loosen the schema.\n"
            "Docs: docs/errors/SOMM_STRUCTURED_OUTPUT_FAILED.md"
        )
    return (
        "SOMM_STRUCTURED_OUTPUT_FAILED\n\n"
        "Problem: generate_structured() could not produce JSON matching the supplied schema.\n"
        f"Cause: Last validation error: {last_error}\n"
        f"Last raw text: {last_raw}\n"
        "Fix: Return only valid JSON matching the requested schema, or relax/replace "
        "the schema validator.\n"
        "Docs: docs/errors/SOMM_STRUCTURED_OUTPUT_FAILED.md"
    )


def _mark_bad_json(result: SommResult, error: str) -> None:
    """Record that a delivered answer was not the JSON the caller needed."""
    result.error_kind = "SommStructuredError"
    result.error_detail = f"structured output: {_scrub_secrets(error)}"[:512]
    result.mark(Outcome.BAD_JSON)


# Stop reasons meaning the model or the provider's filter declined to answer.
_REFUSAL_STOP_REASONS = frozenset({"refusal", "content_filter"})


def _failure_error(
    result: SommResult, exc: BaseException | None, *, allow_empty: bool
) -> SommError | None:
    """The typed error ``raise_on_failure=True`` raises for ``result``, or None
    when the call delivered. It names the route and the ledger row."""
    if result.stop_reason in _REFUSAL_STOP_REASONS:
        err: SommError = SommRefused(
            f"{result.provider}/{result.model} declined to answer "
            f"(stop_reason={result.stop_reason})",
            stop_reason=result.stop_reason,
        )
    elif result.outcome == Outcome.OK or (result.outcome == Outcome.EMPTY and allow_empty):
        return None
    elif result.outcome == Outcome.EMPTY:
        err = SommEmptyResponse(result.error_detail or "empty response")
    elif isinstance(exc, SommError):
        err = exc
    else:
        detail = result.error_detail or result.outcome.value
        err = SommTimeout(detail) if isinstance(exc, TimeoutError) else SommProviderError(detail)
        err.__cause__ = exc
    err.provider = err.provider or result.provider or None
    err.model = err.model or result.model or None
    err.call_id = result.call_id
    return err


def _register_project_once(project: str, db_path: Path) -> None:
    key = (project, str(Path(db_path).resolve()))
    with _registered_project_keys_lock:
        if key in _registered_project_keys:
            return
        register_project(project, db_path)
        _registered_project_keys.add(key)


def _build_plan_governor(config: Config):
    """PlanGovernor from ~/.somm/plans.toml, or None when no plan declares
    metered limits (the common case — zero overhead on the call path).

    A malformed plans.toml warns loudly and disables pacing rather than
    breaking `somm.llm()`: a config typo shouldn't take production down,
    but silently ignoring declared quotas would be worse than either.
    """
    from somm.plan_governor import PlanGovernor

    try:
        plans = load_plans()
    except Exception as exc:
        print(f"[somm] plans.toml is invalid — plan pacing disabled: {exc}", file=sys.stderr)
        return None
    governor = PlanGovernor(plans, lambda: fleet_db_paths(include=config.db_path))
    return governor if governor.has_paceable_limits() else None


def _warn_if_plans_off_pace(governor) -> None:
    """One line per provider per process when a metered plan is exhausted
    or past its soft target and burning faster than the window passes."""
    if governor is None:
        return
    try:
        for st in governor.statuses():
            if st.provider in _warned_plan_pace or st.state == "ok":
                continue
            _warned_plan_pace.add(st.provider)
            print(
                f"[somm] {st.provider} plan {st.plan_name or '(metered)'}: "
                f"{st.used_pct:.0f}% of {st.limit.window} {st.limit.unit} quota used "
                f"with {100 - st.elapsed_pct:.0f}% of the window left "
                f"(pace {st.pace_ratio:.1f}x) — see `somm plans`.",
                file=sys.stderr,
            )
    except Exception:
        pass


def _maybe_start_inprocess_workers(config: Config, repo: Repository):
    """Start the worker scheduler inside this process, once per DB.

    Long-lived processes that set SOMM_INPROCESS_WORKERS=1 get the full
    intelligence loop (model-intel refresh, online-eval grading,
    recommendations) without running `somm serve`. Short-lived processes
    should use `somm loop install` instead.
    """
    key = str(config.db_path)
    if key in _inprocess_schedulers:
        return _inprocess_schedulers[key]
    from somm.workers.inprocess import start_inprocess_scheduler

    try:
        scheduler = start_inprocess_scheduler(config, repo)
    except Exception as exc:
        print(f"[somm] in-process workers failed to start: {exc}", file=sys.stderr)
        return None
    _inprocess_schedulers[key] = scheduler
    return scheduler


def _warn_if_intelligence_loop_dormant(config: Config, repo: Repository) -> None:
    """One-line heads-up when online eval is configured but nothing grades it.

    The most common somm deployment gap: a workload opts into shadow
    sampling, but no scheduler ever runs, so calls pile up ungraded and
    the recommendation loop stays silent forever. Warn once per DB per
    process; never raise.
    """
    key = str(config.db_path)
    if key in _warned_dormant_loop:
        return
    _warned_dormant_loop.add(key)
    try:
        with repo._open() as conn:
            shadow_n = conn.execute(
                "SELECT COUNT(*) FROM workloads WHERE shadow_config_json IS NOT NULL"
            ).fetchone()[0]
            if not shadow_n:
                return
            heartbeat_n, newest_run_at, is_stale = conn.execute(
                "SELECT COUNT(*), MAX(last_run_at), "
                "       CASE "
                "         WHEN MAX(last_run_at) IS NOT NULL "
                "          AND julianday('now') - julianday(MAX(last_run_at)) > 7 "
                "         THEN 1 ELSE 0 "
                "       END "
                "FROM worker_heartbeat"
            ).fetchone()
        if heartbeat_n == 0:
            print(
                f"[somm] online eval is configured for {shadow_n} workload(s) "
                f"but no worker has ever run — sampled calls are piling up "
                f"ungraded. Run `somm loop run`, or `somm loop install` for a "
                f"timer; long-lived processes can set SOMM_INPROCESS_WORKERS=1.",
                file=sys.stderr,
            )
        elif is_stale:
            print(
                f"[somm] online eval is configured for {shadow_n} workload(s) "
                f"but workers have not run since {newest_run_at} — sampled "
                f"calls may be piling up ungraded. Run `somm loop run` or "
                f"`somm loop install`.",
                file=sys.stderr,
            )
    except Exception:
        pass


def _call_event(
    *,
    call_id: str,
    correlation_id: str | None,
    project: str,
    workload: str,
    provider: str,
    model: str,
    outcome: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cost_usd: float,
    temperature: float | None = None,
    max_tokens: int | None = None,
    error_kind: str | None = None,
    short_circuited: str | None = None,
    cost_basis: str = "unknown",
    cost_kind: str = "unknown",
    cost_accuracy: str = "unknown",
    cost_source: str | None = None,
    pricing_version: str | None = None,
    observation_role: str = "production",
    source_call_id: str | None = None,
    eval_result_id: int | None = None,
    provider_request_id: str | None = None,
    billing_id: str | None = None,
    origin: str = "native",
    budget_eligible: bool = True,
    cache_tokens_in: int | None = None,
    cache_tokens_out: int | None = None,
    reasoning_tokens: int | None = None,
    latency_ttft_ms: int | None = None,
) -> dict:
    """Assemble the observer event for one completed call.

    The full row stays in somm's calls.sqlite; this event is what external
    audit hooks receive (see somm.hooks). Keys are stable — additions only.
    """
    return {
        "call_id": call_id,
        "correlation_id": correlation_id,
        "project": project,
        "workload": workload,
        "provider": provider,
        "model": model,
        "outcome": outcome,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "cost_usd": round(cost_usd, 6) if cost_usd else 0.0,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "error_kind": error_kind,
        "short_circuited": short_circuited,
        "cost_basis": cost_basis,
        "cost_kind": cost_kind,
        "cost_accuracy": cost_accuracy,
        "cost_source": cost_source,
        "pricing_version": pricing_version,
        "observation_role": observation_role,
        "source_call_id": source_call_id,
        "eval_result_id": eval_result_id,
        "provider_request_id": provider_request_id,
        "billing_id": billing_id,
        "origin": origin,
        "budget_eligible": budget_eligible,
        "cache_tokens_in": cache_tokens_in,
        "cache_tokens_out": cache_tokens_out,
        "reasoning_tokens": reasoning_tokens,
        "ttft_ms": latency_ttft_ms,
    }


def _provider_custody(raw: dict | None) -> tuple[str | None, str | None]:
    """Recover stable provider request/billing ids only when explicitly present."""
    if not isinstance(raw, dict):
        return None, None
    request_id = next(
        (
            str(raw[key])
            for key in ("request_id", "requestId", "response_id", "id")
            if raw.get(key) not in (None, "")
        ),
        None,
    )
    billing_id = next(
        (
            str(raw[key])
            for key in ("billing_id", "billingId", "invoice_id")
            if raw.get(key) not in (None, "")
        ),
        None,
    )
    return request_id, billing_id


def _fire_call_hooks(event: dict) -> None:
    stamped = hooks.stamp_event(event)
    hooks.fire_post_call(stamped)
    hooks.fire_post_process(stamped)


def build_default_providers(config: Config, tracker=None, full: bool = False) -> list[SommProvider]:
    """Build the provider chain from config.

    Default order is each built-in's ``default_order_rank`` in
    ``providers/registry.py``: minimax → openrouter → deepseek → anthropic
    → gemini → openai → perplexity → ollama, keeping only the configured
    ones. Hosted providers lead; local Ollama is the last fallback, not the
    first choice. The seat executors (claude-cli, codex-cli, chatgpt) have
    no rank and are reachable only by pin. Override with the
    SOMM_PROVIDER_ORDER env var (comma-separated, exclusive, e.g.
    "openrouter,minimax,ollama").

    Every commercial-API provider is opt-in via its env var; with none set
    the chain is just Ollama, so the library still works offline. Shared by
    SommLLM and the MCP server so both expose the identical chain.

    ``full=True`` returns EVERY available provider, ignoring both
    SOMM_PROVIDER_ORDER and the pinned-only exclusions (CLI executors).
    Routing preference is about which provider serves production traffic;
    explicit-selection surfaces — shadow gold models, somm_compare,
    somm_replay — must reach anything that's configured.

    Third-party entry-point providers are appended after built-ins in the
    default routing chain when they declare ``default_order_rank``; ranked
    plugins are ordered by ``(rank, name)`` for deterministic routing.
    """
    logger = logging.getLogger("somm.providers")
    specs: list[ProviderSpec] = list(BUILTIN_PROVIDER_SPECS)
    builtin_names = {spec.name for spec in specs}
    seen_names = set(builtin_names)
    plugin_specs: list[ProviderSpec] = []
    for spec in load_entrypoint_provider_specs():
        if spec.name in builtin_names:
            logger.warning("skipping provider spec %r: provider name is built in", spec.name)
            continue
        if spec.name in seen_names:
            logger.warning(
                "skipping provider spec %r: provider name is already registered", spec.name
            )
            continue
        seen_names.add(spec.name)
        plugin_specs.append(spec)
    specs.extend(plugin_specs)

    available: dict[str, SommProvider] = {}
    spec_by_name: dict[str, ProviderSpec] = {}
    for spec in specs:
        try:
            provider = spec.factory(config, tracker)
        except Exception as exc:  # noqa: BLE001 - provider plugins must not break startup
            logger.warning("skipping provider %r: factory failed: %s", spec.name, exc)
            continue
        if provider is None:
            continue
        if not isinstance(provider, SommProvider):
            logger.warning(
                "skipping provider %r: factory returned non-SommProvider %s",
                spec.name,
                type(provider).__name__,
            )
            continue
        # The provider's own .name must match the spec name the registry
        # keyed/collision-checked on — otherwise a plugin spec "acme" could
        # return a provider named "ollama" and corrupt health tracking,
        # telemetry attribution, and generate(provider="acme") lookup.
        actual_name = getattr(provider, "name", None)
        if actual_name != spec.name:
            logger.warning(
                "skipping provider %r: factory returned a provider named %r",
                spec.name,
                actual_name,
            )
            continue
        available[spec.name] = provider
        spec_by_name[spec.name] = spec

    if full:
        return list(available.values())

    if config.provider_order:
        # Exclusive: ONLY providers in the list, in the listed order.
        # If you set SOMM_PROVIDER_ORDER=openrouter,minimax,ollama
        # then anthropic is NOT in the chain, even if its key is set.
        chain = [available[p] for p in config.provider_order if p in available]
        return chain if chain else list(available.values())

    # Default order: ascending default_order_rank (see the docstring).
    builtin_order = [
        spec.name
        for spec in sorted(
            BUILTIN_PROVIDER_SPECS,
            key=lambda spec: (
                spec.default_order_rank if spec.default_order_rank is not None else 10**9
            ),
        )
        if spec.default_order_rank is not None
    ]
    plugin_order = [
        spec.name
        for spec in sorted(
            plugin_specs,
            key=lambda spec: (
                spec.default_order_rank if spec.default_order_rank is not None else 10**9,
                spec.name,
            ),
        )
        if spec.default_order_rank is not None
    ]
    default_order = builtin_order + plugin_order
    return [available[p] for p in default_order if p in available and p in spec_by_name]



def _failure_outcome(exc: BaseException) -> Outcome:
    """The outcome a call that ended in ``exc`` records.

    Timeouts and rate limits keep their own values so rollups can tell a
    slow or throttled provider from a broken one; the rest is an upstream
    error, or exhaustion when the chain ran out.
    """
    if isinstance(exc, SommProvidersExhausted) or type(exc).__name__ == "SommProvidersExhausted":
        return Outcome.EXHAUSTED
    if isinstance(exc, (SommTimeout, TimeoutError)):
        return Outcome.TIMEOUT
    if isinstance(exc, SommRateLimited):
        return Outcome.RATE_LIMIT
    return Outcome.UPSTREAM_ERROR


# Skip reasons for providers that were never candidates; they do not count
# toward fallback depth.
_NOT_CANDIDATE = frozenset({"boundary", "capability", "plan_blocked"})
_ROUTE_SKIPPED_MAX = 16


def _route_record(
    mode: str,
    first: list | None,
    served: list | None,
    skipped: list,
    max_tokens: int | None,
) -> dict:
    """The ``route`` object every call row carries in ``routing_json``.

    ``first`` and ``served`` are ``[provider, model]`` (``served`` is None
    when no route answered); ``skipped`` is ``[provider, model, reason]``
    per candidate passed over; ``depth`` counts the real candidates among
    them, so it is 0 when the first choice served; ``max_tokens`` is the cap
    sent to the provider that answered, or the requested one.
    """
    return {
        "mode": mode,
        "first": first,
        "served": served,
        "depth": sum(1 for s in skipped if s[2] not in _NOT_CANDIDATE),
        "skipped": [list(s) for s in skipped[:_ROUTE_SKIPPED_MAX]],
        "max_tokens": max_tokens,
    }


def _route_source(*, policy: bool, learned_override: bool, pinned: bool) -> str:
    """Why the route was chosen, for ``routing_json["route_source"]``.

    First match wins: a workload policy (``somm/auto`` included), a learned
    override that changed the request, a caller pin (provider or model),
    else the default chain.
    """
    if policy:
        return "policy"
    if learned_override:
        return "learned_override"
    if pinned:
        return "pin"
    return "default"


def _route_summary(skipped: list) -> str:
    """``provider=reason`` per skipped candidate, for an error_detail."""
    return ", ".join(f"{s[0]}={s[2]}" for s in skipped[:_ROUTE_SKIPPED_MAX])


def _served(resp, provider_name: str, max_tokens: int | None = None) -> tuple:
    """Read one provider response into the fields every call path records.

    Returns ``(text, provider, model, tokens_in, tokens_out, latency_ms,
    tool_calls, stop_reason, reasoning_content, raw, cost_usd, cost_source,
    outcome, error_kind, error_detail)``. A reported ``cost_usd`` wins over a
    computed estimate and names its source; blank text with no tool calls is
    ``Outcome.EMPTY`` with an operator-friendly detail.
    """
    text = resp.text
    tool_calls = _to_core_tool_calls(resp.tool_calls)
    outcome = Outcome.OK
    error_kind = error_detail = None
    if not text.strip() and not tool_calls:
        outcome = Outcome.EMPTY
        error_kind = "EmptyResponse"
        error_detail = _format_empty_detail(
            provider=provider_name,
            model=resp.model,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
            reasoning_tokens=(
                (resp.reasoning_tokens or extract_reasoning_tokens(resp.raw))
                if reasoning_ate_budget(resp)
                else None
            ),
            max_tokens=getattr(resp, "max_tokens_sent", None) or max_tokens,
        )
    cost_source = f"provider:{provider_name}" if resp.cost_usd is not None else None
    return (
        text,
        provider_name,
        resp.model,
        resp.tokens_in,
        resp.tokens_out,
        resp.latency_ms,
        tool_calls,
        resp.stop_reason,
        resp.reasoning_content,
        resp.raw,
        resp.cost_usd,
        cost_source,
        outcome,
        error_kind,
        error_detail,
    )

@dataclass
class _Plan:
    """A call that was admitted: the request as it will be sent, where it
    goes, and what its row needs to say. Built by ``SommLLM._plan_call``."""

    wl: Any
    workload: str
    req: SommRequest
    prompt_text: Any
    messages: list[dict] | None
    system: str
    model: str | None
    provider: str | None
    max_tokens: int
    temperature: float
    call_prompt_id: str | None
    prompt_hash: str
    short_circuit: Any
    short_circuited: str | None
    route_source: str
    route_mode: str
    routing_metadata: dict
    allow_fallback: bool
    wait: Any
    wait_on_exhausted: float | None
    session_id: str | None
    parent_call_id: str | None
    correlation_id: str | None
    observation_role: str
    source_call_id: str | None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class _Attempted:
    """What ``SommLLM._dispatch`` got: the response (or, for a stream, the
    opened stream) and who served it, or the failure; the route either way."""

    resp: Any = None
    provider: str = ""
    model: str = ""
    failure: BaseException | None = None
    error_detail: str | None = None
    fallback_info: dict | None = None
    first: list | None = None
    skipped: list = field(default_factory=list)
    raise_after: Exception | None = None


def _generate_attempt(provider, request):
    return provider.generate(request)


class _StreamStart:
    """A provider stream opened and read up to its first output.

    The dispatcher's attempt for a stream: :class:`_StreamOpener` opens one
    and reads it until there is something to give the caller (visible text,
    a tool-call fragment, or — for an ``events`` stream — reasoning) or the
    stream ends. Until then nothing has reached the caller, so a failure is
    an ordinary failed attempt the router may route past. After that the
    caller drains :meth:`events`, and a failure can only be raised.

    The router reads ``text``, ``tool_calls`` and ``started`` to tell an
    empty answer from a real one, and ``model`` / ``tokens_out`` /
    ``reasoning_tokens`` / ``raw`` to say why an empty one was empty.
    """

    def __init__(self, provider, request: SommRequest, *, events: bool) -> None:
        self.provider = provider
        self.request = request
        self.model = request.model or getattr(provider, "default_model", "") or ""
        self._events = events
        self._chunks = iter(provider.stream(request))
        self._stripper = ThinkStreamStripper()
        self._pending: list = []
        self.collected: list[str] = []
        self.tool_calls: list[ToolCallDelta] = []
        self.final: SommResponse | None = None
        self.done = False
        self.usage = "estimate"

    # What the router reads.
    @property
    def text(self) -> str:
        return "".join(self.collected)

    @property
    def started(self) -> bool:
        return not self.done and any(self._shown(e) for e in self._pending)

    @property
    def tokens_out(self) -> int:
        return self.final.tokens_out if self.final else 0

    @property
    def reasoning_tokens(self) -> int | None:
        return self.final.reasoning_tokens if self.final else None

    @property
    def raw(self) -> dict | None:
        return self.final.raw if self.final else None

    def prime(self) -> _StreamStart:
        while not self.done and not any(self._shown(e) for e in self._pending):
            self._read()
        return self

    def _shown(self, event) -> bool:
        return self._events or not isinstance(event, ReasoningDelta)

    def events(self) -> Iterator:
        while True:
            while self._pending:
                yield self._pending.pop(0)
            if self.done:
                return
            self._read()

    def _read(self) -> None:
        chunk = next(self._chunks, None)
        if chunk is None:
            # Ended without a final chunk. Built-in adapters whose wire format
            # has a completion marker raise when it is missing; a plugin
            # provider may simply stop, so that still counts as the end.
            self._finish(None)
            return
        if chunk.model:
            self.model = chunk.model
        if getattr(chunk, "reasoning", ""):
            self._pending.append(ReasoningDelta(chunk.reasoning))
        if chunk.text:
            self._text(self._stripper.feed(chunk.text))
        for delta in getattr(chunk, "tool_calls", None) or ():
            self.tool_calls.append(delta)
            self._pending.append(delta)
        if chunk.done:
            self._finish(getattr(chunk, "response", None))

    def _text(self, visible: str) -> None:
        if visible:
            self.collected.append(visible)
            self._pending.append(TextDelta(visible))

    def _finish(self, response: SommResponse | None) -> None:
        self._text(self._stripper.flush())
        self.final = response
        self.done = True
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()

    def response(self, plan: _Plan, t0: float) -> SommResponse:
        """The answer as generate would have returned it, from the provider's
        final chunk when it sent one, with the text the caller was given."""
        text = self.text
        latency_ms = int((time.monotonic() - t0) * 1000)
        final = self.final
        if final is not None and (final.tokens_in or final.tokens_out):
            self.usage = "provider"
            return replace(
                final, text=text, latency_ms=latency_ms, model=final.model or self.model
            )
        p, model = self.provider, self.model
        if plan.req.messages is not None:
            tokens_in = sum(
                p.estimate_tokens(m.get("content") or "", model) for m in plan.req.messages
            )
        else:
            tokens_in = p.estimate_tokens(plan.req.prompt, model)
        tokens_in += p.estimate_tokens(plan.req.system, model) if plan.req.system else 0
        return SommResponse(
            text=text,
            model=(final.model if final else "") or model,
            tokens_in=tokens_in,
            tokens_out=p.estimate_tokens(text, model) if text else 0,
            latency_ms=latency_ms,
            raw=final.raw if final else None,
            tool_calls=final.tool_calls if final else [],
            stop_reason=final.stop_reason if final else "",
            reasoning_content=final.reasoning_content if final else "",
            cost_usd=final.cost_usd if final else None,
            max_tokens_sent=final.max_tokens_sent if final else None,
        )


class _StreamOpener:
    """The attempt a stream hands :meth:`SommLLM._dispatch`. Remembers the
    last stream that ended empty, which is a stream's answer when the chain
    has nothing better."""

    def __init__(self, events: bool) -> None:
        self.events = events
        self.empty: _StreamStart | None = None

    def __call__(self, provider, request: SommRequest) -> _StreamStart:
        start = _StreamStart(provider, request, events=self.events).prime()
        if start.done and not start.text.strip() and not start.tool_calls:
            self.empty = start
        return start


def _stream_extra(start: _StreamStart | None) -> dict:
    """What a stream adds to ``routing_json``: where its token counts came from."""
    return {"stream": {"usage": start.usage if start is not None else "provider"}}


def _format_error_detail(exc: Exception, provider: str, model: str | None) -> str:
    """Build a bounded, operator-friendly description of an LLM call failure.

    Captures the exception's text plus, if present, the HTTP response body
    attached by httpx.HTTPStatusError. Truncated to 512 chars so telemetry
    rows stay bounded.
    """
    parts: list[str] = [f"{type(exc).__name__}: {exc}"]
    # httpx.HTTPStatusError carries the response as .response — its .text is
    # the server's error body (e.g. "model 'qwen3:14b' not found").
    response = getattr(exc, "response", None)
    if response is not None:
        body = getattr(response, "text", "") or ""
        status = getattr(response, "status_code", None)
        if status is not None:
            parts.append(f"http_status={status}")
        if body:
            parts.append(f"body={_scrub_secrets(body.strip())[:200]}")
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    return _scrub_secrets(" | ".join(parts))[:512]


def _pinned_error_detail(exc: Exception, provider: str, model: str | None) -> str:
    """`_format_error_detail` plus the one-line escape hatch.

    A sticky pin that fails is a dead end by design, so the recorded detail
    has to say how to opt into the chain — otherwise `somm tail` shows an
    error with no next step. It names the pin, because a caller that did not
    know it was passing ``provider=`` (a wrapper forwarding its own default)
    could not tell why the chain was skipped (issue #71).
    """
    pin = f"provider={provider!r}" + (f", model={model!r}" if model else "")
    hint = f" | pinned call ({pin} passed by the caller): no fallback (allow_fallback=True to route past it)"
    detail = _format_error_detail(exc, provider, model)
    return detail[: 512 - len(hint)] + hint


def _routed_past_info(router_result, model: str | None) -> dict | None:
    """on_fallback payload for an unpinned call the chain served after
    routing past providers that failed authentication; None when it didn't."""
    skipped = getattr(router_result, "skipped", None) or []
    if not skipped:
        return None
    first_provider, first_exc = skipped[0]
    return {
        "pinned_provider": None,
        "pinned_model": model,
        "error_kind": type(first_exc).__name__,
        "error_detail": _format_error_detail(first_exc, first_provider, model),
        "routed_past": [name for name, _ in skipped],
    }


def _scrub_secrets(text: str) -> str:
    """Redact common credentials from persisted operator-facing error text."""
    return scrub_text(text)


def _citations_json(citations: list | None) -> str | None:
    """Serialize citation metadata for calls.citations_json. Never raises."""
    if citations is None:
        return None
    try:
        return json.dumps(citations)
    except Exception:
        return None


def _format_empty_detail(
    *,
    provider: str,
    model: str,
    tokens_out: int,
    latency_ms: int,
    reasoning_tokens: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Build a bounded diagnostic for the EMPTY outcome.

    Two empirically-confirmed modes (see commit 54aa18f):
      - ``no_content``    — upstream returned tokens_out=0; the model never
                            actually generated (e.g. openrouter elephant-alpha
                            with ``{"content": null}``, sub-500ms latency).
      - ``stripped_empty`` — tokens_out>0 but the visible text is empty after
                             post-processing (e.g. minimax M2.7 generating
                             only inside ``<think>`` blocks; full latency).

    ``reasoning_tokens`` is passed only when the provider reported that
    reasoning took (nearly) the whole output; the detail then also carries
    ``reasoning_exhausted`` and says to raise max_tokens. It stays a
    ``stripped_empty`` row, which the self-heal detector keys on.

    Both fields are kept on the row so SQL filters (``WHERE outcome='empty'
    AND error_detail LIKE '%no_content%'``) can split them without joining.
    """
    hint = "no_content" if tokens_out == 0 else "stripped_empty"
    parts = [f"EmptyResponse: {hint}"]
    if reasoning_tokens:
        sent = f" (max_tokens sent: {max_tokens})" if max_tokens else ""
        parts.append(
            f"reasoning_exhausted: {reasoning_tokens} of {tokens_out} output tokens "
            f"were reasoning{sent}; raise max_tokens or lower reasoning_effort"
        )
    parts += [
        f"out_tokens={tokens_out}",
        f"latency_ms={latency_ms}",
    ]
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    return " | ".join(parts)[:512]


def enforce_workload_budget(
    repo,
    workload,
    *,
    fail_closed: bool,
    default_cap_usd_daily: float | None,
) -> None:
    """Fail-closed daily-budget gate. Raises ``SommBudgetExceeded`` BEFORE
    any provider call when ``fail_closed`` is set and the workload's
    accumulated spend today has reached its ``budget_cap_usd_daily``.

    No-op unless both the flag and a per-workload (or default) cap are set, so
    it is inert by default and for capless workloads. Enforces on COMMITTED
    daily spend — the async telemetry writer drains during the seconds-long
    real LLM call, so committed spend tracks actual spend closely. The call
    that crosses the cap still completes and the next is blocked, so overshoot
    is bounded to ~one call plus any spend not yet flushed. Unlike the
    post-call soft-warn it does not dedup — every call past the ceiling is
    refused, not warned once.

    Shared by ``SommLLM`` and the somm-service /v1/messages proxy so the
    library path and the harness-routed path enforce one and the same ceiling
    (one ledger, one policy).
    """
    if not fail_closed:
        return
    cap = workload.budget_cap_usd_daily
    if cap is None:
        cap = default_cap_usd_daily
    if cap is None:
        return
    with repo._open() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM calls "
            "WHERE workload_id = ? AND date(ts) = date('now') "
            "AND budget_eligible != 0",
            (workload.id,),
        ).fetchone()
    spent = float(row[0]) if row else 0.0
    if spent >= cap:
        from somm.errors import SommBudgetExceeded

        raise SommBudgetExceeded(
            f"SOMM_BUDGET_EXCEEDED\n\n"
            f"Problem: workload {workload.name!r} has spent ${spent:.4f} today, "
            f"at/over its daily cap ${float(cap):.4f}.\n"
            f"Cause: budget_fail_closed is on and the ceiling is reached; the "
            f"call was blocked before dispatch (no spend, no telemetry row).\n"
            f"Fix: raise the cap, wait for the UTC-day reset, or unset "
            f"SOMM_BUDGET_FAIL_CLOSED.",
            workload=workload.name,
            spent_usd=spent,
            cap_usd=float(cap),
        )


def _default_stderr_alerter(event: dict) -> None:
    """Default on_error handler — one-line warning to stderr.

    Replace via SommLLM(on_error=...) to forward to logging, Slack, etc.
    Keep it cheap: this runs inline on every non-OK call.
    """
    print(
        f"[somm] {event.get('outcome', '?').upper()} "
        f"workload={event.get('workload')} "
        f"provider={event.get('provider')} "
        f"model={event.get('model')} "
        f"kind={event.get('error_kind')} "
        f"detail={event.get('error_detail')}",
        file=sys.stderr,
    )


def _default_stderr_fallback_notifier(event: dict) -> None:
    """Default on_fallback handler — self-healing notice to stderr.

    Fires when a pinned (provider, model) call failed and the chain
    recovered via fallback, or when an unpinned call was served only after
    the chain routed past a provider that failed authentication. The call
    succeeded; this is *not* an error, just a signal that a degradation
    happened. Suppress via `SommLLM(on_fallback=lambda _: None)` or reroute
    to logging.
    """
    if event.get("pinned_provider") is None:
        origin = "routed_past=" + ",".join(event.get("routed_past") or [])
    else:
        origin = f"pinned={event.get('pinned_provider')}/{event.get('pinned_model')}"
    print(
        f"[somm] FALLBACK workload={event.get('workload')} "
        f"{origin} "
        f"→ actual={event.get('actual_provider')}/{event.get('actual_model')} "
        f"reason={event.get('error_kind')} "
        f"detail={event.get('error_detail')}",
        file=sys.stderr,
    )


class SommLLM:
    """The library handle. One per process per project.

    Default mode is 'observe' (DX default — TTHW first). Strict mode enforces
    workload/prompt registration; enable via `SommLLM(mode="strict")` or env
    `SOMM_MODE=strict`.
    """

    def __init__(
        self,
        project: str | None = None,
        mode: str | None = None,
        providers: list[SommProvider] | None = None,
        config: Config | None = None,
        on_error: Callable[[dict], None] | None = None,
        on_fallback: Callable[[dict], None] | None = None,
    ) -> None:
        hooks.load_entry_points()
        self.config = config or load_config(project=project)
        if mode is not None:
            self.config.mode = mode
        if config is not None and Path(self.config.spool_dir) == Path("./.somm/spool"):
            self.config.spool_dir = Path(self.config.db_dir) / "spool"

        self.repo = Repository(self.config.db_path)
        self._shadow_cfg_cache: dict[str, tuple[float, dict | None]] = {}
        self._prompt_ids_cache: dict[str, tuple[float, set[str]]] = {}
        self._tracker = ProviderHealthTracker(self.repo)
        self.providers: list[SommProvider] = providers or self._default_providers()
        # Out-of-chain executors, built on first pin. See
        # _pinned_only_providers for why they are not in the chain.
        self._pinned_only_cache: dict[str, SommProvider] | None = None
        self._plan_governor = _build_plan_governor(self.config)
        self.router = Router(self.providers, self._tracker, plan_governor=self._plan_governor)
        # Alerting hook — fires on every non-OK outcome with a small context
        # dict. Default writes a one-line warning to stderr so failures are
        # visible in the caller's terminal. Pass on_error=lambda _: None to
        # suppress, or your own callable to forward to logging / Slack / etc.
        self._on_error: Callable[[dict], None] | None = (
            on_error if on_error is not None else _default_stderr_alerter
        )
        # Self-healing notice — fires when a pinned (provider, model) call
        # failed and the chain recovered. The call itself succeeded, so this
        # is intentionally *not* an error; it's observability for silent
        # degradation (pinned provider down, key rotated, rate-limit wave).
        self._on_fallback: Callable[[dict], None] | None = (
            on_fallback if on_fallback is not None else _default_stderr_fallback_notifier
        )

        mirror_repo: Repository | None = None
        if self.config.cross_project_enabled:
            try:
                mirror_repo = Repository(self.config.global_db_path)
                # Mirror workload registrations as well so global rollups can
                # resolve names rather than showing "(unregistered)".
                _mirror_workloads(self.repo, mirror_repo)
            except Exception as exc:  # noqa: BLE001 — mirror must not break primary telemetry
                _warn_mirror_unavailable(self.config.global_db_path, exc)
                mirror_repo = None

        self._writer = WriterQueue(self.repo, self.config.spool_dir, mirror_repo=mirror_repo)
        self._writer.start()
        self._mirror_repo = mirror_repo

        # Seed pricing on first use so cost tracking works out of the box.
        seed_known_pricing(self.repo)
        sync_bundled_pricing(self.repo)

        # Intelligence-loop activation: run the service scheduler in-process
        # when asked; otherwise warn (once) if grading is configured but no
        # worker has ever run.
        self._scheduler = None
        if self.config.inprocess_workers:
            self._scheduler = _maybe_start_inprocess_workers(self.config, self.repo)
        else:
            _warn_if_intelligence_loop_dormant(self.config, self.repo)

        # Fleet registry + metered-plan pacing: quotas are shared across
        # every project on this machine, so announce this DB and warn (once
        # per process) when a metered plan is burning faster than its window.
        _register_project_once(self.config.project, self.config.db_path)
        _warn_if_plans_off_pace(self._plan_governor)

    def register_workload(self, **kwargs):
        """Register a workload in the project repo AND mirror-if-enabled."""
        wl = self.repo.register_workload(project=self.config.project, **kwargs)
        if self._mirror_repo is not None:
            try:
                self._mirror_repo.register_workload(project=self.config.project, **kwargs)
            except Exception:  # noqa: BLE001
                pass
        return wl

    def set_data_boundary(self, workload: str, data_boundary: str | None) -> None:
        """Set (or clear with None) a workload's stored data boundary."""
        wl = self._require_workload(workload)
        self.repo.set_workload_data_boundary(wl.id, data_boundary)
        if self._mirror_repo is not None:
            try:
                mirror_wl = self._mirror_repo.workload_by_name(workload, self.config.project)
                if mirror_wl is not None:
                    self._mirror_repo.set_workload_data_boundary(mirror_wl.id, data_boundary)
            except Exception:  # noqa: BLE001
                pass

    def set_workload_policy(self, workload: str, policy: dict | None) -> None:
        """Attach or clear a workload routing policy."""
        wl = self._require_workload(workload)
        self.repo.set_workload_policy(wl.id, policy)
        if self._mirror_repo is not None:
            try:
                mirror_wl = self._mirror_repo.workload_by_name(workload, self.config.project)
                if mirror_wl is not None:
                    self._mirror_repo.set_workload_policy(mirror_wl.id, policy)
            except Exception:  # noqa: BLE001
                pass

    def _default_providers(self) -> list[SommProvider]:
        return build_default_providers(self.config, tracker=self._tracker)

    def _policy_router(
        self,
        policy: dict,
        *,
        explicit_model: bool,
    ) -> Router:
        providers = _policy_providers(
            self.providers,
            policy,
            explicit_model=explicit_model,
        )
        return Router(
            providers,
            self._tracker,
            circuit_break_after=self.router.circuit_break_after,
            circuit_break_cooldown_s=self.router.circuit_break_cooldown_s,
            exhausted_sleep_cap_s=self.router.exhausted_sleep_cap_s,
            plan_governor=self._plan_governor,
        )

    # ------------------------------------------------------------------

    # Bodies bigger than this (e.g. multimodal prompts with inline base64
    # images) are not captured — a truncated prompt can't be re-run
    # faithfully against the gold model, so skipping beats storing junk.
    _SAMPLE_BODY_CAP = 200_000  # chars

    def _shadow_config_cached(self, workload_id: str) -> dict | None:
        """Shadow config with a 5-minute in-process cache — the capture
        hook runs on every successful call and must not add a query."""
        now = time.monotonic()
        cached = self._shadow_cfg_cache.get(workload_id)
        if cached and cached[0] > now:
            return cached[1]
        try:
            cfg = self.repo.get_shadow_config(workload_id)
        except Exception:
            cfg = None
        self._shadow_cfg_cache[workload_id] = (now + 300.0, cfg)
        return cfg

    def _known_prompt_ids_cached(self, workload_id: str) -> set[str]:
        """Registered prompt ids with a 5-minute in-process cache.

        Plain-string prompt binding runs on the call path; cache misses cost
        one indexed workload lookup per TTL window and failures degrade empty.
        """
        now = time.monotonic()
        cached = self._prompt_ids_cache.get(workload_id)
        if cached and cached[0] > now:
            return cached[1]
        try:
            ids = prompt_ids_for_workload(self.repo, workload_id)
        except Exception:
            ids = set()
        self._prompt_ids_cache[workload_id] = (now + 300.0, ids)
        return ids

    def _resolve_prompt_id(self, workload_id: str | None, prompt) -> str | None:
        """Best-effort call-row prompt binding. Never raises."""
        try:
            if isinstance(prompt, Prompt):
                return prompt.id
            if workload_id is None or not isinstance(prompt, str):
                return None
            pid = _prompt_id(prompt)
            return pid if pid in self._known_prompt_ids_cached(workload_id) else None
        except Exception:
            return None

    def _maybe_capture_sample(self, wl, call_id: str, prompt, messages, text: str, outcome) -> None:
        """Capture prompt/response bodies for online-eval grading.

        A workload's shadow config is the documented opt-in for body
        storage; private workloads are never captured, whatever their
        config says. Rate-gated deterministically by call_id (the shadow
        worker grades exactly what was captured, so sample_rate applies
        here, once). Never raises — capture must not break the call path.
        """
        import json as _json

        try:
            if outcome != Outcome.OK or not text:
                return
            privacy = getattr(wl, "privacy_class", None)
            if str(getattr(privacy, "value", privacy or "")).lower() == "private":
                return
            cfg = self._shadow_config_cached(wl.id)
            if not cfg:
                return
            rate = float(cfg.get("sample_rate", 0.02) or 0.0)
            if rate <= 0.0:
                return
            bucket = int(stable_hash(call_id)[:8], 16) / 0xFFFFFFFF
            if bucket >= rate:
                return
            if messages is not None:
                body = _json.dumps(messages)
            elif isinstance(prompt, str):
                body = prompt
            else:
                body = _json.dumps(prompt)
            if len(body) > self._SAMPLE_BODY_CAP or len(text) > self._SAMPLE_BODY_CAP:
                return
            self.repo.write_sample(call_id, body, text)
        except Exception:
            pass

    def _enforce_budget(self, wl) -> None:
        """Fail-closed daily-budget gate — thin wrapper over the shared helper.

        See :func:`enforce_workload_budget` for the full contract. Lives as an
        instance method so the call site reads naturally; both the library and
        the somm-service /v1/messages proxy share one gate implementation
        (one-ledger, one-policy).
        """
        enforce_workload_budget(
            self.repo,
            wl,
            fail_closed=self.config.budget_fail_closed,
            default_cap_usd_daily=self.config.budget_default_cap_usd_daily,
        )

    # ------------------------------------------------------------------
    # Data boundary

    def _resolve_boundary(self, wl, per_call: str | None):
        """The boundary this call runs under (see somm_core.boundary)."""
        from somm.boundary import resolve_boundary

        return resolve_boundary(wl, per_call=per_call, project=self.config.project)

    def _served_tier(self, provider_name: str, model: str | None) -> str | None:
        """calls.data_tier for the route that served (or was attempted)."""
        from somm.boundary import served_tier

        if not provider_name:
            return None
        obj = None
        for p in self.providers:
            if p.name == provider_name:
                obj = p
                break
        if obj is None and self._pinned_only_cache:
            obj = self._pinned_only_cache.get(provider_name)
        try:
            return served_tier(obj, provider_name, model)
        except Exception:  # noqa: BLE001 — recording must not break the call
            return None

    def _admit_or_refuse(
        self,
        wl,
        workload: str,
        boundary,
        *,
        pinned: str | None,
        model: str | None,
        policy: dict | None,
        prompt_hash: str,
        observation_role: str = "production",
        source_call_id: str | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        correlation_id: str | None = None,
        max_tokens: int | None = None,
        route_source: str = "default",
    ) -> None:
        """Raise SommPrivacyViolation (after recording a refusal row) when no
        route within ``boundary`` remains for this call. A pin is checked
        as-is: it does not override a boundary."""
        from somm.boundary import admissible_providers, refuse

        if pinned:
            candidates = [self._pick_provider(pinned)]
            context = f"pinned provider {pinned!r}"
        else:
            candidates = (
                _policy_providers(self.providers, policy, explicit_model=model is not None)
                if policy
                else list(self.providers)
            )
            context = "the routing chain"
        ok, refused = admissible_providers(candidates, model, boundary.tier)
        if ok:
            return
        exc = refuse(workload=workload, boundary=boundary, refused=refused, context=context)
        self._record_refusal(
            wl,
            workload,
            exc,
            model=model,
            prompt_hash=prompt_hash,
            observation_role=observation_role,
            source_call_id=source_call_id,
            session_id=session_id,
            parent_call_id=parent_call_id,
            correlation_id=correlation_id,
            mode="pinned" if pinned else "chain",
            max_tokens=max_tokens,
            route_source=route_source,
        )
        raise exc

    def _record_refusal(
        self,
        wl,
        workload: str,
        exc,
        *,
        model: str | None,
        prompt_hash: str,
        observation_role: str = "production",
        source_call_id: str | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        correlation_id: str | None = None,
        mode: str = "chain",
        max_tokens: int | None = None,
        route_source: str = "default",
    ) -> None:
        """One calls row for demand the boundary turned away.

        outcome is ``unknown`` (an existing value: consumers that switch on
        outcome must not meet a new one), error_kind ``SommPrivacyViolation``,
        and error_detail starts ``refused: data_boundary``. No provider was
        contacted, so provider is empty, cost is zero, data_tier is NULL, and
        the row is not budget-eligible. ``routing_json`` lists every refused
        route with reason ``boundary``.
        """
        refused = ", ".join(f"{r['provider']}/{r['model'] or '-'}={r['tier']}" for r in exc.refused)
        skipped = [[r["provider"], r["model"] or "", "boundary"] for r in exc.refused]
        route = _route_record(mode, skipped[0][:2] if skipped else None, None, skipped, max_tokens)
        detail = (
            f"refused: data_boundary={exc.data_boundary} (from {exc.boundary_source}) "
            f"| refused routes: {refused or 'none'}"
        )[:512]
        try:
            call = Call(
                id=str(uuid.uuid4()),
                ts=datetime.now(UTC),
                project=self.config.project,
                workload_id=wl.id,
                prompt_id=None,
                provider="",
                model=model or "",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                cost_usd=0.0,
                outcome=Outcome.UNKNOWN,
                error_kind="SommPrivacyViolation",
                error_detail=detail,
                max_tokens=max_tokens,
                routing_json=json.dumps({"route_source": route_source, "route": route}),
                prompt_hash=prompt_hash,
                response_hash=stable_hash(""),
                correlation_id=(
                    correlation_id if correlation_id is not None else hooks.current_correlation_id()
                ),
                session_id=session_id,
                parent_call_id=parent_call_id,
                cost_basis="computed",
                cost_kind="unknown",
                cost_accuracy="actual",
                cost_source=None,
                observation_role=observation_role or "production",
                source_call_id=source_call_id,
                budget_eligible=False,
                call_site=hooks.current_call_site(),
                client_version=_CLIENT_VERSION,
            )
            self._finish_call(call, workload=workload, short_circuited=None)
        except Exception:  # noqa: BLE001 — the refusal itself must still raise
            logging.getLogger("somm.client").warning(
                "could not record boundary refusal for workload %r", workload, exc_info=True
            )

    def _resolve_wait_on_exhausted(self, wait) -> float | None:
        if wait is _WAIT_UNSET:
            wait = self.config.wait_on_exhausted
        if wait is None:
            return None
        try:
            wait_s = float(wait)
        except (TypeError, ValueError):
            return None
        return wait_s if wait_s > 0 else None

    def _resolve_allow_fallback(
        self,
        allow_fallback: bool | None,
        no_fallback: bool,
    ) -> bool:
        """Decide whether a failed *pinned* call may be rescued by the chain.

        A pin is sticky by default: `provider=`/`model=` means "serve this",
        not "try this first". Silent substitution made results unexplainable
        for callers who had already chosen the model. Precedence:

          1. `no_fallback=True` (deprecated) — always sticky. It can only ever
             force stickiness; passing False never re-enables rescue, because
             the pre-0.16 call sites that pass `no_fallback=bool(...)` meant
             "don't rescue", never "please substitute".
          2. explicit `allow_fallback=` on the call.
          3. `SOMM_PINNED_FALLBACK` / `Config.pinned_fallback` (default False).
        """
        if no_fallback:
            return False
        if allow_fallback is not None:
            return bool(allow_fallback)
        return bool(getattr(self.config, "pinned_fallback", False))

    def _finish_call(
        self,
        call: Call,
        *,
        workload: str,
        short_circuited: str | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Persist one call and tell everyone who listens.

        Writes the row, fires the post_call/post_process hooks with the
        observer event, and calls the ``on_error`` alerter when the outcome is
        not OK. The three call paths used to carry their own copies of this
        and drifted: stream had no alerter and embed named the wrong provider.
        """
        self._writer.submit(call)
        _fire_call_hooks(
            _call_event(
                call_id=call.id,
                correlation_id=call.correlation_id,
                project=call.project,
                workload=workload,
                provider=call.provider,
                model=call.model,
                outcome=call.outcome.value,
                tokens_in=call.tokens_in,
                tokens_out=call.tokens_out,
                latency_ms=call.latency_ms,
                cost_usd=call.cost_usd,
                temperature=temperature,
                max_tokens=max_tokens,
                error_kind=call.error_kind,
                short_circuited=short_circuited,
                cost_basis=call.cost_basis,
                cost_kind=call.cost_kind,
                cost_accuracy=call.cost_accuracy,
                cost_source=call.cost_source,
                pricing_version=call.pricing_version,
                observation_role=call.observation_role,
                source_call_id=call.source_call_id,
                eval_result_id=call.eval_result_id,
                provider_request_id=call.provider_request_id,
                billing_id=call.billing_id,
                origin=call.origin,
                budget_eligible=call.budget_eligible,
                cache_tokens_in=call.cache_tokens_in,
                cache_tokens_out=call.cache_tokens_out,
                reasoning_tokens=call.reasoning_tokens,
                latency_ttft_ms=call.ttft_ms,
            )
        )
        # A caller that stops reading a stream early chose to; that is
        # recorded, not alerted.
        if (
            call.outcome != Outcome.OK
            and call.error_kind != "StreamAbandoned"
            and self._on_error is not None
        ):
            with contextlib.suppress(Exception):  # the alerter must not break the caller
                self._on_error(
                    {
                        "call_id": call.id,
                        "workload": workload,
                        "provider": call.provider,
                        "model": call.model,
                        "outcome": call.outcome.value,
                        "error_kind": call.error_kind,
                        "error_detail": call.error_detail,
                    }
                )

    def _mark_persister(self, call: Call) -> Callable[[SommResult], None]:
        """What ``SommResult.mark()`` runs for a call this client recorded.

        The row may still be in the writer queue, so the queued ``Call`` is
        corrected first (an insert that has not happened yet carries the
        mark), then the queue is flushed and the stored row updated (an
        insert that already happened is corrected). Either way the ledger
        ends up with the mark, and ``call_updates`` keeps a record of it.
        """

        def persist(result: SommResult) -> None:
            call.outcome = result.outcome
            call.error_kind = result.error_kind
            call.error_detail = result.error_detail
            self._writer.flush()
            for repo in (self.repo, self._mirror_repo):
                if repo is None:
                    continue
                try:
                    repo.record_outcome_update(
                        call.id,
                        result.outcome,
                        error_kind=result.error_kind,
                        error_detail=result.error_detail,
                    )
                except Exception:  # noqa: BLE001 — a mark must not break the caller
                    logging.getLogger("somm.client").warning(
                        "could not persist outcome mark for call %s", call.id, exc_info=True
                    )

        return persist

    def generate(
        self,
        prompt: str | list[dict] | Prompt,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
        capabilities_required: list[str] | None = None,
        allow_empty: bool = False,
        allow_fallback: bool | None = None,
        no_fallback: bool = False,
        raise_on_empty: bool = False,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        wait: float | None = _WAIT_UNSET,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        correlation_id: str | None = None,
        observation_role: str = "production",
        source_call_id: str | None = None,
        claude_cli_text_only: bool = False,
        data_boundary: str | None = None,
        timeout: float | None = None,
        raise_on_failure: bool = False,
    ) -> SommResult:
        """Run one LLM call. Writes telemetry synchronously at the row level.

        **Recommended: ``raise_on_failure=True``.** A failed call is recorded
        and then raised as a typed error — ``SommTimeout``,
        ``SommRateLimited``, ``SommProvidersExhausted``, ``SommAuthError``,
        ``SommModelNotFound``, ``SommEmptyResponse`` (unless
        ``allow_empty``), ``SommRefused`` (the model or a content filter
        declined) — carrying ``provider``, ``model`` and ``call_id``. The
        default (False) returns the failed result, as it always has; check
        ``result.ok`` and ``result.error``.

        ``timeout`` (seconds) bounds this call at every provider adapter,
        overriding ``SOMM_HTTP_TIMEOUT`` / the adapter's own default.

        demo mode: auto-registers unknown workloads as 'ad_hoc' equivalents.
        strict mode: raises SommStrictMode if workload isn't registered.
        ``prompt`` may be a string, multimodal prompt list, or registered
        ``Prompt`` object. Passing ``Prompt`` dispatches its body and stamps
        that prompt version on the telemetry row.

        ``claude_cli_text_only=True`` disables Claude CLI tools, MCP configuration,
        and setting sources. Requires provider="claude-cli", allow_fallback=False,
        and no configured CLI extra arguments. Default False preserves CLI behavior.

        **Pins are sticky.** Naming a ``provider`` and/or ``model`` means
        "serve this call with that", not "try it first". If the pinned
        target fails, the call returns ``outcome=UPSTREAM_ERROR`` with the
        pinned (provider, model) preserved on both the result and the
        telemetry row — somm never quietly answers with a model the caller
        did not ask for. ``allow_fallback=True`` opts back into rescue
        through the router chain for callers that would rather have an
        answer from *some* model than none (set ``SOMM_PINNED_FALLBACK=1``
        to make that the process-wide default). Both are inert when no
        ``provider`` is pinned: router-driven calls have no pinned target
        to honor, and an explicit ``model`` is carried unchanged across
        every provider the chain tries.

        ``no_fallback=True`` is the deprecated spelling of
        ``allow_fallback=False`` and is now the default; it is still
        accepted and still forces stickiness.

        Tool-calling (``tools``, ``messages``, ``tool_choice``): see
        docs/tool-calling.md. Tools is a list of somm-neutral JSON-Schema-
        shaped declarations; messages overrides prompt for multi-turn
        history; tool_choice forces auto/any/none/specific. Providers that
        don't yet support tool calls raise SommBadRequest if tools is
        non-empty (the router will fall through to a capable provider).
        A response with non-empty ``tool_calls`` and no text is NOT
        treated as Outcome.EMPTY — it is the expected shape of a
        tool-use turn.

        ``wait`` controls admission when every routed provider candidate is
        cooling down. Omit it to use ``SOMM_WAIT_ON_EXHAUSTED`` when set.
        Pass ``None`` or ``0`` to fail fast; pass positive seconds to wait up
        to that deadline before recording and raising final exhaustion.

        ``correlation_id`` ties telemetry directly to an external task, job,
        or request. It takes precedence over the process-wide correlation hook,
        which remains useful for framework integrations.

        ``observation_role`` says what kind of observation this call is:
        ``"production"`` (default), or ``"shadow_gold"`` / ``"shadow_judge"`` /
        ``"shadow"`` for a call made to grade or shadow another one, with
        ``source_call_id`` naming that call. Non-production rows are kept out
        of serving rollups, budgets, and recommendations, exactly as the
        shadow-eval worker's own rows are.

        ``data_boundary`` (``local`` | ``zdr`` | ``subscribed`` |
        ``unvetted``) may only tighten the workload's boundary for this call;
        a looser value is ignored. Routes outside the boundary are never
        tried, whether routed or pinned. When none remain the call records a
        refusal row and raises ``SommPrivacyViolation`` (never retried, never
        rescued by a looser route). See docs/data-boundary.md.
        """
        plan = self._plan_call(
            prompt, system=system, workload=workload, max_tokens=max_tokens,
            temperature=temperature, model=model, provider=provider,
            capabilities_required=capabilities_required, allow_empty=allow_empty,
            allow_fallback=allow_fallback, no_fallback=no_fallback, tools=tools,
            messages=messages, tool_choice=tool_choice, response_format=response_format,
            reasoning_effort=reasoning_effort, wait=wait, session_id=session_id,
            parent_call_id=parent_call_id, correlation_id=correlation_id,
            observation_role=observation_role, source_call_id=source_call_id,
            claude_cli_text_only=claude_cli_text_only, data_boundary=data_boundary,
            timeout=timeout,
        )
        att = self._dispatch(plan, _generate_attempt)
        result = self._record(plan, att)
        if att.raise_after is not None:
            raise att.raise_after

        # Loud-empty: when the caller explicitly set raise_on_empty=True
        # and we ended up with an empty response (every provider returned
        # blanks), raise rather than handing back empty text. This is the
        # 2026-05-06 lesson: a thinking model that consumed its whole token
        # budget on internal reasoning shouldn't silently return empty —
        # callers who care need a typed exception so they can fall back,
        # bump max_tokens, or pick a non-thinking model.
        if raise_on_empty and result.outcome == Outcome.EMPTY:
            if result.error_detail and "reasoning_exhausted" in result.error_detail:
                cause = (
                    "the model spent its whole output budget on reasoning; raise "
                    "max_tokens or lower reasoning_effort"
                )
            else:
                cause = (
                    "bump max_tokens / pick a non-thinking model if a thinking "
                    "model burned the whole budget"
                )
            raise SommEmptyResponse(
                f"empty response from {result.provider}/{result.model} "
                f"(workload={workload!r}): {cause}. Set raise_on_empty=False to "
                f"swallow. Detail: {result.error_detail}"
            )

        if raise_on_failure:
            failure = _failure_error(result, att.failure, allow_empty=allow_empty)
            if failure is not None:
                raise failure

        return result

    # ------------------------------------------------------------------
    # The one call path: admit and plan, attempt, record. generate() and
    # stream() differ only in the attempt they hand _dispatch and in how the
    # answer reaches the caller.

    def _plan_call(
        self,
        prompt,
        *,
        system: str,
        workload: str,
        max_tokens: int,
        temperature: float,
        model: str | None,
        provider: str | None,
        capabilities_required: list[str] | None,
        allow_empty: bool,
        allow_fallback: bool | None,
        no_fallback: bool,
        tools: list[dict] | None,
        messages: list[dict] | None,
        tool_choice: str | dict | None,
        response_format: dict | None,
        reasoning_effort: str | None,
        wait,
        session_id: str | None,
        parent_call_id: str | None,
        correlation_id: str | None,
        observation_role: str,
        source_call_id: str | None,
        claude_cli_text_only: bool,
        data_boundary: str | None,
        timeout: float | None,
    ) -> _Plan:
        """Admit the call and decide where it goes, before any provider is
        touched: workload, boundary, learned override, pre-call hooks,
        ``somm/auto``, the boundary refusal and the budget gate. Raises when
        the call is refused."""
        wl = self.repo.workload_by_name(workload, self.config.project)
        if wl is None:
            if self.config.mode == "strict":
                workload_arg = shlex.quote(workload)
                raise SommStrictMode(
                    f"SOMM_WORKLOAD_UNREGISTERED\n\n"
                    f"Problem: This call used workload {workload!r}, but it is not registered.\n"
                    f"Cause: strict mode requires workload metadata before calls are logged.\n"
                    f"Fix:\n"
                    f"  somm workload add {workload_arg} --from-example structured-extraction\n"
                    f"  # or switch to observe mode:\n"
                    f"  export SOMM_MODE=observe\n"
                    f"Docs: docs/errors/SOMM_WORKLOAD_UNREGISTERED.md"
                )
            wl = self.register_workload(name=workload)

        boundary = self._resolve_boundary(wl, data_boundary)

        # Self-healing: apply any learned parameter override for this
        # (workload, model). The agent worker writes these when it detects a
        # recurring capability failure (e.g. a reasoning model that exhausts its
        # output budget on thinking tokens and returns empty). Fail-open and
        # one-directional — only ever RAISES max_tokens, never lowers it, and a
        # lookup failure must never break a live call.
        learned_override = False
        if wl is not None and model:
            try:
                _ov = self.repo.lookup_learned_override(wl.id, model, provider)
                _floor = _ov.get("max_tokens_floor") if _ov else None
                if _floor and _floor > max_tokens:
                    learned_override = True
                    logging.getLogger("somm.client").info(
                        "somm self-heal: max_tokens %d→%d for workload=%s model=%s (%s)",
                        max_tokens,
                        _floor,
                        workload,
                        model,
                        _ov.get("failure_signature"),
                    )
                    max_tokens = _floor
            except Exception:
                pass  # never let a learned-override lookup break a live call

        if claude_cli_text_only and (
            provider != "claude-cli"
            or self._resolve_allow_fallback(allow_fallback, no_fallback)
        ):
            raise ValueError("claude_cli_text_only requires provider='claude-cli' and allow_fallback=False")

        wait_on_exhausted = self._resolve_wait_on_exhausted(wait)
        allow_fallback_resolved = self._resolve_allow_fallback(allow_fallback, no_fallback)

        prompt_text = prompt.body if isinstance(prompt, Prompt) else prompt
        call_prompt_id = None if messages is not None else self._resolve_prompt_id(wl.id, prompt)

        def request() -> SommRequest:
            return SommRequest(
                prompt=prompt_text,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                capabilities_required=_merge_caps(
                    wl.capabilities_required,
                    capabilities_required,
                    infer_capabilities(prompt_text),
                ),
                allow_empty=allow_empty,
                tools=tools or [],
                messages=messages,
                tool_choice=tool_choice,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                claude_cli_text_only=claude_cli_text_only,
                timeout=timeout,
            )

        req = request()
        short_circuit: hooks.ShortCircuit | None = None
        short_circuited: str | None = None
        if hooks.has_hooks(hooks.PRE_CALL):
            original_prompt_text = prompt_text
            ctx = hooks.PreCallContext(
                workload=workload,
                prompt=prompt_text,
                system=system,
                messages=messages,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools or [],
                tool_choice=tool_choice,
                project=self.config.project,
                response_format=response_format,
                data_boundary=None if boundary.is_default else boundary.tier.value,
            )
            short_circuit = hooks.fire_pre_call(ctx)
            prompt_text = ctx.prompt
            system = ctx.system
            messages = ctx.messages
            model = ctx.model
            provider = ctx.provider
            max_tokens = ctx.max_tokens
            temperature = ctx.temperature
            tools = ctx.tools
            tool_choice = ctx.tool_choice
            response_format = ctx.response_format
            if messages is not None:
                call_prompt_id = None
            elif isinstance(prompt, Prompt) and prompt_text == original_prompt_text:
                call_prompt_id = prompt.id
            else:
                call_prompt_id = self._resolve_prompt_id(wl.id, prompt_text)
            req = request()
            if short_circuit is not None:
                short_circuited = short_circuit.source or short_circuit.provider or "hook"

        adaptive = model == "somm/auto" and short_circuit is None
        route_source = _route_source(
            policy=adaptive or (bool(wl.policy) and provider is None),
            learned_override=learned_override,
            pinned=bool(provider or model),
        )
        if short_circuit is not None:
            route_mode = "hook"
        elif adaptive:
            route_mode = "adaptive"
        else:
            route_mode = "pinned" if provider is not None else "chain"
        routing_metadata: dict = {}
        if adaptive:
            if provider is not None:
                raise ValueError("somm/auto chooses its provider; omit the provider pin")
            from somm.adaptive import decide_workload

            decision = decide_workload(wl, messages or [{"role": "user", "content": prompt_text}])
            routing_metadata = decision.as_dict()
            provider, model = decision.target.provider, decision.target.model
            req.model = model
            req.reasoning_effort = decision.target.reasoning_effort
            if provider != "chatgpt" and req.messages:
                from somm.adaptive import fallback_messages

                req.messages = fallback_messages(req.messages)
            allow_fallback_resolved = False

        prompt_hash = stable_hash(messages if messages is not None else prompt_text)
        # Data boundary: refuse before any provider call when no route within
        # the workload's boundary remains. A pin does not override it.
        if short_circuit is None and not boundary.is_default:
            req.data_boundary = boundary.tier.value
            req.metadata = {
                **(req.metadata or {}),
                "somm_workload": workload,
                "somm_boundary_source": boundary.source,
            }
            self._admit_or_refuse(
                wl,
                workload,
                boundary,
                pinned=provider,
                model=model,
                policy=wl.policy,
                prompt_hash=prompt_hash,
                observation_role=observation_role,
                source_call_id=source_call_id,
                session_id=session_id,
                parent_call_id=parent_call_id,
                correlation_id=correlation_id,
                max_tokens=max_tokens,
                route_source=route_source,
            )

        # Fail-closed budget gate: refuse before any provider call once the
        # workload's daily cap is reached (inert unless budget_fail_closed).
        # Runs AFTER pre_call so a short-circuit (e.g. a cache hit) — which
        # spends nothing — can still serve a capped workload; a real provider
        # call is still refused.
        if short_circuit is None and provider != "chatgpt":
            self._enforce_budget(wl)

        return _Plan(
            wl=wl, workload=workload, req=req, prompt_text=prompt_text, messages=messages,
            system=system, model=model, provider=provider, max_tokens=max_tokens,
            temperature=temperature, call_prompt_id=call_prompt_id, prompt_hash=prompt_hash,
            short_circuit=short_circuit, short_circuited=short_circuited,
            route_source=route_source, route_mode=route_mode,
            routing_metadata=routing_metadata, allow_fallback=allow_fallback_resolved,
            wait=wait, wait_on_exhausted=wait_on_exhausted, session_id=session_id,
            parent_call_id=parent_call_id, correlation_id=correlation_id,
            observation_role=observation_role, source_call_id=source_call_id,
        )

    def _dispatch(self, plan: _Plan, attempt: Callable) -> _Attempted:
        """Try the planned route: a pin (sticky unless ``allow_fallback``),
        a workload policy, or the chain. ``attempt(provider, request)`` is
        the one provider call; the router owns cooldowns, capability and
        plan filters, retries and waiting. Never raises a provider failure:
        it is returned on the attempt for :meth:`_record`."""
        att = _Attempted(model=plan.model or "")
        req, provider, model = plan.req, plan.provider, plan.model
        if plan.short_circuit is not None:
            att.first = [plan.short_circuit.provider, plan.short_circuit.model]
            return att
        if provider is not None:
            chosen = self._pick_provider(provider)
            att.first = [chosen.name, model or getattr(chosen, "default_model", "") or ""]
            try:
                att.resp, att.provider = attempt(chosen, req), chosen.name
                return att
            except Exception as exc:
                att.skipped.append([*att.first, type(exc).__name__])
                att.provider = chosen.name
                if not plan.allow_fallback:
                    # Default since v0.16: a pin is honored or the call fails.
                    # Rescuing a pinned call through the chain answers with a
                    # model the caller never asked for, which made results
                    # unexplainable — and silently invalidated evaluation,
                    # comparison, and replay runs. Surface the original error
                    # with the pinned attribution intact. Attribute it to a
                    # real model even when only the provider was pinned: a
                    # blank model column hides the row from per-model queries.
                    att.failure = exc
                    att.error_detail = _pinned_error_detail(exc, chosen.name, model)
                    att.model = (
                        model
                        or getattr(exc, "model", "")
                        or getattr(chosen, "default_model", "")
                        or ""
                    )
                    return att
                pinned_failure = {
                    "pinned_provider": provider,
                    "pinned_model": model,
                    "error_kind": type(exc).__name__,
                    "error_detail": _format_error_detail(exc, chosen.name, model),
                }
            # Opt-in rescue (allow_fallback=True / SOMM_PINNED_FALLBACK=1): an
            # answer from some healthy provider beats no answer. The pinned
            # model means nothing to other providers (re-using it guarantees
            # a 404), so the chain runs on each provider's own default.
            req.model = None
            rescue_route: dict = {}
            try:
                routed = self.router.dispatch(
                    req, wait=plan.wait_on_exhausted, route=rescue_route, attempt=attempt
                )
                att.resp, att.provider = routed.response, routed.provider
                att.fallback_info = pinned_failure
            except Exception as chain_exc:
                att.failure = chain_exc
                att.error_detail = _format_error_detail(chain_exc, chosen.name, model)
                if getattr(chain_exc, "model", None):
                    att.model = chain_exc.model
                if plan.wait_on_exhausted is not None and isinstance(
                    chain_exc, SommProvidersExhausted
                ):
                    att.raise_after = chain_exc
            att.skipped.extend(rescue_route.get("skipped", []))
            return att

        raise_exhausted = plan.wait_on_exhausted is not None
        chain_route: dict = {}
        try:
            policy = plan.wl.policy
            if policy:
                retry_cfg = policy.get("retry") or {}
                policy_wait = plan.wait_on_exhausted
                if plan.wait is _WAIT_UNSET and retry_cfg.get("deadline_s") is not None:
                    policy_wait = float(retry_cfg["deadline_s"])
                raise_exhausted = retry_cfg.get("max") is not None or policy_wait is not None
                routed = self._policy_router(policy, explicit_model=model is not None).dispatch(
                    req,
                    wait=policy_wait,
                    retry_max=retry_cfg.get("max"),
                    retry_backoff_s=retry_cfg.get("backoff_s"),
                    route=chain_route,
                    attempt=attempt,
                )
            else:
                routed = self.router.dispatch(
                    req, wait=plan.wait_on_exhausted, route=chain_route, attempt=attempt
                )
            att.resp, att.provider = routed.response, routed.provider
            att.fallback_info = _routed_past_info(routed, model)
        except Exception as exc:
            att.failure = exc
            if getattr(exc, "model", None):
                att.model = exc.model
            att.error_detail = _format_error_detail(exc, "", att.model)
            if raise_exhausted and isinstance(exc, SommProvidersExhausted):
                att.raise_after = exc
        att.first = chain_route.get("first")
        att.skipped = chain_route.get("skipped", [])
        return att

    def _record(
        self,
        plan: _Plan,
        att: _Attempted,
        *,
        ttft_ms: int | None = None,
        routing_extra: dict | None = None,
    ) -> SommResult:
        """Write the one ``calls`` row for this call and tell the listeners:
        outcome, route, cost and provenance, mark hook, shadow sample,
        on_fallback, the budget warning. Returns the caller's result."""
        sc = plan.short_circuit
        outcome = Outcome.OK
        error_kind = error_detail = None
        cost_usd_out = reported_cost_source = sent_max_tokens = None
        if sc is not None:
            text, actual_provider, actual_model = sc.text, sc.provider, sc.model
            tokens_in, tokens_out, latency_ms = sc.tokens_in, sc.tokens_out, 0
            tool_calls_out = _to_core_tool_calls(sc.tool_calls)
            stop_reason_out = reasoning_content_out = ""
            raw_out = sc.raw
            cost_usd_out = sc.cost_usd if sc.cost_usd is not None else 0.0
            responded = True
        elif att.resp is not None:
            (
                text, actual_provider, actual_model, tokens_in, tokens_out, latency_ms,
                tool_calls_out, stop_reason_out, reasoning_content_out, raw_out,
                cost_usd_out, reported_cost_source, outcome, error_kind, error_detail,
            ) = _served(att.resp, att.provider, plan.max_tokens)
            sent_max_tokens = att.resp.max_tokens_sent
            responded = True
        else:
            text, actual_provider, actual_model = "", att.provider, att.model
            tokens_in = tokens_out = latency_ms = 0
            tool_calls_out = []
            stop_reason_out = reasoning_content_out = ""
            raw_out = None
            responded = False
        if att.failure is not None:
            # A stream the caller stopped reading is incomplete, not wrong.
            abandoned = not isinstance(att.failure, Exception)
            outcome = Outcome.UNKNOWN if abandoned else _failure_outcome(att.failure)
            error_kind = (
                "StreamAbandoned"
                if isinstance(att.failure, GeneratorExit)
                else type(att.failure).__name__
            )
            error_detail = att.error_detail

        routing_metadata = dict(plan.routing_metadata)
        if not actual_provider and att.skipped:
            # No route answered and none is named on the row: say what was
            # tried and why each candidate was passed over.
            error_detail = (
                f"{error_detail or error_kind} | route: {_route_summary(att.skipped)}"
            )[:512]
        routing_metadata["route_source"] = plan.route_source
        routing_metadata["route"] = _route_record(
            plan.route_mode,
            att.first,
            [actual_provider, actual_model] if responded else None,
            att.skipped,
            sent_max_tokens or plan.max_tokens,
        )
        routing_metadata.update(routing_extra or {})

        cache_tokens_in, cache_tokens_out = extract_cache_tokens(raw_out)
        reasoning_tokens = extract_reasoning_tokens(raw_out)
        citations = extract_citations(raw_out)
        provider_request_id, billing_id = _provider_custody(raw_out)
        call_id = str(uuid.uuid4())
        result = SommResult(
            text=text,
            provider=actual_provider,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=(
                cost_usd_out
                if cost_usd_out is not None
                else cost_for_call(
                    self.repo, actual_provider, actual_model, tokens_in, tokens_out,
                    cache_read_tokens=cache_tokens_in, cache_write_tokens=cache_tokens_out,
                )
            ),
            call_id=call_id,
            outcome=outcome,
            error_kind=error_kind,
            error_detail=error_detail,
            raw=raw_out,
            tool_calls=tool_calls_out,
            stop_reason=stop_reason_out,
            reasoning_content=reasoning_content_out,
            ttft_ms=ttft_ms,
            cache_tokens_in=cache_tokens_in,
            cache_tokens_out=cache_tokens_out,
            reasoning_tokens=reasoning_tokens,
            citations=citations,
        )

        cost_basis, cost_kind, cost_accuracy, cost_source, pricing_version = _cost_provenance(
            self.repo,
            actual_provider,
            actual_model,
            reported=cost_usd_out is not None,
            source=reported_cost_source or plan.short_circuited,
        )
        if raw_out and raw_out.get("somm_recovery"):
            routing_metadata["recovery"] = raw_out["somm_recovery"]
        # A provider that pinned the request (OpenRouter under a strict
        # boundary) says so on the response: the tier it put on the wire, the
        # upstreams it allowed, and which one served when the response names it.
        pinned_route = raw_out.get("somm_route") if isinstance(raw_out, dict) else None
        if isinstance(pinned_route, dict):
            routing_metadata["upstream"] = pinned_route
        call = Call(
            routing_json=json.dumps(routing_metadata, default=str),
            id=call_id,
            ts=plan.ts,
            project=self.config.project,
            workload_id=plan.wl.id,
            prompt_id=plan.call_prompt_id,
            provider=actual_provider,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=result.cost_usd,
            outcome=outcome,
            error_kind=error_kind,
            error_detail=error_detail,
            # Multi-turn calls pass `messages` and a placeholder `prompt`;
            # hash the actual conversation so replay/cache/dedup keys off
            # real content rather than an ignored arg.
            prompt_hash=plan.prompt_hash,
            response_hash=stable_hash(text),
            correlation_id=(
                plan.correlation_id
                if plan.correlation_id is not None
                else hooks.current_correlation_id()
            ),
            temperature=plan.temperature,
            max_tokens=plan.max_tokens,
            ttft_ms=ttft_ms,
            session_id=plan.session_id,
            parent_call_id=plan.parent_call_id,
            cache_tokens_in=cache_tokens_in,
            cache_tokens_out=cache_tokens_out,
            citations_json=_citations_json(citations),
            cost_basis=cost_basis,
            cost_kind=cost_kind,
            cost_accuracy=cost_accuracy,
            cost_source=cost_source,
            pricing_version=pricing_version,
            provider_request_id=provider_request_id,
            billing_id=billing_id,
            call_site=hooks.current_call_site(),
            reasoning_tokens=reasoning_tokens,
            client_version=_CLIENT_VERSION,
            observation_role=plan.observation_role or "production",
            source_call_id=plan.source_call_id,
            data_tier=(
                getattr(sc, "data_tier", None)
                if sc is not None
                else (pinned_route or {}).get("data_tier")
                or self._served_tier(actual_provider, actual_model)
            ),
        )
        self._finish_call(
            call,
            workload=plan.workload,
            short_circuited=plan.short_circuited,
            temperature=plan.temperature,
            max_tokens=plan.max_tokens,
        )
        result._on_mark = MarkHook(self._mark_persister(call))
        self._maybe_capture_sample(
            plan.wl, call_id, plan.prompt_text, plan.messages, text, outcome
        )

        # Fire on_fallback when a pinned call got rescued by the chain AND
        # ended up on a DIFFERENT (provider, model) than what was pinned, or
        # when the chain routed past providers that rejected their keys.
        # Same-model recovery is just a retry, not a pin-vs-actual
        # divergence; keeping the signal narrow prevents alert fatigue on
        # free-tier providers that intermittently 429 or return blanks.
        info = att.fallback_info
        if (
            outcome == Outcome.OK
            and info is not None
            and not (
                info["pinned_provider"] == actual_provider
                and (info["pinned_model"] or "") == (actual_model or "")
            )
            and self._on_fallback is not None
        ):
            with contextlib.suppress(Exception):  # the hook must not break the caller
                self._on_fallback(
                    {
                        "call_id": call_id,
                        "workload": plan.workload,
                        "pinned_provider": info["pinned_provider"],
                        "pinned_model": info["pinned_model"],
                        "actual_provider": actual_provider,
                        "actual_model": actual_model,
                        "error_kind": info["error_kind"],
                        "error_detail": info["error_detail"],
                        "routed_past": info.get("routed_past", []),
                    }
                )

        self._warn_if_over_budget(plan.wl, result.cost_usd)
        return result

    def _warn_if_over_budget(self, wl, cost_usd: float) -> None:
        """Warn once per workload per day when its daily cost passes its cap."""
        if wl.budget_cap_usd_daily is None or cost_usd <= 0:
            return
        budget_key = (wl.name, date.today())
        if budget_key in _warned_budget_exceeded:
            return
        with self.repo._open() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls "
                "WHERE workload_id = ? AND date(ts) = date('now') "
                "AND budget_eligible != 0",
                (wl.id,),
            ).fetchone()
        daily_cost = row[0] if row else 0.0
        if daily_cost > wl.budget_cap_usd_daily:
            _warned_budget_exceeded.add(budget_key)
            print(
                f"[somm] WARNING: workload {wl.name!r} daily cost "
                f"${daily_cost:.4f} exceeds budget cap "
                f"${wl.budget_cap_usd_daily:.4f}.",
                file=sys.stderr,
            )

    async def agenerate(
        self,
        prompt: str | list[dict] | Prompt,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
        capabilities_required: list[str] | None = None,
        allow_empty: bool = False,
        allow_fallback: bool | None = None,
        no_fallback: bool = False,
        raise_on_empty: bool = False,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        wait: float | None = _WAIT_UNSET,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        correlation_id: str | None = None,
        observation_role: str = "production",
        source_call_id: str | None = None,
        claude_cli_text_only: bool = False,
        data_boundary: str | None = None,
        timeout: float | None = None,
        raise_on_failure: bool = False,
    ) -> SommResult:
        """Await :meth:`generate` without blocking the caller's event loop.

        The synchronous method remains the single implementation of workload
        resolution, routing, fallback, plan governance, hooks, and telemetry.
        Provider I/O runs in a worker thread because the provider protocol is
        synchronous; context variables are propagated by ``asyncio.to_thread``.
        """
        import asyncio

        return await asyncio.to_thread(
            self.generate,
            prompt,
            system=system,
            workload=workload,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            provider=provider,
            capabilities_required=capabilities_required,
            allow_empty=allow_empty,
            allow_fallback=allow_fallback,
            no_fallback=no_fallback,
            raise_on_empty=raise_on_empty,
            tools=tools,
            messages=messages,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            claude_cli_text_only=claude_cli_text_only,
            wait=wait,
            session_id=session_id,
            parent_call_id=parent_call_id,
            correlation_id=correlation_id,
            observation_role=observation_role,
            source_call_id=source_call_id,
            data_boundary=data_boundary,
            timeout=timeout,
            raise_on_failure=raise_on_failure,
        )

    # ------------------------------------------------------------------
    # Embeddings

    def embed(
        self,
        text: str,
        *,
        workload: str = "default",
        model: str | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
    ) -> EmbedResult:
        """Compute one embedding via local ollama. v1 forces ollama with
        no fallback chain (caller asked for "force ollama, no fallbacks").

        Telemetry mirrors generate(): a row lands in calls.sqlite with
        provider/model/latency/outcome/error_detail, so failed embeds show
        up in `somm status` and `somm tail` like every other call.

        Args:
            text: input string. Single-string only — batch is the caller's
                  concern (loop with their own cache, mirroring how
                  tiling-check.py keys on sha256(model + body)).
            workload: registered workload name. Auto-registers in observe
                      mode; raises in strict mode (mirrors generate()).
            model: ollama embed model. Defaults to `nomic-embed-text` via
                   the ollama provider's `default_embed_model`.

        Returns:
            EmbedResult. On failure, `embedding=[]` and `outcome != OK`;
            inspect `error_detail` for the operator-friendly description.
        """
        wl = self.repo.workload_by_name(workload, self.config.project)
        if wl is None:
            if self.config.mode == "strict":
                workload_arg = shlex.quote(workload)
                raise SommStrictMode(
                    f"SOMM_WORKLOAD_UNREGISTERED\n\n"
                    f"Problem: This embed call used workload {workload!r}, "
                    f"but it is not registered.\n"
                    f"Cause: strict mode requires workload metadata.\n"
                    f"Fix: somm workload add {workload_arg} --from-example structured-extraction\n"
                    f"     # or: export SOMM_MODE=observe"
                )
            wl = self.register_workload(name=workload)

        # v1: pin to ollama. No router involvement — `force ollama, no
        # fallbacks` is the explicit posture embeddings callers asked for.
        # Loosening this means deciding what cross-provider embedding
        # routing should look like (different models = different vector
        # spaces; cosine across them is meaningless), so it's a design
        # call we're deferring.
        provider_obj = None
        for p in self.providers:
            if p.name == "ollama":
                provider_obj = p
                break
        if provider_obj is None:
            raise ValueError(
                "embed() requires the ollama provider to be configured "
                "(it is in the default chain — check $SOMM_PROVIDER_ORDER "
                "if you customized it)"
            )

        call_id = str(uuid.uuid4())
        ts = datetime.now(UTC)
        outcome = Outcome.OK
        error_kind: str | None = None
        error_detail: str | None = None
        embedding: list[float] = []
        actual_model = model or provider_obj.default_embed_model
        tokens_in = 0
        latency_ms = 0
        raw_out: dict | None = None

        req = SommEmbedRequest(text=text, model=model)
        short_circuit: hooks.ShortCircuit | None = None
        short_circuited: str | None = None
        if hooks.has_hooks(hooks.PRE_CALL):
            ctx = hooks.PreCallContext(
                workload=workload,
                prompt=text,
                system="",
                messages=None,
                model=model,
                provider="ollama",
                max_tokens=0,
                temperature=0.0,
                tools=[],
                tool_choice=None,
                project=self.config.project,
            )
            short_circuit = hooks.fire_pre_call(ctx)
            text = ctx.prompt
            model = ctx.model
            actual_model = model or provider_obj.default_embed_model
            req = SommEmbedRequest(text=text, model=model)
            if short_circuit is not None:
                short_circuited = short_circuit.source or short_circuit.provider or "hook"

        # Same fail-closed budget gate as generate(): a capped workload may
        # still be served from a short-circuit, never from a real call.
        if short_circuit is None:
            self._enforce_budget(wl)

        if short_circuit is not None:
            raw_embedding = (
                short_circuit.raw.get("embedding", [])
                if isinstance(short_circuit.raw, dict)
                else []
            )
            embedding = list(raw_embedding) if raw_embedding is not None else []
            actual_model = short_circuit.model or actual_model
            tokens_in = short_circuit.tokens_in
            latency_ms = 0
            cost_usd = short_circuit.cost_usd if short_circuit.cost_usd is not None else 0.0
        else:
            try:
                resp = provider_obj.embed(req)
                embedding = resp.embedding
                actual_model = resp.model
                tokens_in = resp.tokens_in
                latency_ms = resp.latency_ms
                raw_out = resp.raw
            except Exception as exc:
                outcome = Outcome.UPSTREAM_ERROR
                error_kind = type(exc).__name__
                error_detail = _format_error_detail(exc, "ollama", actual_model)

            # Embeddings bill on input tokens only (no output). Computed once and
            # shared across the result, the telemetry row, and the observer event.
            cost_usd = cost_for_call(self.repo, "ollama", actual_model, tokens_in, 0)
        if short_circuit is not None:
            raw_out = short_circuit.raw
        provider_request_id, billing_id = _provider_custody(raw_out)

        result = EmbedResult(
            embedding=embedding,
            provider=short_circuit.provider if short_circuit is not None else "ollama",
            model=actual_model,
            dim=len(embedding),
            tokens_in=tokens_in,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            call_id=call_id,
            outcome=outcome,
            error_kind=error_kind,
            error_detail=error_detail,
            raw=raw_out,
        )

        # response_hash: sha256 of the joined float repr. Cheap dedup
        # signal in calls.sqlite without persisting the full vector.
        response_hash = stable_hash(",".join(f"{v:.6f}" for v in embedding))
        cost_basis, cost_kind, cost_accuracy, cost_source, pricing_version = _cost_provenance(
            self.repo,
            result.provider,
            actual_model,
            reported=short_circuit is not None,
            source=short_circuited,
        )
        call = Call(
            id=call_id,
            ts=ts,
            project=self.config.project,
            workload_id=wl.id,
            prompt_id=None,
            provider=result.provider,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=0,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            outcome=outcome,
            error_kind=error_kind,
            error_detail=error_detail,
            prompt_hash=stable_hash(text),
            response_hash=response_hash,
            correlation_id=hooks.current_correlation_id(),
            call_site=hooks.current_call_site(),
            client_version=_CLIENT_VERSION,
            session_id=session_id,
            parent_call_id=parent_call_id,
            cost_basis=cost_basis,
            cost_kind=cost_kind,
            cost_accuracy=cost_accuracy,
            cost_source=cost_source,
            pricing_version=pricing_version,
            provider_request_id=provider_request_id,
            billing_id=billing_id,
        )
        self._finish_call(call, workload=workload, short_circuited=short_circuited)

        return result

    async def aembed(
        self,
        text: str,
        *,
        workload: str = "default",
        model: str | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
    ) -> EmbedResult:
        """Await :meth:`embed` without blocking the caller's event loop.

        The synchronous method remains the single implementation of provider
        I/O, hooks, costing, and telemetry. Because that implementation is
        intentionally Ollama-only, fail loudly when the configured provider
        order pins something else first instead of silently overriding it.
        """
        configured_provider = (
            self.config.provider_order[0]
            if self.config.provider_order
            else self.providers[0].name if self.providers else None
        )
        if configured_provider not in (None, "ollama"):
            raise ValueError(
                f"aembed() cannot honor configured provider {configured_provider!r}; "
                "the shared embed() implementation currently supports only 'ollama'. "
                "Put 'ollama' first in $SOMM_PROVIDER_ORDER or use the configured "
                "provider's native embedding API."
            )

        import asyncio

        return await asyncio.to_thread(
            self.embed,
            text,
            workload=workload,
            model=model,
            session_id=session_id,
            parent_call_id=parent_call_id,
        )

    # ------------------------------------------------------------------

    def _pinned_only_providers(self) -> dict[str, SommProvider]:
        """Executors reachable by pin but deliberately outside the chain.

        Built once and cached: constructing a CLI executor shells out to
        ``shutil.which``, and ``_pick_provider`` runs on every call.

        Chain members win — a provider configured into the routing order is
        the instance that answers its own name, so a pin never reaches a
        second copy with different settings.
        """
        if self._pinned_only_cache is None:
            from somm.providers.registry import build_pinned_only_providers

            self._pinned_only_cache = build_pinned_only_providers(
                self.config,
                self._tracker,
                exclude={p.name for p in self.providers},
            )
        return self._pinned_only_cache

    def all_providers(self) -> list[SommProvider]:
        """The routing chain, then the pin-only executors behind it."""
        return [*self.providers, *self._pinned_only_providers().values()]

    def health(self) -> dict[str, dict]:
        """Every known provider's state, read from what somm already records.

        No network call: this reads the cooldown table and the ledger, so it
        is cheap enough to call before every batch. Keyed by provider name:

        - ``configured``: built on this machine (key set, binary found);
          ``in_chain``: in the routing order rather than reachable by pin only.
        - ``cooled`` / ``cooled_until``: the whole provider is sitting out
          until that UTC time; ``cooled_models`` maps models cooling on their
          own to theirs. ``consecutive_failures`` and ``last_ok_at`` as kept.
        - ``last_error_kind`` / ``last_error_at``: the newest delivery
          failure (empty, timeout, rate limit, upstream error, exhaustion)
          among its last 100 recorded calls, or None.
        """
        now = datetime.now(UTC)
        built = self.all_providers()
        chain = {p.name for p in self.providers}
        configured = {p.name for p in built}
        names = dict.fromkeys(
            [*(p.name for p in built), *(spec.name for spec in BUILTIN_PROVIDER_SPECS)]
        )
        report = {
            name: {
                "configured": name in configured,
                "in_chain": name in chain,
                "cooled": False,
                "cooled_until": None,
                "cooled_models": {},
                "consecutive_failures": 0,
                "last_ok_at": None,
                "last_error_kind": None,
                "last_error_at": None,
            }
            for name in names
        }
        with self.repo._open() as conn:
            for name, model, last_ok, until, failures in conn.execute(
                "SELECT provider, model, last_ok_at, cooldown_until, consecutive_failures "
                "FROM provider_health"
            ):
                entry = report.get(name)
                if entry is None:
                    continue
                cooling_until = _parse_ts(until)
                cooling = cooling_until is not None and cooling_until > now
                if model == "":
                    entry["cooled"] = cooling
                    entry["cooled_until"] = cooling_until.isoformat() if cooling else None
                    entry["consecutive_failures"] = failures or 0
                    entry["last_ok_at"] = last_ok
                elif cooling:
                    entry["cooled_models"][model] = cooling_until.isoformat()
            for name in configured:
                row = conn.execute(
                    "SELECT error_kind, outcome, ts FROM ("
                    "  SELECT error_kind, outcome, ts FROM calls WHERE provider = ?"
                    "  ORDER BY ts DESC LIMIT 100"
                    ") WHERE outcome IN ('empty', 'timeout', 'rate_limit', 'upstream_error', "
                    "'exhausted') ORDER BY ts DESC LIMIT 1",
                    (name,),
                ).fetchone()
                if row:
                    report[name]["last_error_kind"] = row[0] or row[1]
                    report[name]["last_error_at"] = row[2]
        return report

    def _pick_provider(self, name: str | None) -> SommProvider:
        if name:
            for p in self.providers:
                if p.name == name:
                    return p
            # A seat executor is pinned precisely to leave the metered chain,
            # so it is not IN the chain to be found above. Reaching it here
            # keeps the pin honest without changing routing for anyone else.
            pinned_only = self._pinned_only_providers().get(name)
            if pinned_only is not None:
                return pinned_only
            from somm.providers.registry import pinned_only_unavailable_reason

            reason = pinned_only_unavailable_reason(name, self.config)
            if reason:
                # Say where somm looked: "not configured" alone reads as "not
                # installed" when the unit's PATH is the real problem.
                raise SommProviderNotConfigured(
                    f"provider {name!r} not configured: {reason}", provider=name
                )
            raise SommProviderNotConfigured(f"provider {name!r} not configured", provider=name)
        return self.providers[0]

    # ------------------------------------------------------------------
    # Streaming

    def stream(
        self,
        prompt: str | list[dict] | Prompt,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
        wait: float | None = _WAIT_UNSET,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        data_boundary: str | None = None,
        timeout: float | None = None,
        *,
        events: bool = False,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        capabilities_required: list[str] | None = None,
        allow_empty: bool = False,
        allow_fallback: bool | None = None,
        correlation_id: str | None = None,
        observation_role: str = "production",
        source_call_id: str | None = None,
        claude_cli_text_only: bool = False,
    ) -> Iterator[str] | Iterator[StreamEvent]:
        """Stream the answer as it arrives. Yields user-visible text chunks
        (``<think>`` blocks stripped across chunk boundaries); with
        ``events=True``, yields :mod:`somm.events` instead — ``TextDelta``,
        ``ToolCallDelta``, ``ReasoningDelta`` — and finally one
        ``StreamDone`` carrying the recorded ``SommResult``.

        A stream is the same call as :meth:`generate`: same workload policy,
        pins, ``somm/auto``, capability and boundary admission, budget gate,
        cooldowns and route record, and the same arguments where they make
        sense for a stream. One rule is its own: an attempt that fails before
        anything reached the caller falls through to the next candidate
        exactly as generate does; once a chunk has been yielded it cannot be
        taken back, so a later failure is recorded and raised. A pinned
        ``provider`` is sticky, as in generate.

        A stream raises its failures (a typed ``SommError``); an empty answer
        ends the stream and is recorded ``empty``. Token counts come from the
        provider's final usage report when it sends one, otherwise from a
        length estimate; the row's ``routing_json["stream"]["usage"]`` says
        which. Telemetry is written when the stream ends, is abandoned, or
        fails.

        Usage:
            for piece in llm.stream("tell a story", workload="story"):
                print(piece, end="", flush=True)
        """
        plan = self._plan_call(
            prompt, system=system, workload=workload, max_tokens=max_tokens,
            temperature=temperature, model=model, provider=provider,
            capabilities_required=capabilities_required, allow_empty=allow_empty,
            allow_fallback=allow_fallback, no_fallback=False, tools=tools,
            messages=messages, tool_choice=tool_choice, response_format=response_format,
            reasoning_effort=reasoning_effort, wait=wait, session_id=session_id,
            parent_call_id=parent_call_id, correlation_id=correlation_id,
            observation_role=observation_role, source_call_id=source_call_id,
            claude_cli_text_only=claude_cli_text_only, data_boundary=data_boundary,
            timeout=timeout,
        )
        t0 = time.monotonic()
        opener = _StreamOpener(events)
        att = self._dispatch(plan, opener)
        if (
            att.resp is None
            and opener.empty is not None
            and att.skipped
            and att.skipped[-1][2].startswith("empty")
        ):
            # The chain ended on an empty answer: for a stream that answer is
            # the result (recorded empty), as it has always been.
            att.resp, att.provider = opener.empty, opener.empty.provider.name
            att.failure = att.raise_after = att.error_detail = None
        start: _StreamStart | None = att.resp
        ttft_ms: int | None = None
        try:
            if plan.short_circuit is not None:
                pending = [TextDelta(plan.short_circuit.text)] if plan.short_circuit.text else []
            else:
                pending = start.events() if start is not None else []
            for event in pending:
                if ttft_ms is None:
                    ttft_ms = int((time.monotonic() - t0) * 1000)
                if events:
                    yield event
                elif isinstance(event, TextDelta):
                    yield event.text
        except BaseException as exc:
            # A provider failure after the first chunk, or the caller stopped
            # reading (GeneratorExit, cancellation): record what happened.
            att.failure = exc
            served = start.model if start is not None else att.model
            if isinstance(exc, Exception):
                att.error_detail = _format_error_detail(exc, att.provider, served)
            else:
                kind = "StreamAbandoned" if isinstance(exc, GeneratorExit) else type(exc).__name__
                att.error_detail = (
                    f"{kind}: caller stopped reading after {len(start.collected) if start else 0} chars"
                )
            self._finish_stream(plan, att, start, t0, ttft_ms)
            raise
        result = self._finish_stream(plan, att, start, t0, ttft_ms)
        if att.failure is not None:
            raise att.failure
        if events:
            yield StreamDone(result, usage=start.usage if start is not None else "provider")

    def _finish_stream(self, plan, att, start, t0: float, ttft_ms: int | None) -> SommResult:
        """Record a stream that ended, failed or was abandoned; a failure
        raised to the caller names the row it was recorded on."""
        if start is not None:
            att.resp = start.response(plan, t0)
        result = self._record(plan, att, ttft_ms=ttft_ms, routing_extra=_stream_extra(start))
        if isinstance(att.failure, SommError):
            _failure_error(result, att.failure, allow_empty=True)
        return result

    async def astream(
        self,
        prompt: str | list[dict] | Prompt,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
        wait: float | None = _WAIT_UNSET,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        data_boundary: str | None = None,
        timeout: float | None = None,
        *,
        events: bool = False,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        capabilities_required: list[str] | None = None,
        allow_empty: bool = False,
        allow_fallback: bool | None = None,
        correlation_id: str | None = None,
        observation_role: str = "production",
        source_call_id: str | None = None,
        claude_cli_text_only: bool = False,
    ) -> AsyncIterator[str] | AsyncIterator[StreamEvent]:
        """Asynchronously iterate over :meth:`stream` without blocking the loop.

        The synchronous generator owns the whole call; each step runs off the
        event loop, and closing this iterator closes it, so a partially read
        stream is still recorded.
        """
        import asyncio

        iterator = self.stream(
            prompt, system=system, workload=workload, max_tokens=max_tokens,
            temperature=temperature, model=model, provider=provider, wait=wait,
            session_id=session_id, parent_call_id=parent_call_id,
            data_boundary=data_boundary, timeout=timeout, events=events, messages=messages,
            tools=tools, tool_choice=tool_choice, response_format=response_format,
            reasoning_effort=reasoning_effort, capabilities_required=capabilities_required,
            allow_empty=allow_empty, allow_fallback=allow_fallback,
            correlation_id=correlation_id, observation_role=observation_role,
            source_call_id=source_call_id, claude_cli_text_only=claude_cli_text_only,
        )
        try:
            while True:
                done, piece = await asyncio.to_thread(_next_stream_item, iterator)
                if done:
                    break
                yield piece
        finally:
            await asyncio.to_thread(iterator.close)

    # ------------------------------------------------------------------
    # Structured output

    def generate_structured(
        self,
        prompt,
        *,
        schema=None,
        workload: str = "default",
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
        model: str | None = None,
        provider: str | None = None,
        retries: int = 2,
        allow_fallback: bool | None = None,
        reasoning_effort: str | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        wait: float | None = _WAIT_UNSET,
        observation_role: str = "production",
        source_call_id: str | None = None,
        data_boundary: str | None = None,
        timeout: float | None = None,
    ) -> tuple[Any, SommResult]:
        """Ask for JSON. Returns ``(value, result)``; raises on failure.

        This is somm's one way to get JSON from a model. ``schema`` may be a
        Pydantic model, a JSON Schema dict or a callable validator; omit it
        to use the workload's registered ``output_schema``, and with neither
        any JSON object or array is accepted. The schema goes in the system
        prompt and, where the adapter has a native structured-output mode,
        in ``response_format`` too. Each answer runs the ``extract_json``
        repair ladder (fences, prose around the JSON, quote and newline
        quirks); an answer that still does not parse or validate is marked
        ``bad_json`` on the ledger and retried with the error fed back.

        ``value`` is the validated object (a model instance for Pydantic);
        ``result`` says who served it (``provider``, ``model``, ``call_id``,
        ``cost_usd``, ``outcome``) and carries the JSON as ``result.parsed``.
        When no attempt succeeds this raises ``SommStructuredError``, which
        carries ``provider``, ``model``, ``call_id`` and ``last_result``;
        ``delivery_failure`` is True when the model never answered at all.

        A pinned ``provider``/``model`` is sticky, exactly as in generate():
        every retry goes back to the same target rather than wandering onto
        another model mid-schedule. Pass ``allow_fallback=True`` to let a
        failed pin be rescued by the router chain.
        """
        if schema is None:
            wl = self.repo.workload_by_name(workload, self.config.project)
            schema = wl.output_schema if wl is not None else None
        return self._structured(
            prompt,
            schema=schema,
            system=system,
            temperature=temperature,
            retries=retries,
            corrective=True,
            workload=workload,
            max_tokens=max_tokens,
            model=model,
            provider=provider,
            allow_fallback=allow_fallback,
            reasoning_effort=reasoning_effort,
            wait=wait,
            session_id=session_id,
            parent_call_id=parent_call_id,
            observation_role=observation_role,
            source_call_id=source_call_id,
            data_boundary=data_boundary,
            timeout=timeout,
        )

    def _structured(
        self,
        prompt,
        *,
        schema,
        system: str,
        temperature: float,
        retries: int,
        corrective: bool,
        temperature_jitter: float = _STRUCTURED_TEMPERATURE_JITTER,
        **generate_kwargs,
    ) -> tuple[Any, SommResult]:
        """The structured-output loop behind generate_structured and
        extract_structured. ``corrective`` feeds the last raw answer and its
        error back on a retry; extract_structured has always retried without."""
        schema_kind, schema_doc = _classify_structured_schema(schema)
        base_system = _structured_system(system, schema_doc)
        response_format = _json_schema_response_format(schema_doc)
        last_raw = ""
        last_error = "no response"
        last_result: SommResult | None = None
        delivery_failure = False

        for attempt in range(retries + 1):
            attempt_system = base_system
            if corrective and attempt > 0:
                attempt_system = _structured_retry_system(
                    base_system,
                    last_raw,
                    last_error,
                )
            result = self.generate(
                prompt=prompt,
                system=attempt_system,
                temperature=temperature + (attempt * temperature_jitter),
                response_format=response_format,
                **generate_kwargs,
            )
            last_result = result
            last_raw = result.text

            if result.outcome != Outcome.OK and not result.text.strip():
                # Nothing came back to validate. An empty or errored delivery
                # is a provider problem (burst throttling, a 5xx, an exhausted
                # chain) and must not be reported as "the model produced
                # invalid JSON" — that misread eliminated a provider from a
                # bake-off once (issue #37). Leave the row's outcome alone;
                # it already says what happened.
                last_error = (
                    f"{result.outcome.value}: "
                    f"{result.error_detail or result.error_kind or 'no model output'}"
                )
                delivery_failure = True
                if result.error_kind == "SommAuthError":
                    # A rejected key fails the same way on every retry; the
                    # retries only re-hit the provider (issue #71).
                    break
                continue
            delivery_failure = False

            parsed = extract_json(result.text)
            if parsed is None:
                last_error = "response did not contain a parseable JSON object or array"
                _mark_bad_json(result, last_error)
                continue

            try:
                if schema_kind == "any":
                    validated = parsed
                elif schema_kind == "pydantic":
                    validated = schema.model_validate(parsed)
                elif schema_kind == "json_schema":
                    validated = _validate_structured_json_schema(parsed, schema)
                else:
                    validated = schema(parsed)
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                _mark_bad_json(result, last_error)
                continue

            result.parsed = parsed
            return validated, result

        error = SommStructuredError(
            _format_structured_failure(last_raw, last_error, delivery_failure=delivery_failure),
            last_raw=last_raw,
            last_error=last_error,
            last_outcome=last_result.outcome.value if last_result is not None else None,
            delivery_failure=delivery_failure,
            last_result=last_result,
        )
        if last_result is not None:
            error.provider = last_result.provider or None
            error.model = last_result.model or None
            error.call_id = last_result.call_id
        raise error

    async def agenerate_structured(
        self,
        prompt,
        *,
        schema=None,
        workload: str = "default",
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
        model: str | None = None,
        provider: str | None = None,
        retries: int = 2,
        allow_fallback: bool | None = None,
        reasoning_effort: str | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        wait: float | None = _WAIT_UNSET,
        observation_role: str = "production",
        source_call_id: str | None = None,
        data_boundary: str | None = None,
        timeout: float | None = None,
    ) -> tuple[Any, SommResult]:
        """Await :meth:`generate_structured` using the same strict pipeline."""
        import asyncio

        return await asyncio.to_thread(
            self.generate_structured,
            prompt,
            schema=schema,
            workload=workload,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            provider=provider,
            retries=retries,
            allow_fallback=allow_fallback,
            reasoning_effort=reasoning_effort,
            session_id=session_id,
            parent_call_id=parent_call_id,
            wait=wait,
            observation_role=observation_role,
            source_call_id=source_call_id,
            data_boundary=data_boundary,
            timeout=timeout,
        )

    def extract_structured(
        self,
        prompt: str | Prompt,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.1,
        model: str | None = None,
        provider: str | None = None,
        retries: int = 2,
        temperature_jitter: float = 0.05,
        allow_fallback: bool | None = None,
        reasoning_effort: str | None = None,
        wait: float | None = _WAIT_UNSET,
        with_result: bool = False,
        timeout: float | None = None,
    ) -> dict | list | tuple[dict | list, SommResult]:
        """Call the LLM and extract JSON from the response.

        Handles markdown fences, bracket-balanced extraction, qwen2.5 double-
        quote quirk, `<think>` blocks (already stripped by adapters), control
        chars, and unescaped newlines.
        ``prompt`` may be a string or registered ``Prompt`` object; the
        generate call records the prompt version when one is provided or
        hash-matched.

        On parse failure, retries up to `retries` more times. Each retry bumps
        temperature by `temperature_jitter` to break deterministic bad output.
        After total exhaustion, returns `{"raw": <last text>,
        "_somm_parse_err": True}` so the caller can distinguish between "LLM
        said nothing parseable" and "LLM said something parseable".
        Omit ``wait`` to use ``SOMM_WAIT_ON_EXHAUSTED`` when set; pass
        ``None`` or ``0`` to fail fast on provider exhaustion.
        A pinned ``provider``/``model`` is sticky across every retry; pass
        ``allow_fallback=True`` to let a failed pin fall through the chain.
        ``with_result=True`` returns ``(parsed, result)`` so the caller can
        read which provider and model served the parse, the ``call_id``, and
        the cost, instead of re-reading the ledger to find out; on failure
        ``result.ok`` is False and ``result.error`` says why.

        Kept for existing callers: it is the schema-less, sentinel-returning
        form of :meth:`generate_structured`, which new code should use.
        """
        try:
            value, result = self._structured(
                prompt,
                schema=None,
                system=system,
                temperature=temperature,
                retries=retries,
                corrective=False,
                temperature_jitter=temperature_jitter,
                workload=workload,
                max_tokens=max_tokens,
                model=model,
                provider=provider,
                allow_fallback=allow_fallback,
                reasoning_effort=reasoning_effort,
                wait=wait,
                timeout=timeout,
            )
        except SommStructuredError as exc:
            failed = {"raw": exc.last_raw, "_somm_parse_err": True}
            return (failed, exc.last_result) if with_result else failed
        return (value, result) if with_result else value

    async def aextract_structured(
        self,
        prompt: str | Prompt,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.1,
        model: str | None = None,
        provider: str | None = None,
        retries: int = 2,
        temperature_jitter: float = 0.05,
        allow_fallback: bool | None = None,
        reasoning_effort: str | None = None,
        wait: float | None = _WAIT_UNSET,
        with_result: bool = False,
        timeout: float | None = None,
    ) -> dict | list | tuple[dict | list, SommResult]:
        """Await :meth:`extract_structured` using the same retry pipeline."""
        import asyncio

        return await asyncio.to_thread(
            self.extract_structured,
            prompt,
            system=system,
            workload=workload,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            provider=provider,
            retries=retries,
            temperature_jitter=temperature_jitter,
            allow_fallback=allow_fallback,
            reasoning_effort=reasoning_effort,
            wait=wait,
            with_result=with_result,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Prompt versioning

    def register_prompt(
        self,
        workload: str,
        body: str,
        bump: str = "minor",
    ) -> Prompt:
        """Commit a prompt body for a named workload. Idempotent on hash match.

        Args:
            workload: workload name (must exist, or be auto-registered in observe mode).
            body: the prompt body.
            bump: "minor" (default), "major", or an explicit version "vN".
        """
        wl = self._require_workload(workload)
        prompt = register_prompt(self.repo, wl.id, body, bump=bump)
        self._prompt_ids_cache.pop(wl.id, None)
        return prompt

    def set_prompt_label(self, workload: str, label: str, version: str) -> None:
        """Move a project-local prompt label to a workload prompt version."""
        wl = self._require_workload(workload)
        prompt = get_prompt(self.repo, wl.id, version=version)
        set_label(self.repo, wl.id, label, prompt.id)

    def prompt(
        self,
        workload: str,
        version: str = "latest",
        *,
        label: str | None = None,
        bucket_key: str | None = None,
    ) -> Prompt:
        """Fetch a prompt by workload + version or label.

        Use in calling code:
            body = llm.prompt("claim_extract", version="latest").body
            body = llm.prompt("claim_extract", label="production").body
            result = llm.generate(body, workload="claim_extract")

        For weighted labels, bucket_key deterministically selects the variant.
        When omitted, the current correlation id is used so a session resolves
        to a stable variant. Pass the returned Prompt to generate(); telemetry
        records the selected prompt_id for per-version scoring.
        """
        wl = self._require_workload(workload)
        if label is not None:
            resolved = resolve_label(
                self.repo,
                wl.id,
                label,
                bucket_key if bucket_key is not None else hooks.current_correlation_id(),
            )
            if resolved is None:
                from somm.prompts import PromptNotFound

                raise PromptNotFound(f"no prompt label {label!r} for workload {wl.id!r}")
            return resolved
        return get_prompt(self.repo, wl.id, version=version, label=label)

    def fork_prompt(
        self,
        workload: str,
        from_version_or_label: str,
        body: str,
    ) -> Prompt:
        """Create a prompt variant in the same workload with lineage recorded."""
        wl = self._require_workload(workload)
        prompt = fork_prompt_version(self.repo, wl.id, from_version_or_label, body)
        self._prompt_ids_cache.pop(wl.id, None)
        return prompt

    def enable_shadow(
        self,
        workload: str,
        gold_provider: str,
        gold_model: str,
        sample_rate: float = 0.02,
        budget_usd_daily: float = 1.0,
        max_grades_per_run: int = 20,
    ) -> None:
        """Opt a workload in to shadow-eval.

        Off by default per sovereignty + privacy. Private workloads
        (privacy_class=PRIVATE) cannot be shadow-graded — the schema view
        enforces this + the worker defense-in-depth checks again.
        """
        wl = self._require_workload(workload)
        from somm_core.models import PrivacyClass

        if wl.privacy_class == PrivacyClass.PRIVATE:
            from somm.errors import SommPrivacyViolation

            raise SommPrivacyViolation(
                f"SOMM_PRIVACY_VIOLATION\n\n"
                f"Problem: workload {workload!r} is privacy_class=private; shadow-eval forbidden.\n"
                f"Cause: shadow-eval re-sends prompt/response to gold-model provider.\n"
                f"Fix: downgrade privacy_class OR keep shadow off for this workload.\n"
                f"Docs: docs/errors/SOMM_PRIVACY_VIOLATION.md",
                workload=workload,
                data_boundary="local",
                boundary_source="privacy_class=private",
            )
        # Shadow-eval re-sends captured prompts to the gold route, so the gold
        # route must be inside the workload's boundary like any other.
        boundary = self._resolve_boundary(wl, None)
        if not boundary.is_default:
            from somm.boundary import base_url_of, check_route

            gold = None
            for p in self.all_providers():
                if p.name == gold_provider:
                    gold = p
                    break
            check_route(
                provider=gold_provider,
                model=gold_model,
                boundary=boundary,
                workload=workload,
                base_url=base_url_of(gold) if gold is not None else None,
                context="the shadow-eval gold route",
            )
        self.repo.set_shadow_config(
            wl.id,
            {
                "gold_provider": gold_provider,
                "gold_model": gold_model,
                "sample_rate": sample_rate,
                "budget_usd_daily": budget_usd_daily,
                "max_grades_per_run": max_grades_per_run,
            },
        )

    def disable_shadow(self, workload: str) -> None:
        wl = self._require_workload(workload)
        self.repo.set_shadow_config(wl.id, None)

    def _require_workload(self, name: str):
        wl = self.repo.workload_by_name(name, self.config.project)
        if wl is None:
            if self.config.mode == "strict":
                raise _SommStrictMode(
                    f"SOMM_WORKLOAD_UNREGISTERED\n\n"
                    f"Problem: Workload {name!r} is not registered.\n"
                    f"Cause: strict mode requires workload metadata first.\n"
                    f"Fix:\n"
                    f"  somm.llm().repo.register_workload(name={name!r}, project=...)\n"
                    f"Docs: docs/errors/SOMM_WORKLOAD_UNREGISTERED.md"
                )
            wl = self.register_workload(name=name)
        return wl

    # ------------------------------------------------------------------
    # Parallel-worker slot assignment

    def parallel_slots(self, n: int) -> list[str]:
        """Return a striped assignment of provider names for n parallel workers.

        Preserves sovereignty-first ordering and avoids stampeding one
        provider. Cooled providers are excluded. Use:

            assignments = llm.parallel_slots(4)
            # e.g. ['ollama', 'openrouter', 'ollama', 'openrouter']
            for i, provider_name in enumerate(assignments):
                spawn_worker(i, provider=provider_name)
        """
        return _parallel_slots(self.providers, n, tracker=self._tracker)

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drain the writer queue and stop the thread. Optional for short-lived processes."""
        self._writer.flush()
        self._writer.stop()

    def __enter__(self) -> SommLLM:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def llm(**kwargs) -> SommLLM:
    """Factory matching the plan's `somm.llm(project=...)` signature."""
    return SommLLM(**kwargs)


def _to_core_tool_calls(provider_tool_calls) -> list[CoreToolCall]:
    """Convert provider-layer ToolCall dataclasses to the core-layer shape.

    Two separate dataclasses exist so provider modules don't need to
    import from somm-core (and so the public SommResult contract lives
    in core where library users import from). Structurally identical;
    this is a one-to-one translation.
    """
    if not provider_tool_calls:
        return []
    return [
        CoreToolCall(
            id=tc.id,
            name=tc.name,
            arguments=tc.arguments,
            arguments_raw=tc.arguments_raw,
        )
        for tc in provider_tool_calls
    ]


def _merge_caps(*sources: list[str] | None) -> list[str]:
    """Merge capability lists deterministically — preserves declaration order
    of the first source to name each capability."""
    seen: dict[str, None] = {}
    for src in sources:
        if not src:
            continue
        for cap in src:
            if cap and cap not in seen:
                seen[cap] = None
    return list(seen.keys())


class _PolicyProvider:
    """Per-call provider wrapper for workload policy model/timeout overrides."""

    def __init__(
        self,
        provider: SommProvider,
        *,
        model: str | None,
        explicit_model: bool,
        timeout_s: float | None,
    ) -> None:
        if timeout_s is not None and hasattr(provider, "timeout"):
            provider = copy.copy(provider)
            provider.timeout = float(timeout_s)
        self._provider = provider
        self._model = model
        self._explicit_model = explicit_model
        self.name = provider.name
        default_model = getattr(provider, "default_model", None)
        self.default_model = model if model is not None and not explicit_model else default_model

    def generate(self, request: SommRequest):
        req = request
        if self._model is not None and not self._explicit_model:
            req = replace(request, model=self._model)
        return self._provider.generate(req)

    def stream(self, request: SommRequest):
        req = request
        if self._model is not None and not self._explicit_model:
            req = replace(request, model=self._model)
        return self._provider.stream(req)

    def health(self):
        return self._provider.health()

    def models(self):
        return self._provider.models()

    def estimate_tokens(self, text, model):
        return self._provider.estimate_tokens(text, model)


def _policy_providers(
    providers: list[SommProvider],
    policy: dict,
    *,
    explicit_model: bool,
) -> list[SommProvider]:
    timeout_s = policy.get("timeout_s")
    fallback = policy.get("fallback")
    if fallback:
        by_name = {p.name: p for p in providers}
        selected: list[tuple[SommProvider, str | None]] = []
        for entry in fallback:
            provider = by_name.get(entry["provider"])
            if provider is not None:
                selected.append((provider, entry.get("model")))
    else:
        selected = [(provider, None) for provider in providers]
    return [
        _PolicyProvider(
            provider,
            model=model,
            explicit_model=explicit_model,
            timeout_s=timeout_s,
        )
        for provider, model in selected
    ]


def _mirror_workloads(src: Repository, dst: Repository) -> None:
    """Copy workloads rows from src to dst (idempotent on id). Called once
    on SommLLM init when cross_project_enabled is set."""
    try:
        with src._open() as s_conn:
            rows = s_conn.execute(
                "SELECT id, name, project, description, input_schema_json, "
                "output_schema_json, quality_criteria_json, budget_cap_usd_daily, "
                "privacy_class, created_at, shadow_config_json, "
                "capabilities_required_json, policy_json, data_boundary FROM workloads"
            ).fetchall()
        if not rows:
            return
        with dst._open() as d_conn:
            d_conn.executemany(
                """
                INSERT OR IGNORE INTO workloads
                    (id, name, project, description,
                     input_schema_json, output_schema_json, quality_criteria_json,
                     budget_cap_usd_daily, privacy_class, created_at, shadow_config_json,
                     capabilities_required_json, policy_json, data_boundary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            # The mirror's boundary follows the project's (it is read by
            # fleet tooling, never enforced from), so refresh it on start.
            d_conn.executemany(
                "UPDATE workloads SET data_boundary = ? WHERE id = ? AND project = ?",
                [(r[13], r[0], r[2]) for r in rows],
            )
    except Exception:  # noqa: BLE001 — best-effort mirror
        pass


def _warn_mirror_unavailable(path: Path, exc: Exception) -> None:
    key = str(Path(path).resolve())
    if key in _warned_mirror_unavailable:
        return
    _warned_mirror_unavailable.add(key)
    print(
        f"[somm] global mirror unavailable at {key}: {exc}; continuing without it", file=sys.stderr
    )

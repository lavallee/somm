"""SommLLM — the library entry point.

D1 minimal: .generate() against a single provider (ollama), telemetry writes
via WriterQueue, demo-mode default (auto-registers unknown workloads with a
warning), call_id in result for provenance.
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
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from somm_core import EmbedResult, Outcome, SommResult, cost_for_call
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.models import Call, Prompt
from somm_core.models import ToolCall as CoreToolCall
from somm_core.parse import (
    ThinkStreamStripper,
    extract_cache_tokens,
    extract_citations,
    extract_json,
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

from somm import hooks
from somm._redaction import scrub_text
from somm.errors import SommProvidersExhausted, SommStructuredError
from somm.errors import SommStrictMode as _SommStrictMode
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
)
from somm.providers.registry import (
    BUILTIN_PROVIDER_SPECS,
    ProviderSpec,
    load_entrypoint_provider_specs,
)
from somm.routing import ProviderHealthTracker, Router
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
        type_matches = (
            ("object" in expected_types and isinstance(parsed, dict))
            or ("array" in expected_types and isinstance(parsed, list))
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


def _format_structured_failure(last_raw: str, last_error: str) -> str:
    return (
        "SOMM_STRUCTURED_OUTPUT_FAILED\n\n"
        "Problem: generate_structured() could not produce JSON matching the supplied schema.\n"
        f"Cause: Last validation error: {last_error}\n"
        f"Last raw text: {last_raw}\n"
        "Fix: Return only valid JSON matching the requested schema, or relax/replace "
        "the schema validator.\n"
        "Docs: docs/errors/SOMM_STRUCTURED_OUTPUT_FAILED.md"
    )


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
    """Start the somm-service scheduler inside this process, once per DB.

    Library-only deployments that set SOMM_INPROCESS_WORKERS=1 get the
    full intelligence loop (model-intel refresh, online-eval grading,
    recommendations) without running `somm serve`. Requires somm-service.
    """
    key = str(config.db_path)
    if key in _inprocess_schedulers:
        return _inprocess_schedulers[key]
    try:
        from somm_service.inprocess import start_inprocess_scheduler
    except ImportError:
        print(
            "[somm] SOMM_INPROCESS_WORKERS=1 but somm-service is not "
            "installed — `pip install somm-service` to enable the "
            "in-process intelligence loop.",
            file=sys.stderr,
        )
        return None
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
                f"ungraded. Run `somm serve`, `somm-serve admin run-shadow`, "
                f"or set SOMM_INPROCESS_WORKERS=1.",
                file=sys.stderr,
            )
        elif is_stale:
            print(
                f"[somm] online eval is configured for {shadow_n} workload(s) "
                f"but workers have not run since {newest_run_at} — sampled "
                f"calls may be piling up ungraded. Run `somm serve`, "
                f"`somm-serve admin run-shadow`, or set SOMM_INPROCESS_WORKERS=1.",
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
    }


def _fire_call_hooks(event: dict) -> None:
    stamped = hooks.stamp_event(event)
    hooks.fire_post_call(stamped)
    hooks.fire_post_process(stamped)


def build_default_providers(
    config: Config, tracker=None, full: bool = False
) -> list[SommProvider]:
    """Build the provider chain from config.

    Default order (sovereign-first): ollama → openrouter → deepseek →
    minimax → anthropic → gemini → openai → perplexity. Override with
    SOMM_PROVIDER_ORDER env var (comma-separated, e.g.
    "openrouter,minimax,ollama").

    Every commercial-API provider is opt-in via its env var. Library
    works offline with just ollama. Shared by SommLLM and the MCP
    server so both expose the identical chain.

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
            logger.warning("skipping provider spec %r: provider name is already registered", spec.name)
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

    # Default order — sovereign-first, then strong-paid (deepseek now in slot 3).
    builtin_order = [
        spec.name
        for spec in sorted(
            BUILTIN_PROVIDER_SPECS,
            key=lambda spec: spec.default_order_rank if spec.default_order_rank is not None else 10**9,
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
) -> str:
    """Build a bounded diagnostic for the EMPTY outcome.

    Two empirically-confirmed modes (see commit 54aa18f):
      - ``no_content``    — upstream returned tokens_out=0; the model never
                            actually generated (e.g. openrouter elephant-alpha
                            with ``{"content": null}``, sub-500ms latency).
      - ``stripped_empty`` — tokens_out>0 but the visible text is empty after
                             post-processing (e.g. minimax M2.7 generating
                             only inside ``<think>`` blocks; full latency).

    Both fields are kept on the row so SQL filters (``WHERE outcome='empty'
    AND error_detail LIKE '%no_content%'``) can split them without joining.
    """
    hint = "no_content" if tokens_out == 0 else "stripped_empty"
    parts = [
        f"EmptyResponse: {hint}",
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
            "WHERE workload_id = ? AND date(ts) = date('now')",
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
            workload=workload.name, spent_usd=spent, cap_usd=float(cap),
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
    recovered via fallback. The call succeeded; this is *not* an error,
    just a signal that a degradation happened. Suppress via
    `SommLLM(on_fallback=lambda _: None)` or reroute to logging.
    """
    print(
        f"[somm] FALLBACK workload={event.get('workload')} "
        f"pinned={event.get('pinned_provider')}/{event.get('pinned_model')} "
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
        self._plan_governor = _build_plan_governor(self.config)
        self.router = Router(
            self.providers, self._tracker, plan_governor=self._plan_governor
        )
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

        # Decisions are ALWAYS mirrored to the global store — unlike calls
        # (per-project for privacy), advisory memory is explicitly
        # cross-project: "last time I picked a vision model, here's why."
        # We lazily create the handle so cold starts don't touch the global
        # path unless a decision is actually recorded.
        self._decision_mirror: Repository | None = mirror_repo

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

    def set_workload_policy(self, workload: str, policy: dict | None) -> None:
        """Attach or clear a workload routing policy."""
        wl = self._require_workload(workload)
        self.repo.set_workload_policy(wl.id, policy)
        if self._mirror_repo is not None:
            try:
                mirror_wl = self._mirror_repo.workload_by_name(
                    workload, self.config.project
                )
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

    def _maybe_capture_sample(
        self, wl, call_id: str, prompt, messages, text: str, outcome
    ) -> None:
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
        no_fallback: bool = False,
        raise_on_empty: bool = False,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        wait: float | None = _WAIT_UNSET,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        correlation_id: str | None = None,
    ) -> SommResult:
        """Run one LLM call. Writes telemetry synchronously at the row level.

        demo mode: auto-registers unknown workloads as 'ad_hoc' equivalents.
        strict mode: raises SommStrictMode if workload isn't registered.
        ``prompt`` may be a string, multimodal prompt list, or registered
        ``Prompt`` object. Passing ``Prompt`` dispatches its body and stamps
        that prompt version on the telemetry row.

        ``no_fallback``: when a ``provider`` is pinned, suppress the normal
        rescue path through the router chain. The pinned (provider, model)
        either succeeds or the call returns with ``outcome=UPSTREAM_ERROR``
        — useful for evaluation runs where silent fallback to a different
        model invalidates the experiment. Has no effect when ``provider``
        is None (router-driven calls have no pinned target to honor).

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
        """
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

        # Self-healing: apply any learned parameter override for this
        # (workload, model). The agent worker writes these when it detects a
        # recurring capability failure (e.g. a reasoning model that exhausts its
        # output budget on thinking tokens and returns empty). Fail-open and
        # one-directional — only ever RAISES max_tokens, never lowers it, and a
        # lookup failure must never break a live call.
        if wl is not None and model:
            try:
                _ov = self.repo.lookup_learned_override(wl.id, model, provider)
                _floor = _ov.get("max_tokens_floor") if _ov else None
                if _floor and _floor > max_tokens:
                    logging.getLogger("somm.client").info(
                        "somm self-heal: max_tokens %d→%d for workload=%s model=%s (%s)",
                        max_tokens, _floor, workload, model, _ov.get("failure_signature"),
                    )
                    max_tokens = _floor
            except Exception:
                pass  # never let a learned-override lookup break a live call

        wait_on_exhausted = self._resolve_wait_on_exhausted(wait)

        prompt_text = prompt.body if isinstance(prompt, Prompt) else prompt
        call_prompt_id = None if messages is not None else self._resolve_prompt_id(wl.id, prompt)

        effective_caps = _merge_caps(
            wl.capabilities_required,
            capabilities_required,
            infer_capabilities(prompt_text),
        )

        req = SommRequest(
            prompt=prompt_text,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            capabilities_required=effective_caps,
            allow_empty=allow_empty,
            tools=tools or [],
            messages=messages,
            tool_choice=tool_choice,
            response_format=response_format,
        )

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
            effective_caps = _merge_caps(
                wl.capabilities_required,
                capabilities_required,
                infer_capabilities(prompt_text),
            )
            req = SommRequest(
                prompt=prompt_text,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                capabilities_required=effective_caps,
                allow_empty=allow_empty,
                tools=tools or [],
                messages=messages,
                tool_choice=tool_choice,
                response_format=response_format,
            )
            if short_circuit is not None:
                short_circuited = short_circuit.source or short_circuit.provider or "hook"

        # Fail-closed budget gate: refuse before any provider call once the
        # workload's daily cap is reached (inert unless budget_fail_closed).
        # Runs AFTER pre_call so a short-circuit (e.g. a cache hit) — which
        # spends nothing — can still serve a capped workload; a real provider
        # call is still refused.
        if short_circuit is None:
            self._enforce_budget(wl)

        call_id = str(uuid.uuid4())
        ts = datetime.now(UTC)
        outcome = Outcome.OK
        error_kind: str | None = None
        error_detail: str | None = None
        tokens_in = tokens_out = latency_ms = 0
        actual_model = model or ""
        actual_provider = ""
        text = ""
        tool_calls_out: list[CoreToolCall] = []
        stop_reason_out: str = ""
        reasoning_content_out: str = ""
        raw_out: dict | None = None
        cost_usd_out: float | None = None

        # Track whether we took the fallback path so we can fire on_fallback
        # only on the narrow "pinned failed + chain saved us" window.
        fallback_info: dict | None = None
        raise_after_record: Exception | None = None

        if short_circuit is not None:
            text = short_circuit.text
            actual_provider = short_circuit.provider
            actual_model = short_circuit.model
            tokens_in = short_circuit.tokens_in
            tokens_out = short_circuit.tokens_out
            latency_ms = 0
            tool_calls_out = _to_core_tool_calls(short_circuit.tool_calls)
            raw_out = short_circuit.raw
            cost_usd_out = short_circuit.cost_usd if short_circuit.cost_usd is not None else 0.0
        elif provider is not None:
            chosen = self._pick_provider(provider)
            try:
                resp = chosen.generate(req)
                text = resp.text
                actual_provider = chosen.name
                actual_model = resp.model
                tokens_in = resp.tokens_in
                tokens_out = resp.tokens_out
                latency_ms = resp.latency_ms
                tool_calls_out = _to_core_tool_calls(resp.tool_calls)
                stop_reason_out = resp.stop_reason
                reasoning_content_out = resp.reasoning_content
                raw_out = resp.raw
                if not text.strip() and not tool_calls_out:
                    outcome = Outcome.EMPTY
                    error_kind = "EmptyResponse"
                    error_detail = _format_empty_detail(
                        provider=actual_provider,
                        model=actual_model,
                        tokens_out=tokens_out,
                        latency_ms=latency_ms,
                    )
            except Exception as exc:
                # Preferred provider failed — fall through to the full
                # router chain instead of giving up. "provider=X" means
                # "try X first", not "only X". This is critical for
                # parallel workers: if one provider goes down mid-batch,
                # those workers recover via fallthrough instead of
                # producing empty results.
                from somm.errors import SommFatalError
                if isinstance(exc, SommFatalError):
                    # Auth errors etc. — don't retry, but still try router
                    pass
                # Remember what the caller asked for before clearing the pin
                # — on_fallback needs to show pinned vs. actual.
                fallback_info = {
                    "pinned_provider": provider,
                    "pinned_model": model,
                    "error_kind": type(exc).__name__,
                    "error_detail": _format_error_detail(exc, chosen.name, model),
                }
                if no_fallback:
                    # Caller wants pinned-or-bust semantics — surface the
                    # original error and stop. Used by evaluation harnesses
                    # where a silent re-route to a different model would
                    # invalidate the experiment.
                    fallback_info = None
                    outcome = Outcome.UPSTREAM_ERROR
                    error_kind = type(exc).__name__
                    error_detail = _format_error_detail(exc, chosen.name, model)
                    actual_provider = chosen.name
                    actual_model = model or ""
                else:
                    # Clear the pinned model before chain fallback: the pin is
                    # only meaningful to the pinned provider. Other providers
                    # serve different model inventories, so re-using the pinned
                    # model name (e.g. "qwen3:14b" on Minimax) guarantees a 404.
                    req.model = None
                    try:
                        router_result = self.router.dispatch(req, wait=wait_on_exhausted)
                        resp = router_result.response
                        text = resp.text
                        actual_provider = router_result.provider
                        actual_model = resp.model
                        tokens_in = resp.tokens_in
                        tokens_out = resp.tokens_out
                        latency_ms = resp.latency_ms
                        tool_calls_out = _to_core_tool_calls(resp.tool_calls)
                        stop_reason_out = resp.stop_reason
                        reasoning_content_out = resp.reasoning_content
                        raw_out = resp.raw
                        if not text.strip() and not tool_calls_out:
                            outcome = Outcome.EMPTY
                            error_kind = "EmptyResponse"
                            error_detail = _format_empty_detail(
                                provider=actual_provider,
                                model=actual_model,
                                tokens_out=tokens_out,
                                latency_ms=latency_ms,
                            )
                    except Exception as chain_exc:
                        # Total failure — clear fallback_info so on_fallback
                        # doesn't fire; on_error handles final-failure signal.
                        fallback_info = None
                        outcome = Outcome.UPSTREAM_ERROR
                        error_kind = type(chain_exc).__name__
                        error_detail = _format_error_detail(chain_exc, chosen.name, model)
                        actual_provider = chosen.name
                        if hasattr(chain_exc, "model") and chain_exc.model:
                            actual_model = chain_exc.model
                        if (
                            wait_on_exhausted is not None
                            and isinstance(chain_exc, SommProvidersExhausted)
                        ):
                            raise_after_record = chain_exc
        else:
            dispatch_raise_exhausted = wait_on_exhausted is not None
            try:
                if wl.policy:
                    retry_cfg = wl.policy.get("retry") or {}
                    dispatch_raise_exhausted = retry_cfg.get("max") is not None
                    policy_wait = wait_on_exhausted
                    if wait is _WAIT_UNSET and retry_cfg.get("deadline_s") is not None:
                        policy_wait = float(retry_cfg["deadline_s"])
                    dispatch_raise_exhausted = (
                        dispatch_raise_exhausted or policy_wait is not None
                    )
                    router_result = self._policy_router(
                        wl.policy,
                        explicit_model=model is not None,
                    ).dispatch(
                        req,
                        wait=policy_wait,
                        retry_max=retry_cfg.get("max"),
                        retry_backoff_s=retry_cfg.get("backoff_s"),
                    )
                else:
                    router_result = self.router.dispatch(req, wait=wait_on_exhausted)
                resp = router_result.response
                text = resp.text
                actual_provider = router_result.provider
                actual_model = resp.model
                tokens_in = resp.tokens_in
                tokens_out = resp.tokens_out
                latency_ms = resp.latency_ms
                tool_calls_out = _to_core_tool_calls(resp.tool_calls)
                stop_reason_out = resp.stop_reason
                reasoning_content_out = resp.reasoning_content
                raw_out = resp.raw
                if not text.strip() and not tool_calls_out:
                    outcome = Outcome.EMPTY
                    error_kind = "EmptyResponse"
                    error_detail = _format_empty_detail(
                        provider=actual_provider,
                        model=actual_model,
                        tokens_out=tokens_out,
                        latency_ms=latency_ms,
                    )
            except Exception as exc:
                if hasattr(exc, "model") and exc.model:
                    actual_model = exc.model
                outcome = (
                    Outcome.EXHAUSTED
                    if type(exc).__name__ == "SommProvidersExhausted"
                    else Outcome.UPSTREAM_ERROR
                )
                error_kind = type(exc).__name__
                error_detail = _format_error_detail(exc, actual_provider, actual_model)
                if (
                    dispatch_raise_exhausted
                    and isinstance(exc, SommProvidersExhausted)
                ):
                    raise_after_record = exc

        cache_tokens_in, cache_tokens_out = extract_cache_tokens(raw_out)
        citations = extract_citations(raw_out)
        citations_json = _citations_json(citations)

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
                else cost_for_call(self.repo, actual_provider, actual_model, tokens_in, tokens_out)
            ),
            call_id=call_id,
            outcome=outcome,
            error_kind=error_kind,
            error_detail=error_detail,
            raw=raw_out,
            tool_calls=tool_calls_out,
            stop_reason=stop_reason_out,
            reasoning_content=reasoning_content_out,
            cache_tokens_in=cache_tokens_in,
            cache_tokens_out=cache_tokens_out,
            citations=citations,
        )

        call = Call(
            id=call_id,
            ts=ts,
            project=self.config.project,
            workload_id=wl.id,
            prompt_id=call_prompt_id,
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
            prompt_hash=stable_hash(messages if messages is not None else prompt_text),
            response_hash=stable_hash(text),
            correlation_id=(
                effective_correlation_id := correlation_id
                if correlation_id is not None
                else hooks.current_correlation_id()
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id,
            parent_call_id=parent_call_id,
            cache_tokens_in=cache_tokens_in,
            cache_tokens_out=cache_tokens_out,
            citations_json=citations_json,
        )
        self._writer.submit(call)
        self._maybe_capture_sample(wl, call_id, prompt_text, messages, text, outcome)
        _fire_call_hooks(_call_event(
            call_id=call_id, correlation_id=effective_correlation_id,
            project=self.config.project, workload=workload,
            provider=actual_provider, model=actual_model,
            outcome=outcome.value, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency_ms, cost_usd=result.cost_usd,
            temperature=temperature, max_tokens=max_tokens,
            error_kind=error_kind,
            short_circuited=short_circuited,
        ))

        # Fire the on_error alerter whenever the call did not succeed.
        if outcome != Outcome.OK and self._on_error is not None:
            try:
                self._on_error({
                    "call_id": call_id,
                    "workload": workload,
                    "provider": actual_provider,
                    "model": actual_model,
                    "outcome": outcome.value,
                    "error_kind": error_kind,
                    "error_detail": error_detail,
                })
            except Exception:
                # Alerter must not break the caller. Swallow and continue.
                pass

        # Fire on_fallback when a pinned call got rescued by the chain
        # AND ended up on a DIFFERENT (provider, model) than what was
        # pinned. Same-model recovery is just a retry — useful for
        # telemetry but not worth an on_fallback alert (it's not a
        # structural pin-vs-actual divergence). Keeping the signal
        # narrow prevents alert fatigue on free-tier providers that
        # intermittently 429 or return empty responses.
        same_provider_and_model = (
            fallback_info is not None
            and fallback_info["pinned_provider"] == actual_provider
            and (fallback_info["pinned_model"] or "") == (actual_model or "")
        )
        if (
            outcome == Outcome.OK
            and fallback_info is not None
            and not same_provider_and_model
            and self._on_fallback is not None
        ):
            try:
                self._on_fallback({
                    "call_id": call_id,
                    "workload": workload,
                    "pinned_provider": fallback_info["pinned_provider"],
                    "pinned_model": fallback_info["pinned_model"],
                    "actual_provider": actual_provider,
                    "actual_model": actual_model,
                    "error_kind": fallback_info["error_kind"],
                    "error_detail": fallback_info["error_detail"],
                })
            except Exception:
                # Hook must not break the caller.
                pass

        if raise_after_record is not None:
            raise raise_after_record

        # Feature 3: warn if daily budget cap is exceeded.
        if wl.budget_cap_usd_daily is not None and result.cost_usd > 0:
            today = date.today()
            budget_key = (wl.name, today)
            if budget_key not in _warned_budget_exceeded:
                with self.repo._open() as conn:
                    row = conn.execute(
                        "SELECT COALESCE(SUM(cost_usd), 0) FROM calls "
                        "WHERE workload_id = ? AND date(ts) = date('now')",
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

        # Loud-empty: when the caller explicitly set raise_on_empty=True
        # and we ended up with an empty response (every provider returned
        # blanks), raise rather than handing back empty text. This is the
        # 2026-05-06 lesson: a thinking model that consumed its whole token
        # budget on internal reasoning shouldn't silently return empty —
        # callers who care need a typed exception so they can fall back,
        # bump max_tokens, or pick a non-thinking model.
        if raise_on_empty and outcome == Outcome.EMPTY:
            from somm.errors import SommEmptyResponse
            raise SommEmptyResponse(
                f"empty response from {actual_provider}/{actual_model} "
                f"(workload={workload!r}); set raise_on_empty=False to "
                f"swallow, or bump max_tokens / pick a non-thinking model "
                f"if a thinking model burned the whole budget."
            )

        return result

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
            correlation_id=(correlation_id := hooks.current_correlation_id()),
            session_id=session_id,
            parent_call_id=parent_call_id,
        )
        self._writer.submit(call)
        _fire_call_hooks(_call_event(
            call_id=call_id, correlation_id=correlation_id,
            project=self.config.project, workload=workload,
            provider=result.provider, model=actual_model,
            outcome=outcome.value, tokens_in=tokens_in, tokens_out=0,
            latency_ms=latency_ms, cost_usd=result.cost_usd,
            error_kind=error_kind,
            short_circuited=short_circuited,
        ))

        if outcome != Outcome.OK and self._on_error is not None:
            with contextlib.suppress(Exception):
                self._on_error({
                    "call_id": call_id,
                    "workload": workload,
                    "provider": "ollama",
                    "model": actual_model,
                    "outcome": outcome.value,
                    "error_kind": error_kind,
                    "error_detail": error_detail,
                })

        return result

    # ------------------------------------------------------------------

    def _pick_provider(self, name: str | None) -> SommProvider:
        if name:
            for p in self.providers:
                if p.name == name:
                    return p
            raise ValueError(f"provider {name!r} not configured")
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
    ) -> Iterator[str]:
        """Stream text deltas from the LLM. Yields user-visible text chunks
        (with `<think>` blocks stripped across chunk boundaries).

        Telemetry is written after the stream completes. No mid-stream
        router fallback in v0.1 — first non-cooled provider handles the
        whole stream or errors out.
        ``prompt`` may be a string, multimodal prompt list, or registered
        ``Prompt`` object. Passing ``Prompt`` dispatches its body and stamps
        that prompt version on the telemetry row.
        Omit ``wait`` to use ``SOMM_WAIT_ON_EXHAUSTED`` when set; pass
        ``None`` or ``0`` to fail fast when all stream candidates are cooling.

        Usage:
            for piece in llm.stream("tell a story", workload="story"):
                print(piece, end="", flush=True)
        """
        from somm.providers.base import SommRequest

        wl = self._require_workload(workload)
        wait_on_exhausted = self._resolve_wait_on_exhausted(wait)
        prompt_text = prompt.body if isinstance(prompt, Prompt) else prompt
        messages = None
        call_prompt_id = self._resolve_prompt_id(wl.id, prompt)
        req = SommRequest(
            prompt=prompt_text,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )

        short_circuit: hooks.ShortCircuit | None = None
        short_circuited: str | None = None
        if hooks.has_hooks(hooks.PRE_CALL):
            original_prompt_text = prompt_text
            ctx = hooks.PreCallContext(
                workload=workload,
                prompt=prompt_text,
                system=system,
                messages=None,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=[],
                tool_choice=None,
                project=self.config.project,
            )
            short_circuit = hooks.fire_pre_call(ctx)
            prompt_text = ctx.prompt
            system = ctx.system
            messages = ctx.messages
            model = ctx.model
            provider = ctx.provider
            max_tokens = ctx.max_tokens
            temperature = ctx.temperature
            if messages is not None:
                call_prompt_id = None
            elif isinstance(prompt, Prompt) and prompt_text == original_prompt_text:
                call_prompt_id = prompt.id
            else:
                call_prompt_id = self._resolve_prompt_id(wl.id, prompt_text)
            req = SommRequest(
                prompt=prompt_text,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                tools=ctx.tools or [],
                messages=messages,
                tool_choice=ctx.tool_choice,
            )
            if short_circuit is not None:
                short_circuited = short_circuit.source or short_circuit.provider or "hook"

        # Budget gate after pre_call — a short-circuit spends nothing, so a
        # capped workload may still stream from cache; a real call is refused.
        if short_circuit is None:
            self._enforce_budget(wl)

        chosen = None
        if short_circuit is None and wait_on_exhausted is not None:
            candidates = [self._pick_provider(provider)] if provider else list(self.providers)
            self.router.wait_until_any_uncool(candidates, wait_on_exhausted)
        if short_circuit is None:
            chosen = self._pick_stream_provider(provider)
        call_id = str(uuid.uuid4())
        ts = datetime.now(UTC)
        stripper = ThinkStreamStripper()

        t0 = time.monotonic()
        collected = []
        outcome = Outcome.OK
        error_kind: str | None = None
        error_detail: str | None = None
        tokens_in = tokens_out = 0
        actual_model = model or ""
        actual_provider = short_circuit.provider if short_circuit is not None else ""
        cost_usd_out: float | None = None
        ttft_ms: int | None = None
        raw_out: dict | None = None

        try:
            if short_circuit is not None:
                text = short_circuit.text
                actual_model = short_circuit.model
                tokens_in = short_circuit.tokens_in
                tokens_out = short_circuit.tokens_out
                cost_usd_out = short_circuit.cost_usd if short_circuit.cost_usd is not None else 0.0
                raw_out = short_circuit.raw
                if text:
                    collected.append(text)
                    ttft_ms = int((time.monotonic() - t0) * 1000)
                    yield text
            else:
                assert chosen is not None
                for chunk in chosen.stream(req):
                    chunk_raw = getattr(chunk, "raw", None)
                    if isinstance(chunk_raw, dict):
                        raw_out = chunk_raw
                    if chunk.text:
                        visible = stripper.feed(chunk.text)
                        if visible:
                            collected.append(visible)
                            if ttft_ms is None:
                                ttft_ms = int((time.monotonic() - t0) * 1000)
                            yield visible
                    if chunk.done:
                        tail = stripper.flush()
                        if tail:
                            collected.append(tail)
                            if ttft_ms is None:
                                ttft_ms = int((time.monotonic() - t0) * 1000)
                            yield tail
                        break
                # Streaming providers don't always give token counts — estimate
                # from length as a fallback.
                text = "".join(collected)
                if not actual_model:
                    actual_model = chosen.name
                actual_provider = chosen.name
                tokens_in = chosen.estimate_tokens(prompt_text, actual_model) + chosen.estimate_tokens(
                    system, actual_model
                )
                tokens_out = chosen.estimate_tokens(text, actual_model)
                if not text.strip():
                    outcome = Outcome.EMPTY
                    error_kind = "EmptyResponse"
                    # Stream latency is computed in `finally`; pass what we have
                    # so far (since we're between `try` and `finally`, t0 is set).
                    error_detail = _format_empty_detail(
                        provider=chosen.name,
                        model=actual_model or chosen.name,
                        tokens_out=tokens_out,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
        except Exception as exc:
            outcome = Outcome.UPSTREAM_ERROR
            error_kind = type(exc).__name__
            text = "".join(collected)
            raise
        finally:
            latency_ms = int((time.monotonic() - t0) * 1000)
            full_text = "".join(collected)
            provider_name = actual_provider or (chosen.name if chosen is not None else "hook")
            model_name = actual_model or provider_name
            cache_tokens_in, cache_tokens_out = extract_cache_tokens(raw_out)
            citations = extract_citations(raw_out)
            citations_json = _citations_json(citations)
            cost_usd = (
                cost_usd_out
                if cost_usd_out is not None
                else cost_for_call(
                    self.repo,
                    provider_name,
                    model_name,
                    tokens_in,
                    tokens_out,
                )
            )
            call = Call(
                id=call_id,
                ts=ts,
                project=self.config.project,
                workload_id=wl.id,
                prompt_id=call_prompt_id,
                provider=provider_name,
                model=model_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                outcome=outcome,
                error_kind=error_kind,
                error_detail=error_detail,
                prompt_hash=stable_hash(messages if messages is not None else prompt_text),
                response_hash=stable_hash(full_text),
                correlation_id=(correlation_id := hooks.current_correlation_id()),
                temperature=temperature,
                max_tokens=max_tokens,
                ttft_ms=ttft_ms,
                session_id=session_id,
                parent_call_id=parent_call_id,
                cache_tokens_in=cache_tokens_in,
                cache_tokens_out=cache_tokens_out,
                citations_json=citations_json,
            )
            self._writer.submit(call)
            _fire_call_hooks(_call_event(
                call_id=call_id, correlation_id=correlation_id,
                project=self.config.project, workload=workload,
                provider=provider_name, model=model_name,
                outcome=outcome.value, tokens_in=tokens_in,
                tokens_out=tokens_out, latency_ms=latency_ms,
                cost_usd=call.cost_usd,
                temperature=temperature, max_tokens=max_tokens,
                error_kind=error_kind,
                short_circuited=short_circuited,
            ))

    def _pick_stream_provider(self, name: str | None):
        if name:
            return self._pick_provider(name)
        # First non-cooled provider for streams (no mid-stream fallback).
        for p in self.providers:
            if not self._tracker.get(p.name).is_cooling():
                return p
        # If all cooled, use the first and let it raise.
        return self.providers[0]

    # ------------------------------------------------------------------
    # Structured output

    def generate_structured(
        self,
        prompt,
        *,
        schema,
        workload: str = "default",
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
        model: str | None = None,
        provider: str | None = None,
        retries: int = 2,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        wait: float | None = _WAIT_UNSET,
    ) -> tuple[Any, SommResult]:
        """Generate JSON, validate it, and return (validated_object, result).

        Unlike extract_structured(), this is strict: it retries with corrective
        feedback and raises SommStructuredError instead of returning a sentinel
        dict when the model never produces a valid object.
        """
        schema_kind, schema_doc = _classify_structured_schema(schema)
        base_system = _structured_system(system, schema_doc)
        response_format = _json_schema_response_format(schema_doc)
        last_raw = ""
        last_error = "no response"
        last_result: SommResult | None = None

        for attempt in range(retries + 1):
            attempt_system = base_system
            if attempt > 0:
                attempt_system = _structured_retry_system(
                    base_system,
                    last_raw,
                    last_error,
                )
            result = self.generate(
                prompt=prompt,
                system=attempt_system,
                workload=workload,
                max_tokens=max_tokens,
                temperature=temperature + (attempt * _STRUCTURED_TEMPERATURE_JITTER),
                model=model,
                provider=provider,
                response_format=response_format,
                wait=wait,
                session_id=session_id,
                parent_call_id=parent_call_id,
            )
            last_result = result
            last_raw = result.text

            parsed = extract_json(result.text)
            if parsed is None:
                last_error = "response did not contain a parseable JSON object or array"
                result.mark(Outcome.BAD_JSON)
                continue

            try:
                if schema_kind == "pydantic":
                    validated = schema.model_validate(parsed)
                elif schema_kind == "json_schema":
                    validated = _validate_structured_json_schema(parsed, schema)
                else:
                    validated = schema(parsed)
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                result.mark(Outcome.BAD_JSON)
                continue

            return validated, result

        if last_result is not None:
            last_result.mark(Outcome.BAD_JSON)
        raise SommStructuredError(_format_structured_failure(last_raw, last_error))

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
        wait: float | None = _WAIT_UNSET,
    ) -> dict | list:
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
        """
        last_text = ""
        for attempt in range(retries + 1):
            temp = temperature + (attempt * temperature_jitter)
            result = self.generate(
                prompt=prompt,
                system=system,
                workload=workload,
                max_tokens=max_tokens,
                temperature=temp,
                model=model,
                provider=provider,
                wait=wait,
            )
            last_text = result.text
            parsed = extract_json(result.text)
            if parsed is not None:
                return parsed
            result.mark(Outcome.BAD_JSON)
        return {"raw": last_text, "_somm_parse_err": True}

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

                raise PromptNotFound(
                    f"no prompt label {label!r} for workload {wl.id!r}"
                )
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
                f"Docs: docs/errors/SOMM_PRIVACY_VIOLATION.md"
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

    def stream(self, request):  # pragma: no cover - policy routing is generate-only
        return self._provider.stream(request)

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
                "capabilities_required_json, policy_json FROM workloads"
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
                     capabilities_required_json, policy_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    except Exception:  # noqa: BLE001 — best-effort mirror
        pass


def _warn_mirror_unavailable(path: Path, exc: Exception) -> None:
    key = str(Path(path).resolve())
    if key in _warned_mirror_unavailable:
        return
    _warned_mirror_unavailable.add(key)
    print(f"[somm] global mirror unavailable at {key}: {exc}; continuing without it", file=sys.stderr)

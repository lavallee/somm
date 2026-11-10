"""OpenTelemetry span exporter for completed somm calls."""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
from typing import Any

from somm import hooks

_registered = False
_tracer: Any = None
_trace_api: Any = None
_status_cls: Any = None
_status_code_cls: Any = None
_owned_tracer_provider: Any = None
_lifecycle_lock = threading.RLock()


def _load_trace_api() -> tuple[Any, Any, Any]:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
        from opentelemetry.trace import Status, StatusCode
    except ImportError as exc:
        raise ImportError("OpenTelemetry support requires: pip install somm[otel]") from exc
    return trace, Status, StatusCode


def _load_otlp_components() -> tuple[Any, Any, Any]:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ImportError("OTLP HTTP export requires: pip install somm[otel]") from exc
    return TracerProvider, OTLPSpanExporter, BatchSpanProcessor


def _flush_and_shutdown(provider: Any) -> None:
    with contextlib.suppress(Exception):
        provider.force_flush()
    with contextlib.suppress(Exception):
        provider.shutdown()


# The GenAI semantic conventions this exporter follows. They are still marked
# Development upstream and publish no schema URL yet, so the pin is the
# repository commit that was read, plus the core semconv release it builds on
# (for ``error.type``). Changing an emitted ``gen_ai.*`` name is a deliberate
# edit to this constant, the docs and the allow-list in the tests.
GENAI_SEMCONV_REPO = "https://github.com/open-telemetry/semantic-conventions-genai"
GENAI_SEMCONV_COMMIT = "e6658f14c9d9ea0760825f029cb82e7673f48fab"  # 2026-11-06
CORE_SEMCONV_VERSION = "1.45.0"

# somm provider name -> the spec's well-known gen_ai.provider.name value.
# Anything else is passed through unchanged (the spec allows custom values).
_PROVIDER_NAMES = {"gemini": "gcp.gemini", "claude-cli": "anthropic", "codex-cli": "openai"}

_NS_PER_MS = 1_000_000


def _provider_name(provider: str) -> str:
    return _PROVIDER_NAMES.get(provider, provider)


def _emit(
    name: str,
    attrs: dict[str, Any],
    *,
    start_ns: int,
    end_ns: int,
    failed: bool,
) -> None:
    kwargs: dict[str, Any] = {"start_time": start_ns}
    kind = getattr(getattr(_trace_api, "SpanKind", None), "CLIENT", None)
    if kind is not None:
        kwargs["kind"] = kind
    span = _tracer.start_span(name, **kwargs)
    try:
        for key, value in attrs.items():
            span.set_attribute(key, value)
        if failed:
            span.set_status(_status_cls(_status_code_cls.ERROR))
    finally:
        span.end(end_time=end_ns)


def _times(emitted_ns: Any, duration_ms: Any) -> tuple[int, int]:
    """(start, end) in epoch ns. The end is when the call finished (stamped by
    the hook bus on the caller's thread), not when this worker got to it."""
    end = int(emitted_ns) if isinstance(emitted_ns, int) and emitted_ns > 0 else time.time_ns()
    try:
        duration = max(0, int(duration_ms or 0)) * _NS_PER_MS
    except (TypeError, ValueError):
        duration = 0
    return end - duration, end


def _post_process(event: dict[str, Any]) -> None:
    try:
        workload = event.get("workload") or "default"
        provider = event.get("provider") or ""
        model = event.get("model") or ""
        outcome = str(event.get("outcome") or "")
        failed = outcome.lower() != "ok"
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": _provider_name(provider),
            "gen_ai.usage.input_tokens": int(event.get("tokens_in") or 0),
            "gen_ai.usage.output_tokens": int(event.get("tokens_out") or 0),
            "somm.call_id": event.get("call_id") or "",
            "somm.outcome": outcome,
            "somm.cost_usd": float(event.get("cost_usd") or 0.0),
            "somm.workload": workload,
            "somm.project": event.get("project") or "",
            "somm.cost_basis": event.get("cost_basis") or "unknown",
            "somm.cost_kind": event.get("cost_kind") or "unknown",
            "somm.cost_accuracy": event.get("cost_accuracy") or "unknown",
            "somm.cost_source": event.get("cost_source") or "",
            "somm.pricing_version": event.get("pricing_version") or "",
            "somm.observation_role": event.get("observation_role") or "production",
            "somm.source_call_id": event.get("source_call_id") or "",
            "somm.eval_result_id": int(event.get("eval_result_id") or 0),
            "somm.provider_request_id": event.get("provider_request_id") or "",
            "somm.billing_id": event.get("billing_id") or "",
            "somm.origin": event.get("origin") or "native",
            "somm.budget_eligible": bool(event.get("budget_eligible", True)),
        }
        if model:
            attrs["gen_ai.request.model"] = model
            attrs["gen_ai.response.model"] = model
        if event.get("provider_request_id"):
            attrs["gen_ai.response.id"] = str(event["provider_request_id"])
        if event.get("correlation_id"):
            attrs["gen_ai.conversation.id"] = str(event["correlation_id"])
        if isinstance(event.get("max_tokens"), int):
            attrs["gen_ai.request.max_tokens"] = event["max_tokens"]
        if isinstance(event.get("temperature"), (int, float)):
            attrs["gen_ai.request.temperature"] = float(event["temperature"])
        if failed:
            attrs["error.type"] = str(event.get("error_kind") or outcome or "_OTHER")
        for name, key in (
            ("gen_ai.usage.cache_read.input_tokens", "cache_tokens_in"),
            ("gen_ai.usage.cache_write.input_tokens", "cache_tokens_out"),
            ("gen_ai.usage.reasoning.output_tokens", "reasoning_tokens"),
        ):
            value = event.get(key)
            if isinstance(value, int) and value > 0:
                attrs[name] = value
        start_ns, end_ns = _times(event.get("emitted_at_ns"), event.get("latency_ms"))
        span_name = f"chat {model}" if model else "chat"
        _emit(span_name, attrs, start_ns=start_ns, end_ns=end_ns, failed=failed)
    except Exception:
        return None
    return None


def export_harness_attempt(call: Any, result: Any) -> None:
    """Emit one ``invoke_agent`` span for a recorded harness attempt.

    Called by :func:`somm.harnesses.ledger.record_attempt` after the ledger
    write, on the caller's thread, so the span nests under the caller's active
    span. A no-op unless the exporter is registered; never raises. Normalized
    harness results carry no per-tool events, so there are no ``execute_tool``
    child spans.
    """
    if _tracer is None:
        return None
    try:
        outcome = str(getattr(call.outcome, "value", call.outcome))
        failed = outcome.lower() != "ok"
        harness = str(result.harness)
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": _provider_name(harness),
            "gen_ai.agent.name": harness,
            "gen_ai.usage.input_tokens": int(call.tokens_in or 0),
            "gen_ai.usage.output_tokens": int(call.tokens_out or 0),
            "somm.call_id": call.id,
            "somm.outcome": outcome,
            "somm.harness_outcome": str(getattr(result.outcome, "value", result.outcome)),
            "somm.cost_usd": float(call.cost_usd or 0.0),
            "somm.project": call.project or "",
            "somm.origin": "harness",
        }
        if call.model:
            attrs["gen_ai.request.model"] = call.model
        conversation = result.session_id or call.correlation_id
        if conversation:
            attrs["gen_ai.conversation.id"] = str(conversation)
        if failed:
            attrs["error.type"] = str(call.error_kind or outcome or "_OTHER")
        for name, value in (
            ("gen_ai.usage.cache_read.input_tokens", call.cache_tokens_in),
            ("gen_ai.usage.cache_write.input_tokens", call.cache_tokens_out),
            ("gen_ai.usage.reasoning.output_tokens", call.reasoning_tokens),
        ):
            if isinstance(value, int) and value > 0:
                attrs[name] = value
        # The ledger stamps ``ts`` with the attempt's start.
        start_ns = int(call.ts.timestamp() * 1_000_000_000)
        end_ns = start_ns + max(0, int(call.latency_ms or 0)) * _NS_PER_MS
        _emit(f"invoke_agent {harness}", attrs, start_ns=start_ns, end_ns=end_ns, failed=failed)
    except Exception:
        return None
    return None


def register(tracer_provider=None) -> None:
    """Install the OpenTelemetry post-process hook.

    Raises:
        ImportError: when OpenTelemetry is not installed. Install with
            ``pip install somm[otel]``.
    """
    global _registered, _tracer, _status_cls, _status_code_cls
    global _owned_tracer_provider, _trace_api

    trace, status_cls, status_code_cls = _load_trace_api()
    trace_module = trace
    provider = tracer_provider
    tracer = (
        provider.get_tracer("somm.plugins.otel_exporter")
        if provider is not None
        else trace.get_tracer("somm.plugins.otel_exporter")
    )

    owned_provider = None
    with _lifecycle_lock:
        if _owned_tracer_provider is not None and provider is not _owned_tracer_provider:
            owned_provider = _owned_tracer_provider
            _owned_tracer_provider = None
        _status_cls = status_cls
        _status_code_cls = status_code_cls
        _tracer = tracer
        _trace_api = trace_module
        if not _registered:
            hooks.register_hook(hooks.POST_PROCESS, _post_process)
            _registered = True

    if owned_provider is not None:
        _flush_and_shutdown(owned_provider)


def register_from_env() -> None:
    """Configure OTLP HTTP/protobuf span export from ``SOMM_OTEL_ENDPOINT``.

    ``SOMM_OTEL_ENDPOINT`` is the full traces endpoint. When it is absent or
    empty, this function is a no-op and does not import the optional
    OpenTelemetry dependencies, register a hook, or start a batch processor.
    """
    global _registered, _tracer, _status_cls, _status_code_cls
    global _owned_tracer_provider, _trace_api

    endpoint = os.environ.get("SOMM_OTEL_ENDPOINT")
    if not endpoint:
        return

    with _lifecycle_lock:
        if _registered:
            return

        trace_module, status_cls, status_code_cls = _load_trace_api()
        tracer_provider_cls, exporter_cls, processor_cls = _load_otlp_components()
        provider = None
        exporter = None
        processor = None
        processor_attached = False
        try:
            provider = tracer_provider_cls(shutdown_on_exit=False)
            exporter = exporter_cls(endpoint=endpoint)
            processor = processor_cls(exporter)
            provider.add_span_processor(processor)
            processor_attached = True
            tracer = provider.get_tracer("somm.plugins.otel_exporter")
            hooks.register_hook(hooks.POST_PROCESS, _post_process)
        except Exception:
            if provider is not None:
                _flush_and_shutdown(provider)
            if processor is not None and not processor_attached:
                with contextlib.suppress(Exception):
                    processor.shutdown()
            elif exporter is not None and processor is None:
                with contextlib.suppress(Exception):
                    exporter.shutdown()
            raise

        _status_cls = status_cls
        _status_code_cls = status_code_cls
        _tracer = tracer
        _trace_api = trace_module
        _owned_tracer_provider = provider
        _registered = True


def unregister() -> None:
    """Remove the hook and close only a plugin-owned tracer provider."""
    global _registered, _tracer, _status_cls, _status_code_cls
    global _owned_tracer_provider, _trace_api

    with _lifecycle_lock:
        hooks.unregister_hook(hooks.POST_PROCESS, _post_process)
        provider = _owned_tracer_provider
        _owned_tracer_provider = None
        _registered = False
        _tracer = None
        _trace_api = None
        _status_cls = None
        _status_code_cls = None

    if provider is not None:
        _flush_and_shutdown(provider)


def _shutdown_at_exit() -> None:
    if _owned_tracer_provider is not None:
        hooks.shutdown_hooks(wait=True)
    unregister()


atexit.register(_shutdown_at_exit)

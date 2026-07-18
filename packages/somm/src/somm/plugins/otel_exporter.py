"""OpenTelemetry span exporter for completed somm calls."""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
from typing import Any

from somm import hooks

_registered = False
_tracer: Any = None
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


def _post_process(event: dict[str, Any]) -> None:
    try:
        workload = event.get("workload") or "default"
        with _tracer.start_as_current_span(f"llm {workload}") as span:
            provider = event.get("provider") or ""
            model = event.get("model") or ""
            attrs = {
                "gen_ai.system": provider,
                "gen_ai.request.model": model,
                "gen_ai.response.model": model,
                "gen_ai.usage.input_tokens": int(event.get("tokens_in") or 0),
                "gen_ai.usage.output_tokens": int(event.get("tokens_out") or 0),
                "somm.call_id": event.get("call_id") or "",
                "somm.outcome": event.get("outcome") or "",
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
            for key, value in attrs.items():
                span.set_attribute(key, value)
            if str(event.get("outcome") or "").lower() != "ok":
                span.set_status(_status_cls(_status_code_cls.ERROR))
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
    global _owned_tracer_provider

    trace, status_cls, status_code_cls = _load_trace_api()
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
    global _owned_tracer_provider

    endpoint = os.environ.get("SOMM_OTEL_ENDPOINT")
    if not endpoint:
        return

    with _lifecycle_lock:
        if _registered:
            return

        _trace, status_cls, status_code_cls = _load_trace_api()
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
        _owned_tracer_provider = provider
        _registered = True


def unregister() -> None:
    """Remove the hook and close only a plugin-owned tracer provider."""
    global _registered, _tracer, _status_cls, _status_code_cls
    global _owned_tracer_provider

    with _lifecycle_lock:
        hooks.unregister_hook(hooks.POST_PROCESS, _post_process)
        provider = _owned_tracer_provider
        _owned_tracer_provider = None
        _registered = False
        _tracer = None
        _status_cls = None
        _status_code_cls = None

    if provider is not None:
        _flush_and_shutdown(provider)


def _shutdown_at_exit() -> None:
    if _owned_tracer_provider is not None:
        hooks.shutdown_hooks(wait=True)
    unregister()


atexit.register(_shutdown_at_exit)

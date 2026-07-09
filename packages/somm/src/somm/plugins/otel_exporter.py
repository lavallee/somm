"""OpenTelemetry span exporter for completed somm calls."""

from __future__ import annotations

from typing import Any

from somm import hooks

_registered = False
_tracer: Any = None
_status_cls: Any = None
_status_code_cls: Any = None


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
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
        from opentelemetry.trace import Status, StatusCode
    except ImportError as exc:
        raise ImportError("OpenTelemetry support requires: pip install somm[otel]") from exc

    _status_cls = Status
    _status_code_cls = StatusCode
    provider = tracer_provider
    _tracer = (
        provider.get_tracer("somm.plugins.otel_exporter")
        if provider is not None
        else trace.get_tracer("somm.plugins.otel_exporter")
    )
    if not _registered:
        hooks.register_hook(hooks.POST_PROCESS, _post_process)
        _registered = True


def unregister() -> None:
    """Remove the OpenTelemetry hook."""
    global _registered
    hooks.unregister_hook(hooks.POST_PROCESS, _post_process)
    _registered = False

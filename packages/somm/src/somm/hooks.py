"""Extension hooks — let external tools observe somm without coupling.

somm ships no integrations with other tools. Instead it exposes two small
hook points any package can attach to at runtime:

- A **correlation-id provider**: a zero-arg callable returning the caller's
  ambient correlation id (a request id, trace id, job id, …) or ``None``.
  The value is stamped on every telemetry row (``calls.correlation_id``),
  letting an external system join somm telemetry to its own records
  without a shared schema or a cross-database join.

- **Call observers**: callables invoked with an event dict after every
  completed call — generate, stream, or embed; success or failure. Useful
  for mirroring LLM activity into an external audit log or event bus.

Event dict keys (all always present): ``call_id``, ``correlation_id``,
``project``, ``workload``, ``provider``, ``model``, ``outcome`` (str),
``tokens_in``, ``tokens_out``, ``latency_ms``, ``cost_usd``,
``temperature``, ``max_tokens``, ``error_kind``.

Registration is either explicit::

    import somm.hooks
    somm.hooks.set_correlation_provider(my_request_id_getter)
    somm.hooks.add_call_observer(my_audit_logger)

or automatic via the ``somm.hooks`` entry-point group: each entry point
resolves to a zero-arg callable, invoked once on first client
construction. A package that wants to attach without any code change in
the host project declares::

    [project.entry-points."somm.hooks"]
    mytool = "mytool.somm_integration:register"

Hook failures are swallowed: extensions must never break the call path.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

CorrelationProvider = Callable[[], "str | None"]
CallObserver = Callable[[dict[str, Any]], None]

_correlation_provider: CorrelationProvider | None = None
_call_observers: list[CallObserver] = []
_entry_points_loaded = False


def set_correlation_provider(fn: CorrelationProvider | None) -> None:
    """Set (or clear, with None) the process-wide correlation-id provider."""
    global _correlation_provider
    _correlation_provider = fn


def add_call_observer(fn: CallObserver) -> None:
    """Register an observer invoked after every completed call."""
    if fn not in _call_observers:
        _call_observers.append(fn)


def remove_call_observer(fn: CallObserver) -> None:
    """Unregister a previously added observer. No-op if absent."""
    with contextlib.suppress(ValueError):
        _call_observers.remove(fn)


def current_correlation_id() -> str | None:
    """Read the ambient correlation id. Never raises."""
    if _correlation_provider is None:
        return None
    try:
        return _correlation_provider()
    except Exception:
        return None


def notify_call_observers(event: dict[str, Any]) -> None:
    """Fan an event out to all observers. A failing observer never
    breaks the call path or starves later observers."""
    for fn in list(_call_observers):
        with contextlib.suppress(Exception):
            fn(event)


def load_entry_points() -> None:
    """Invoke every ``somm.hooks`` entry point once per process.

    Idempotent; called on first SommLLM construction. A broken entry
    point is skipped silently — same never-break rule as observers.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="somm.hooks"):
            with contextlib.suppress(Exception):
                ep.load()()
    except Exception:
        pass

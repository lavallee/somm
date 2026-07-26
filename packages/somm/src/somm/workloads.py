"""Workload handlers — a workload is a unit of work, not necessarily an LLM call.

somm's unit has always been the **workload**: a named contract with schemas,
budgets, quality criteria, and SLOs, against which every call is metered. What
serves a workload has always been implicit — the provider chain. This module
makes it explicit and pluggable.

A workload has a **kind**. The default kind is ``llm``: the provider chain
answers it, exactly as before. A project may register other kinds and bind
workloads to them::

    import somm
    from somm import workloads

    workloads.register(MyHandler())              # once, at startup
    workloads.bind("triage", "my-kind", …)       # per workload

somm knows nothing about any particular kind, and takes no dependency on one.
Handlers arrive either by explicit ``register`` or through the
``somm.workload_handlers`` entry-point group, which is loaded on demand.

**Declining is normal.** A handler returns ``None`` to mean "not mine, or not
this time", and the call proceeds down the provider chain exactly as it would
have. That is what makes a handler safe to bind to a live workload: the worst
case is the behavior you already had.

Why this matters beyond dispatch: a workload served by a handler and the same
workload served by a model land in the same ``calls`` table under the same
workload id, distinguished by provider/source. So "should this workload still be
an LLM call?" becomes a query over recorded cost, latency, and outcome instead of
an argument — which is the evidence a workload needs to move to the right rung.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from somm import hooks

_logger = logging.getLogger("somm.workloads")

#: The implicit kind of every workload that is not bound to something else.
LLM = "llm"

#: Entry-point group through which a package publishes a workload handler.
ENTRY_POINT_GROUP = "somm.workload_handlers"

_lock = threading.RLock()
_handlers: dict[str, WorkloadHandler] = {}
_bindings: dict[str, _Binding] = {}
_hook_installed = False
_entry_points_loaded = False


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    """What a handler is asked to serve.

    A read-only view of the outbound call. A handler that wants to *modify* the
    request rather than serve it wants a ``pre_call`` hook, not a handler.
    """

    workload: str
    project: str
    prompt: str | list[Any] | None = None
    system: str = ""
    messages: list[Any] | None = None
    model: str | None = None
    provider: str | None = None
    response_format: dict[str, Any] | None = None
    #: Whatever was passed to :func:`bind` for this workload.
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """The outbound request as one string — messages if present, else the prompt."""
        if self.messages:
            import json

            return json.dumps(self.messages, sort_keys=True, default=str)
        body = getattr(self.prompt, "body", self.prompt)
        return "" if body is None else str(body)


@dataclass(frozen=True, slots=True)
class WorkloadResult:
    """What a handler returns when it serves a workload."""

    text: str
    model: str = ""
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    raw: dict[str, Any] | None = None
    tool_calls: list[Any] | None = None


@runtime_checkable
class WorkloadHandler(Protocol):
    """Serves workloads of one kind.

    ``kind`` is the name a workload binds to. ``serve`` returns a
    :class:`WorkloadResult`, or ``None`` to decline and let the provider chain
    answer. A handler that raises is treated as one that declined: an extension
    must never break the call path.
    """

    kind: str

    def serve(self, request: WorkloadRequest) -> WorkloadResult | None: ...


@dataclass(frozen=True, slots=True)
class _Binding:
    kind: str
    config: dict[str, Any]
    enabled: bool = True


# -- registry ---------------------------------------------------------------


def register(handler: WorkloadHandler, *, replace: bool = False) -> None:
    """Register a handler for its ``kind``.

    Raises :class:`ValueError` on a duplicate kind unless ``replace`` is set —
    two packages silently fighting over one kind is the kind of ambiguity that
    is much easier to prevent than to debug.
    """
    kind = getattr(handler, "kind", "")
    if not kind or not isinstance(kind, str):
        raise ValueError("a workload handler must declare a non-empty string 'kind'")
    if kind == LLM:
        raise ValueError(f"{LLM!r} is the built-in kind and cannot be overridden")
    if not callable(getattr(handler, "serve", None)):
        raise ValueError(f"handler for kind {kind!r} has no callable 'serve'")
    with _lock:
        if kind in _handlers and not replace:
            raise ValueError(
                f"a handler for kind {kind!r} is already registered; pass replace=True "
                "if that is intended"
            )
        _handlers[kind] = handler
    _install_hook()


def unregister(kind: str) -> None:
    """Remove a handler. Workloads bound to it fall through to the provider chain."""
    with _lock:
        _handlers.pop(kind, None)


def handlers() -> dict[str, WorkloadHandler]:
    """Every registered handler, by kind."""
    _load_entry_points()
    with _lock:
        return dict(_handlers)


def load_entry_point_handlers() -> dict[str, WorkloadHandler]:
    """Load handlers published through the ``somm.workload_handlers`` group.

    Each entry point must resolve to a zero-argument callable that registers a
    handler (the same shape as somm's plugin entry points).
    """
    _load_entry_points(force=True)
    return handlers()


# -- bindings ---------------------------------------------------------------


def bind(workload: str, kind: str, *, enabled: bool = True, **config: Any) -> None:
    """Bind a workload to a handler kind.

    ``config`` is handed back to the handler on every request, so routing detail
    (which installation, which budget) lives in one place and the handler stays
    stateless about it. Binding a workload to a kind whose handler is not
    registered is allowed and inert — it starts serving if and when the handler
    appears.
    """
    if kind == LLM:
        unbind(workload)
        return
    with _lock:
        _bindings[workload] = _Binding(kind=kind, config=dict(config), enabled=enabled)
    _install_hook()


def unbind(workload: str) -> None:
    """Return a workload to the provider chain."""
    with _lock:
        _bindings.pop(workload, None)


def bindings() -> dict[str, dict[str, Any]]:
    """Every bound workload, as plain dicts."""
    with _lock:
        return {
            name: {"kind": b.kind, "enabled": b.enabled, "config": dict(b.config)}
            for name, b in _bindings.items()
        }


def kind_of(workload: str) -> str:
    """The kind serving ``workload`` — :data:`LLM` when it is not bound."""
    with _lock:
        binding = _bindings.get(workload)
    return binding.kind if binding and binding.enabled else LLM


def clear() -> None:
    """Drop every handler and binding (test hygiene)."""
    global _entry_points_loaded
    with _lock:
        _handlers.clear()
        _bindings.clear()
        _entry_points_loaded = False


# -- dispatch ---------------------------------------------------------------


def resolve(request: WorkloadRequest) -> WorkloadResult | None:
    """Serve ``request`` through its bound handler, or return None.

    None means the provider chain answers — because the workload is unbound, its
    binding is disabled, no handler is registered for its kind, the handler
    declined, or the handler raised.
    """
    _load_entry_points()
    with _lock:
        binding = _bindings.get(request.workload)
        if binding is None or not binding.enabled:
            return None
        handler = _handlers.get(binding.kind)
    if handler is None:
        return None
    try:
        result = handler.serve(request)
    except Exception:
        _logger.warning(
            "workload handler %r raised serving %r; falling through to the provider chain",
            binding.kind, request.workload, exc_info=True,
        )
        return None
    if result is None:
        return None
    if not isinstance(result, WorkloadResult):
        _logger.warning(
            "workload handler %r returned %s, expected WorkloadResult; ignoring",
            binding.kind, type(result).__name__,
        )
        return None
    return result


def _pre_call(ctx: hooks.PreCallContext) -> hooks.ShortCircuit | None:
    """Bridge the handler registry onto somm's own call path.

    Handlers are a somm concept, not an extension: this hook is somm's internal
    dispatch, registered once, and is not part of the handler contract. A handler
    author implements :class:`WorkloadHandler` and never touches hooks.
    """
    with _lock:
        binding = _bindings.get(ctx.workload)
    if binding is None or not binding.enabled:
        return None
    request = WorkloadRequest(
        workload=ctx.workload,
        project=ctx.project,
        prompt=ctx.prompt,
        system=ctx.system,
        messages=ctx.messages,
        model=ctx.model,
        provider=ctx.provider,
        response_format=ctx.response_format,
        config=dict(binding.config),
    )
    result = resolve(request)
    if result is None:
        return None
    return hooks.ShortCircuit(
        text=result.text,
        provider=binding.kind,
        model=result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
        raw=result.raw,
        tool_calls=result.tool_calls,
        source=binding.kind,
    )


def _install_hook() -> None:
    """Attach somm's handler dispatch to its own pre_call phase, once.

    Priority 50 puts it after redaction (which must see the outbound text first)
    and after the response cache — a cached answer is cheaper than any handler.
    """
    global _hook_installed
    with _lock:
        if _hook_installed:
            return
        _hook_installed = True
    hooks.register_hook(hooks.PRE_CALL, _pre_call, priority=50)


def _load_entry_points(force: bool = False) -> None:
    global _entry_points_loaded
    with _lock:
        if _entry_points_loaded and not force:
            return
        _entry_points_loaded = True
    try:
        from importlib.metadata import entry_points

        found = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - importlib differences
        return
    for entry in found:
        try:
            entry.load()()
        except Exception:
            _logger.warning(
                "workload handler entry point %r failed to load", entry.name, exc_info=True
            )

"""Extension hooks that must never break somm's call path.

somm ships no integrations with other tools. Instead it exposes process-wide
hook points any package can attach to at runtime. Hook failures are isolated:
extensions must never break the call path, starve later hooks, or prevent
telemetry from being written.

The lifecycle hook bus has three named phases:

``pre_call``
    Synchronous request-rewrite hooks. Hooks receive a mutable
    :class:`PreCallContext` and may edit it in place. The first hook to return a
    :class:`ShortCircuit` stops the phase and lets the caller use that synthetic
    response. Broken hooks are logged at WARNING and treated as "continue";
    mutations made before the exception are intentionally kept. Async functions
    are rejected at registration because this phase must run synchronously.

``post_call``
    Observe-only hooks for completed call events. Lower priority runs first;
    ties keep registration order. Async hooks are supported and are run to
    completion when there is no ambient event loop. This phase backs the legacy
    call-observer API.

``post_process``
    Observe-only hooks intended for heavier graders, exporters, and notifiers.
    Hooks are dispatched onto a lazy single-worker background executor so the
    caller does not wait for them. Async hooks run inside that worker.

Lower priority values run first, following the WordPress convention.

Existing integrations can keep using:

    import somm.hooks
    somm.hooks.set_correlation_provider(my_request_id_getter)
    somm.hooks.add_call_observer(my_audit_logger)

Registration is also available through entry points. Both ``somm.hooks`` and
``somm.plugins`` entry-point groups are loaded once per process; each entry
point must resolve to a zero-arg callable that registers hooks::

    [project.entry-points."somm.plugins"]
    mytool = "mytool.somm_integration:register"

Event dict keys for call observers include ``call_id``, ``correlation_id``,
``project``, ``workload``, ``provider``, ``model``, ``outcome``,
``tokens_in``, ``tokens_out``, ``latency_ms``, ``cost_usd``,
``temperature``, ``max_tokens``, and ``error_kind``. The
:func:`stamp_event` helper adds the current hook event schema version for new
call sites without mutating the original event.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import inspect
import logging
import threading
from collections.abc import Callable, MutableSequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

PRE_CALL = "pre_call"
POST_CALL = "post_call"
POST_PROCESS = "post_process"
HOOK_PHASES = (PRE_CALL, POST_CALL, POST_PROCESS)
HOOK_EVENT_SCHEMA_VERSION = 1

CorrelationProvider = Callable[[], "str | None"]
CallObserver = Callable[[dict[str, Any]], None]

_logger = logging.getLogger("somm.hooks")


@dataclass
class PreCallContext:
    """Mutable request context passed to ``pre_call`` hooks."""

    workload: str
    prompt: str | list[Any] | None
    system: str
    messages: list[Any] | None
    model: str | None
    provider: str | None
    max_tokens: int
    temperature: float
    tools: list[Any]
    tool_choice: Any
    project: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "project" and "project" in self.__dict__:
            raise AttributeError("project is read-only")
        super().__setattr__(name, value)


@dataclass
class ShortCircuit:
    """Synthetic response returned by a ``pre_call`` hook."""

    text: str
    provider: str = "hook"
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] | None = None
    tool_calls: list[Any] | None = None
    source: str = ""


@dataclass
class _HookRegistration:
    priority: int
    insertion_index: int
    fn: Callable[..., Any]


_correlation_provider: CorrelationProvider | None = None
_hooks_by_phase: dict[str, list[_HookRegistration]] = {
    phase: [] for phase in HOOK_PHASES
}
_next_insertion_index = 0
_entry_points_loaded = False
_entry_points_lock = threading.Lock()
_post_process_executor: ThreadPoolExecutor | None = None
_post_process_executor_lock = threading.Lock()
# Bound the background post_process work queue: a slow or hung exporter/
# notifier must not let pending jobs (and their captured event dicts) grow
# without limit. When saturated we drop the newest job — dropping telemetry
# side-effects under sustained backpressure is strictly better than an OOM.
_POST_PROCESS_MAX_PENDING = 1024
_post_process_pending = 0
_post_process_pending_lock = threading.Lock()


def _validate_phase(phase: str) -> None:
    if phase not in _hooks_by_phase:
        raise ValueError(f"invalid hook phase {phase!r}; expected one of {HOOK_PHASES!r}")


def _next_index() -> int:
    global _next_insertion_index
    index = _next_insertion_index
    _next_insertion_index += 1
    return index


def _sort_phase(phase: str) -> None:
    _hooks_by_phase[phase].sort(key=lambda hook: (hook.priority, hook.insertion_index))


def _registration(fn: Callable[..., Any], priority: int) -> _HookRegistration:
    return _HookRegistration(priority=priority, insertion_index=_next_index(), fn=fn)


def register_hook(phase: str, fn: Callable[..., Any], priority: int = 100) -> None:
    """Register ``fn`` in ``phase``.

    Lower priority runs first; equal priorities keep registration order.
    Invalid phases and async ``pre_call`` hooks fail immediately so broken
    wiring is visible before a request is in flight.
    """
    _validate_phase(phase)
    if phase == PRE_CALL and inspect.iscoroutinefunction(fn):
        raise ValueError("pre_call hooks must be synchronous; async hooks cannot mutate or short-circuit safely")
    _hooks_by_phase[phase].append(_registration(fn, priority))
    _sort_phase(phase)


def unregister_hook(phase: str, fn: Callable[..., Any]) -> None:
    """Unregister ``fn`` from ``phase``. No-op if absent."""
    _validate_phase(phase)
    _hooks_by_phase[phase][:] = [
        hook for hook in _hooks_by_phase[phase] if hook.fn != fn
    ]


class _CallObserverView(MutableSequence[CallObserver]):
    """Compatibility view over the ``post_call`` phase.

    Older tests and integrations may have reached into ``_call_observers``.
    Keep it list-like without making it a second storage source.
    """

    def __len__(self) -> int:
        return len(_hooks_by_phase[POST_CALL])

    def __getitem__(self, index):
        hooks = _hooks_by_phase[POST_CALL]
        if isinstance(index, slice):
            return [hook.fn for hook in hooks[index]]
        return hooks[index].fn

    def __delitem__(self, index) -> None:
        del _hooks_by_phase[POST_CALL][index]

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            hooks = list(_hooks_by_phase[POST_CALL])
            hooks[index] = [_registration(fn, 100) for fn in value]
            _hooks_by_phase[POST_CALL][:] = hooks
            _sort_phase(POST_CALL)
            return
        _hooks_by_phase[POST_CALL][index] = _registration(value, 100)
        _sort_phase(POST_CALL)

    def insert(self, index: int, value: CallObserver) -> None:
        _hooks_by_phase[POST_CALL].insert(index, _registration(value, 100))
        _sort_phase(POST_CALL)


_call_observers: MutableSequence[CallObserver] = _CallObserverView()


def set_correlation_provider(fn: CorrelationProvider | None) -> None:
    """Set (or clear, with None) the process-wide correlation-id provider."""
    global _correlation_provider
    _correlation_provider = fn


def add_call_observer(fn: CallObserver) -> None:
    """Register an observer invoked after every completed call."""
    if fn not in _call_observers:
        register_hook(POST_CALL, fn)


def remove_call_observer(fn: CallObserver) -> None:
    """Unregister a previously added observer. No-op if absent."""
    unregister_hook(POST_CALL, fn)


def current_correlation_id() -> str | None:
    """Read the ambient correlation id. Never raises."""
    if _correlation_provider is None:
        return None
    try:
        return _correlation_provider()
    except Exception:
        return None


def fire_pre_call(ctx: PreCallContext) -> ShortCircuit | None:
    """Run ``pre_call`` hooks and return the first short-circuit response."""
    for hook in list(_hooks_by_phase[PRE_CALL]):
        try:
            result = hook.fn(ctx)
        except Exception:
            _logger.warning("pre_call hook failed; continuing", exc_info=True)
            continue
        if isinstance(result, ShortCircuit):
            return result
    return None


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(Exception):
        task.result()


def _run_awaitable(awaitable: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awaitable)
        return
    task = loop.create_task(awaitable)
    task.add_done_callback(_consume_task_exception)


def _fire_observer(fn: Callable[[dict[str, Any]], Any], event: dict[str, Any]) -> None:
    try:
        if inspect.iscoroutinefunction(fn):
            _run_awaitable(fn(event))
            return
        result = fn(event)
        if inspect.isawaitable(result):
            _run_awaitable(result)
    except Exception:
        pass


def fire_post_call(event: dict[str, Any]) -> None:
    """Run observe-only ``post_call`` hooks in priority order.

    Each hook gets its own shallow copy of the event so an observe-only
    hook that mutates (or stashes) the dict can't corrupt what later hooks
    — or the post_process phase reading the same event on another thread —
    see. Event values are flat scalars, so a shallow copy fully isolates.
    """
    for hook in list(_hooks_by_phase[POST_CALL]):
        _fire_observer(hook.fn, dict(event))


def notify_call_observers(event: dict[str, Any]) -> None:
    """Fan an event out to all post-call observers.

    Compatibility wrapper for the legacy observer API.
    """
    fire_post_call(event)


def _get_post_process_executor() -> ThreadPoolExecutor | None:
    global _post_process_executor
    if _post_process_executor is not None:
        return _post_process_executor
    with _post_process_executor_lock:
        if _post_process_executor is not None:
            return _post_process_executor
        try:
            _post_process_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="somm-postproc",
            )
        except Exception:
            return None
        return _post_process_executor


def fire_post_process(event: dict[str, Any]) -> None:
    """Dispatch observe-only ``post_process`` hooks without blocking callers."""
    hooks = list(_hooks_by_phase[POST_PROCESS])
    if not hooks:
        return
    executor = _get_post_process_executor()
    # Per-hook shallow copy: the caller's event dict must not be read on the
    # worker thread (the caller keeps running and post_call already touched
    # it), and one background hook must not mutate another's view.
    if executor is None:
        for hook in hooks:
            _fire_observer(hook.fn, dict(event))
        return
    for hook in hooks:
        with _post_process_pending_lock:
            if _post_process_pending >= _POST_PROCESS_MAX_PENDING:
                _logger.warning(
                    "post_process queue saturated (%d pending); dropping a job",
                    _post_process_pending,
                )
                continue
            _bump_pending(1)
        try:
            executor.submit(_run_post_process_job, hook.fn, dict(event))
        except Exception:
            with _post_process_pending_lock:
                _bump_pending(-1)
            _fire_observer(hook.fn, dict(event))


def _bump_pending(delta: int) -> None:
    global _post_process_pending
    _post_process_pending += delta


def _run_post_process_job(fn: Callable[[dict[str, Any]], Any], event: dict[str, Any]) -> None:
    try:
        _fire_observer(fn, event)
    finally:
        with _post_process_pending_lock:
            _bump_pending(-1)


def shutdown_hooks(wait: bool = False) -> None:
    """Stop the background post-process executor."""
    global _post_process_executor
    executor = _post_process_executor
    _post_process_executor = None
    if executor is not None:
        executor.shutdown(wait=wait)


def stamp_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``event`` stamped with the hook event schema version."""
    stamped = dict(event)
    stamped["schema_version"] = HOOK_EVENT_SCHEMA_VERSION
    return stamped


def _callable_name(fn: Callable[..., Any]) -> str:
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        qualname = type(fn).__qualname__
    if module:
        return f"{module}.{qualname}"
    return str(qualname)


def has_hooks(phase: str) -> bool:
    """O(1) check: are any hooks registered for a phase?

    The call path uses this to skip building a PreCallContext (or any
    per-phase work) when nothing is attached — the common case.
    """
    return bool(_hooks_by_phase.get(phase))


def registered_hooks() -> dict[str, list[tuple[str, int]]]:
    """Return registered hook names and priorities, never raising."""
    try:
        return {
            phase: [(_callable_name(hook.fn), hook.priority) for hook in hooks]
            for phase, hooks in _hooks_by_phase.items()
        }
    except Exception:
        return {phase: [] for phase in HOOK_PHASES}


def load_entry_points() -> None:
    """Invoke every ``somm.hooks`` and ``somm.plugins`` entry point once.

    Idempotent; called on first SommLLM construction. A broken entry point is
    skipped silently, following the same never-break rule as hooks.
    """
    global _entry_points_loaded
    # Lock the check-and-set: two threads constructing SommLLM at startup
    # could both pass an unguarded flag check and register every entry-point
    # hook twice (duplicate webhooks/spans for plugins without their own
    # idempotency guard).
    with _entry_points_lock:
        if _entry_points_loaded:
            return
        _entry_points_loaded = True
    try:
        from importlib.metadata import entry_points

        for group in ("somm.hooks", "somm.plugins"):
            with contextlib.suppress(Exception):
                for ep in entry_points(group=group):
                    with contextlib.suppress(Exception):
                        ep.load()()
    except Exception:
        pass


atexit.register(lambda: shutdown_hooks(wait=True))

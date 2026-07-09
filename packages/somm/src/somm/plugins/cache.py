"""Per-process response cache for somm calls.

Hook events intentionally do not include response bodies, so this plugin cannot
populate itself from ``post_call`` without weakening that privacy boundary.
Instead, ``register()`` installs a ``pre_call`` lookup hook that can
short-circuit already-known responses, while ``wrap(llm)`` is the convenience
used to populate the cache from returned ``SommResult`` objects.

The cache is in-memory, per-process, TTL-bound, and LRU-evicted. It is not
shared across workers or processes.
"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from somm_core import Outcome, SommResult
from somm_core.models import Prompt
from somm_core.parse import stable_hash

from somm import hooks

_CACHE_KEY_METADATA = "_cache_key"
_DEFAULT_TTL_S = 300.0
_DEFAULT_MAXSIZE = 512


@dataclass
class _Entry:
    text: str
    expires_at: float
    raw: dict[str, Any] | None
    tool_calls: list[Any] | None
    model: str


_lock = threading.RLock()
_entries: OrderedDict[str, _Entry] = OrderedDict()
_ttl_s = _DEFAULT_TTL_S
_maxsize = _DEFAULT_MAXSIZE
_workloads: set[str] | None = None
_registered = False


def _prompt_content(prompt: Any, messages: list[Any] | None) -> Any:
    if messages is not None:
        return messages
    if isinstance(prompt, Prompt):
        return prompt.body
    return "" if prompt is None else prompt


def _cache_key(
    *,
    workload: str,
    model: str | None,
    provider: str | None = None,
    system: str,
    prompt: Any,
    messages: list[Any] | None,
    temperature: float,
    max_tokens: int,
    tools: list[Any],
    tool_choice: Any = None,
) -> str:
    content_hash = stable_hash(_prompt_content(prompt, messages))
    payload = {
        "workload": workload,
        "model": model or "",
        # provider and tool_choice change the semantics of the response, so a
        # cache that ignored them could return one provider's answer for a
        # call pinned to another, or a text answer for a forced tool call.
        # (Required capabilities are not keyed: they are typically inferred
        # from the prompt content — which IS keyed via content_hash — so an
        # explicit-capability collision on otherwise-identical content is a
        # narrow edge left to a future revision.)
        "provider": provider or "",
        "system": system,
        "content_hash": content_hash,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    return stable_hash(payload)


def _enabled_for_workload(workload: str) -> bool:
    return _workloads is None or workload in _workloads


def _lookup(key: str) -> _Entry | None:
    now = time.monotonic()
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _entries.pop(key, None)
            return None
        _entries.move_to_end(key)
        return entry


def _store(key: str, result: SommResult, model: str | None) -> None:
    with _lock:
        _entries[key] = _Entry(
            text=result.text,
            expires_at=time.monotonic() + _ttl_s,
            raw=result.raw,
            tool_calls=list(result.tool_calls) if result.tool_calls else None,
            model=model or result.model or "",
        )
        _entries.move_to_end(key)
        while len(_entries) > _maxsize:
            _entries.popitem(last=False)


def _pre_call(ctx: hooks.PreCallContext) -> hooks.ShortCircuit | None:
    try:
        if not _enabled_for_workload(ctx.workload):
            return None
        key = _cache_key(
            workload=ctx.workload,
            model=ctx.model,
            provider=ctx.provider,
            system=ctx.system,
            prompt=ctx.prompt,
            messages=ctx.messages,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            tools=ctx.tools,
            tool_choice=ctx.tool_choice,
        )
        ctx.metadata[_CACHE_KEY_METADATA] = key
        entry = _lookup(key)
        if entry is None:
            return None
        return hooks.ShortCircuit(
            text=entry.text,
            model=entry.model,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            raw=entry.raw,
            tool_calls=entry.tool_calls,
            source="cache",
        )
    except Exception:
        return None


def register(
    ttl_s: float = _DEFAULT_TTL_S,
    maxsize: int = _DEFAULT_MAXSIZE,
    workloads: set[str] | None = None,
) -> None:
    """Install the cache lookup hook.

    Lower-priority redaction hooks should run before this hook so cache keys are
    derived from redacted outbound text instead of secret values.
    """
    global _ttl_s, _maxsize, _workloads, _registered
    with _lock:
        _ttl_s = max(0.0, float(ttl_s))
        _maxsize = max(1, int(maxsize))
        _workloads = set(workloads) if workloads is not None else None
    if not _registered:
        hooks.register_hook(hooks.PRE_CALL, _pre_call, priority=10)
        _registered = True


def unregister() -> None:
    """Remove the cache lookup hook."""
    global _registered
    hooks.unregister_hook(hooks.PRE_CALL, _pre_call)
    _registered = False


def clear() -> None:
    """Clear cached responses without changing hook registration."""
    with _lock:
        _entries.clear()


class _CachingProxy:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def generate(self, *args: Any, **kwargs: Any) -> SommResult:
        workload = "default"
        key: str | None = None
        model: str | None = None
        try:
            bound = inspect.signature(self._llm.generate).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            workload = values.get("workload", "default")
            model = values.get("model")
            key = _cache_key(
                workload=workload,
                model=model,
                provider=values.get("provider"),
                system=values.get("system", ""),
                prompt=values.get("prompt"),
                messages=values.get("messages"),
                temperature=values.get("temperature", 0.2),
                max_tokens=values.get("max_tokens", 256),
                tools=values.get("tools") or [],
                tool_choice=values.get("tool_choice"),
            )
            if _enabled_for_workload(workload):
                entry = _lookup(key)
                if entry is not None:
                    return SommResult(
                        text=entry.text,
                        provider="cache",
                        model=entry.model,
                        tokens_in=0,
                        tokens_out=0,
                        latency_ms=0,
                        cost_usd=0.0,
                        call_id=str(uuid.uuid4()),
                        outcome=Outcome.OK,
                        raw=entry.raw,
                        tool_calls=entry.tool_calls or [],
                    )
        except Exception:
            key = None
        result = self._llm.generate(*args, **kwargs)
        try:
            if (
                key is not None
                and _enabled_for_workload(workload)
                and result.outcome == Outcome.OK
            ):
                _store(key, result, model)
        except Exception:
            pass
        return result


def wrap(llm: Any) -> Any:
    """Return a proxy that populates the in-memory cache for ``generate()``."""
    return _CachingProxy(llm)

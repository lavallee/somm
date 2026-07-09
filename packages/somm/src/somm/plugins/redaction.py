"""Outbound prompt redaction hook.

Register this before the cache plugin (default priority 5 vs cache's 10) so
cache keys are derived from redacted text, not raw secret or PII values.
"""

from __future__ import annotations

from typing import Any

from somm import hooks
from somm._redaction import scrub_text

_registered = False
_extra_patterns: list[str] | None = None
_priority = 5


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value, _extra_patterns)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _redact_value(item)
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _redact_value(item)
        return value
    return value


def _pre_call(ctx: hooks.PreCallContext) -> None:
    try:
        ctx.prompt = _redact_value(ctx.prompt)
        ctx.system = scrub_text(ctx.system, _extra_patterns)
        if ctx.messages is not None:
            ctx.messages = _redact_value(ctx.messages)
    except Exception:
        return None
    return None


def register(extra_patterns: list[str] | None = None, priority: int = 5) -> None:
    """Install outbound request redaction."""
    global _registered, _extra_patterns, _priority
    _extra_patterns = list(extra_patterns) if extra_patterns is not None else None
    _priority = priority
    if not _registered:
        hooks.register_hook(hooks.PRE_CALL, _pre_call, priority=priority)
        _registered = True


def unregister() -> None:
    """Remove outbound request redaction."""
    global _registered
    hooks.unregister_hook(hooks.PRE_CALL, _pre_call)
    _registered = False

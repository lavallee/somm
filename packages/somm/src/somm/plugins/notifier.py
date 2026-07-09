"""Webhook notifications for selected somm call events.

The hook runs in ``post_process``, so webhook I/O is dispatched through the
hook bus background executor and stays off the caller's hot path.

Outcome matching is deliberately liberal: ``error`` covers upstream errors,
empty/bad outputs, ``exhausted`` covers exhausted outcomes or error kinds, and
``fallback`` covers explicit fallback-shaped events when callers include them.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from somm import hooks

_logger = logging.getLogger("somm.plugins.notifier")
_DEFAULT_ON = {"error", "exhausted", "fallback"}

_registered = False
_webhook_url = ""
_on = set(_DEFAULT_ON)
_min_cost_usd: float | None = None
_timeout_s = 5.0


def _matches(event: dict[str, Any]) -> bool:
    outcome = str(event.get("outcome") or "").lower()
    error_kind = str(event.get("error_kind") or "").lower()
    wanted = {item.lower() for item in _on}
    if _min_cost_usd is not None:
        try:
            if float(event.get("cost_usd") or 0.0) >= _min_cost_usd:
                return True
        except (TypeError, ValueError):
            pass
    if "error" in wanted and outcome in {"error", "upstream_error", "empty", "bad_json"}:
        return True
    if "exhausted" in wanted and (outcome == "exhausted" or "exhausted" in error_kind):
        return True
    if "fallback" in wanted and (
        outcome == "fallback"
        or bool(event.get("fallback"))
        or "fallback" in error_kind
    ):
        return True
    return outcome in wanted


def _text(event: dict[str, Any]) -> str:
    bits = [
        f"somm {event.get('outcome') or 'event'}",
        f"workload={event.get('workload') or ''}",
        f"provider={event.get('provider') or ''}",
        f"model={event.get('model') or ''}",
    ]
    cost = event.get("cost_usd")
    if cost:
        bits.append(f"cost=${float(cost):.6f}")
    call_id = event.get("call_id")
    if call_id:
        bits.append(f"call_id={call_id}")
    return " ".join(bits)


def _post_process(event: dict[str, Any]) -> None:
    try:
        if not _matches(event):
            return
        response = httpx.post(
            _webhook_url,
            json={"text": _text(event)},
            timeout=_timeout_s,
        )
        response.raise_for_status()
    except Exception as exc:
        _logger.warning("somm notifier webhook failed: %s", exc)


def register(
    webhook_url: str,
    on: set[str] = _DEFAULT_ON,
    min_cost_usd: float | None = None,
    timeout_s: float = 5.0,
) -> None:
    """Install the webhook notification hook."""
    global _registered, _webhook_url, _on, _min_cost_usd, _timeout_s
    _webhook_url = webhook_url
    _on = set(on)
    _min_cost_usd = min_cost_usd
    _timeout_s = timeout_s
    if not _registered:
        hooks.register_hook(hooks.POST_PROCESS, _post_process)
        _registered = True


def unregister() -> None:
    """Remove the webhook notification hook."""
    global _registered
    hooks.unregister_hook(hooks.POST_PROCESS, _post_process)
    _registered = False

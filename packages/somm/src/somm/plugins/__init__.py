"""Opt-in reference plugins for somm's hook bus."""

from __future__ import annotations

from somm.plugins.cache import register as register_cache
from somm.plugins.notifier import register as register_notifier
from somm.plugins.otel_exporter import register as register_otel_exporter
from somm.plugins.redaction import register as register_redaction

REFERENCE_PLUGINS: dict[str, dict] = {
    "cache": {
        "module": "somm.plugins.cache",
        "summary": "Per-process in-memory response cache with pre-call short-circuit hits.",
        "phase": "pre_call",
        "extra": None,
    },
    "redaction": {
        "module": "somm.plugins.redaction",
        "summary": "Outbound secret and PII redaction before provider calls.",
        "phase": "pre_call",
        "extra": None,
    },
    "notifier": {
        "module": "somm.plugins.notifier",
        "summary": "Slack-compatible webhook notifications for notable call events.",
        "phase": "post_process",
        "extra": None,
    },
    "otel_exporter": {
        "module": "somm.plugins.otel_exporter",
        "summary": "OpenTelemetry span export for completed calls.",
        "phase": "post_process",
        "extra": "otel",
    },
}

__all__ = [
    "REFERENCE_PLUGINS",
    "register_cache",
    "register_redaction",
    "register_notifier",
    "register_otel_exporter",
]

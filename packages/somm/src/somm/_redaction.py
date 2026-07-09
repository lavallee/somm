"""Shared redaction helpers for operator-facing text and outbound hooks."""

from __future__ import annotations

import re
from collections.abc import Iterable

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[a-z]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
)
_GENERIC_SECRET_PATTERNS = (
    re.compile(
        r"\b(api[_-]?key|authorization|bearer|token)([\"':= ]+)([A-Za-z0-9_.~+/-]{16,})",
        re.IGNORECASE,
    ),
    re.compile(r"\bBearer\s+([A-Za-z0-9_.~+/-]{16,})", re.IGNORECASE),
)
_PII_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
)


def scrub_text(text: str, extra_patterns: Iterable[str] | None = None) -> str:
    """Redact common credentials and PII from ``text``.

    ``extra_patterns`` are regular expressions supplied by an opt-in caller.
    Invalid extra patterns are ignored so redaction can never break an LLM call.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    for pattern in _GENERIC_SECRET_PATTERNS:
        if pattern.pattern.startswith("\\bBearer"):
            text = pattern.sub("Bearer [redacted]", text)
        else:
            text = pattern.sub(r"\1\2[redacted]", text)
    for pattern in _PII_PATTERNS:
        text = pattern.sub("[redacted]", text)
    if extra_patterns:
        for raw_pattern in extra_patterns:
            try:
                text = re.sub(raw_pattern, "[redacted]", text)
            except re.error:
                continue
    return text

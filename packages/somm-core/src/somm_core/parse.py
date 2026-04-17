"""Shared parsing helpers: JSON extraction, <think> stripping, hashing.

These patterns are battle-tested against real flaky LLM output across multiple
providers (qwen2.5 double-quote quirk, ollama/minimax think-blocks, markdown
fences). Lifted conceptually from prior wrapper work without copying any
project-specific strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_MD_FENCE_RE = re.compile(r"^\s*```(?:json|javascript|js)?\s*\n?|\n?```\s*$", re.MULTILINE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_DOUBLE_QUOTE_RE = re.compile(r'""([^"]*)""')


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks. Handles nested-safe single-level."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def strip_markdown_fence(text: str) -> str:
    return _MD_FENCE_RE.sub("", text).strip()


def extract_balanced(text: str, open_ch: str = "{", close_ch: str = "}") -> str | None:
    """Return the first balanced open_ch..close_ch substring, or None."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> dict | list | None:
    """Best-effort extract JSON from a possibly-messy LLM response.

    Handles: markdown fences, think-blocks, qwen2.5 double-quote quirk,
    JSON embedded in prose, bracket-balanced sub-extraction. Returns
    the parsed value or None if nothing parses.
    """
    if not text:
        return None

    cleaned = strip_think_block(text)
    cleaned = strip_markdown_fence(cleaned)

    for parser in (
        lambda s: json.loads(s),
        lambda s: json.loads(_DOUBLE_QUOTE_RE.sub(r'"\1"', s)),
        lambda s: json.loads(extract_balanced(s, "{", "}") or ""),
        lambda s: json.loads(extract_balanced(s, "[", "]") or ""),
    ):
        try:
            result = parser(cleaned)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(result, (dict, list)):
            return result
    return None


def stable_hash(value: str | dict | list) -> str:
    """Deterministic 16-char hash for content addressing."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def workload_id(name: str, input_schema: Any = None, output_schema: Any = None) -> str:
    """Content-address a workload by name + schemas."""
    payload = json.dumps(
        {"name": name, "in": input_schema, "out": output_schema},
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_hash(payload)


def prompt_id(body: str) -> str:
    return stable_hash(body)

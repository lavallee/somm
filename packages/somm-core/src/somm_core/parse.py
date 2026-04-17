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


# ---------------------------------------------------------------------------
# Streaming: <think>...</think> buffered strip across chunks
# ---------------------------------------------------------------------------


class ThinkStreamStripper:
    """Stateful filter that strips <think>...</think> across arbitrary chunk
    boundaries.

    Usage:
        stripper = ThinkStreamStripper(lookahead_bytes=2048)
        for chunk_text in provider.stream():
            for out in stripper.feed(chunk_text):
                yield out
        tail = stripper.flush()
        if tail:
            yield tail

    Guarantees:
    - Never emits partial think content mid-stream (even if chunk boundary
      falls inside a <think> block).
    - If a <think> block exceeds `lookahead_bytes`, the buffer is flushed
      AS-IS (with the tag visible) to avoid unbounded memory — caller can
      detect this via `capped` on the next feed()'s result.
    - Tolerates think-blocks spanning multiple chunks.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, lookahead_bytes: int = 2048) -> None:
        self.lookahead_bytes = lookahead_bytes
        self._buf = ""
        self._in_think = False
        self.capped = False

    def feed(self, chunk: str) -> str:
        """Feed a new chunk. Return the strip-safe prefix that can be emitted."""
        if not chunk:
            return ""
        self._buf += chunk
        out_parts: list[str] = []

        while self._buf:
            if self._in_think:
                # Looking for </think>
                idx = self._buf.find(self._CLOSE)
                if idx < 0:
                    # No close yet — keep buffering, but guard against runaway
                    if len(self._buf) > self.lookahead_bytes:
                        self.capped = True
                        # Emit as-is and exit think mode (best-effort escape).
                        out_parts.append(self._buf)
                        self._buf = ""
                        self._in_think = False
                    return "".join(out_parts)
                # Drop the think content + close tag, then continue
                self._buf = self._buf[idx + len(self._CLOSE) :]
                self._in_think = False
                continue

            # Not in think: emit up to the next "<think>" or the last safe byte
            idx = self._buf.find(self._OPEN)
            if idx >= 0:
                if idx > 0:
                    out_parts.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self._OPEN) :]
                self._in_think = True
                continue

            # No "<think>" found. Hold back only the longest suffix of the
            # buffer that is a prefix of "<think>" — everything before is
            # safe to emit.
            max_k = min(len(self._buf), len(self._OPEN) - 1)
            hold = 0
            for k in range(max_k, 0, -1):
                if self._buf.endswith(self._OPEN[:k]):
                    hold = k
                    break
            if hold == 0:
                out_parts.append(self._buf)
                self._buf = ""
            else:
                out_parts.append(self._buf[:-hold])
                self._buf = self._buf[-hold:]
            return "".join(out_parts)

        return "".join(out_parts)

    def flush(self) -> str:
        """End of stream. Emit whatever remains in the buffer.

        If we ended mid-think (no close tag ever seen), emit the contents
        visibly — better to leak the think block than silently drop data.
        """
        tail = self._buf
        self._buf = ""
        if self._in_think:
            # Mid-think at end of stream: mark capped and emit.
            self.capped = True
            self._in_think = False
        return tail

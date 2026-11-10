"""What ``SommLLM.stream(..., events=True)`` yields.

A stream called without ``events`` yields ``str`` text chunks, as it always
has. With ``events=True`` it yields these instead, in arrival order, and
ends with exactly one :class:`StreamDone` once the call is recorded:

    for ev in llm.stream("...", workload="chat", events=True):
        if isinstance(ev, TextDelta):
            print(ev.text, end="")
        elif isinstance(ev, ToolCallDelta):
            ...  # ev.index, ev.id, ev.name, ev.arguments (a JSON fragment)
        elif isinstance(ev, StreamDone):
            print(ev.result.tokens_out, ev.result.call_id)
"""

from __future__ import annotations

from dataclasses import dataclass

from somm_core import SommResult

from somm.providers.base import ToolCallDelta


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Visible answer text (``<think>`` blocks already removed)."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """Chain-of-thought text, from adapters whose wire format streams it."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamDone:
    """The recorded call: usage, served provider and model, call id, outcome.

    ``usage`` says where the token counts came from: ``"provider"`` when the
    provider reported them, ``"estimate"`` when somm estimated them from the
    text because the provider sent none.
    """

    result: SommResult
    usage: str


StreamEvent = TextDelta | ReasoningDelta | ToolCallDelta | StreamDone

__all__ = ["ReasoningDelta", "StreamDone", "StreamEvent", "TextDelta", "ToolCallDelta"]

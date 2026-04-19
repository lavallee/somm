"""SommProvider protocol — public, stable, entry-point registerable.

Third parties implement this to ship new providers without forking somm.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class SommRequest:
    prompt: str | list[dict]
    system: str = ""
    max_tokens: int = 256
    temperature: float = 0.2
    model: str | None = None  # None = provider's default
    metadata: dict = field(default_factory=dict)
    # Capabilities required of the (provider, model) serving this request.
    # Auto-inferred from image blocks, merged with workload defaults, and
    # filtered against model_intel.capabilities_json by the router.
    capabilities_required: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SommResponse:
    text: str
    model: str  # actual model used (after routing)
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: dict | None = None


@dataclass(slots=True)
class SommChunk:
    text: str
    done: bool = False


@dataclass(slots=True)
class ProviderHealth:
    available: bool
    detail: str = ""


@dataclass(slots=True)
class SommModel:
    name: str
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=list)


@runtime_checkable
class SommProvider(Protocol):
    """Every provider adapter implements this protocol."""

    name: str

    def generate(self, request: SommRequest) -> SommResponse: ...

    def stream(self, request: SommRequest) -> Iterator[SommChunk]: ...

    def health(self) -> ProviderHealth: ...

    def models(self) -> list[SommModel]: ...

    def estimate_tokens(self, text: str | list[dict], model: str) -> int: ...

"""Parallel-worker slot assignment.

Problem: if you spin up N parallel workers all using the same library
instance, they'll stampede one provider (usually ollama). `parallel_slots`
returns a plan that stripes them across available providers so no single
one saturates.

The per-provider concurrency hints below are conservative starting points;
the model-intel worker (D3) will tune these from observed throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm.providers.base import SommProvider
    from somm.routing import ProviderHealthTracker


# Conservative per-provider concurrency. "How many parallel calls before
# we expect contention?" Updated by model-intel (D3).
CONCURRENCY_HINTS: dict[str, int] = {
    "ollama": 2,  # single local GPU in most setups
    "openrouter": 4,  # roster spreads load across models
    "minimax": 4,
    "anthropic": 8,
    "openai": 8,
}

_DEFAULT_HINT = 2


def parallel_slots(
    providers: list[SommProvider],
    n_slots: int,
    tracker: ProviderHealthTracker | None = None,
) -> list[str]:
    """Return a list of provider names, length n_slots, striped across providers.

    Cooled providers are skipped (via the tracker). If ALL providers are cooled
    or the list is empty, returns [first-provider-name] * n_slots as a
    last-resort — the caller will still see errors at call time, but the
    signature contract holds.
    """
    if n_slots <= 0:
        return []
    if not providers:
        return []

    available = providers
    if tracker is not None:
        available = [p for p in providers if not tracker.get(p.name).is_cooling()]
        if not available:
            # Fallback: first provider regardless of cooldown; user sees
            # the error when the call is made (no silent starvation).
            return [providers[0].name] * n_slots

    weights = [CONCURRENCY_HINTS.get(p.name, _DEFAULT_HINT) for p in available]
    total = sum(weights) or 1
    # Floor-allocate proportionally, then distribute the remainder so the
    # total exactly matches n_slots.
    alloc = [n_slots * w // total for w in weights]
    remainder = n_slots - sum(alloc)
    # Give remainder slots to providers with the largest fractional part.
    fractions = sorted(
        range(len(available)),
        key=lambda i: (n_slots * weights[i] / total) - alloc[i],
        reverse=True,
    )
    for i in fractions[:remainder]:
        alloc[i] += 1

    # Interleave so consecutive workers don't all land on the same provider.
    slots: list[str] = []
    remaining = list(alloc)
    while sum(remaining) > 0:
        for i, p in enumerate(available):
            if remaining[i] > 0:
                slots.append(p.name)
                remaining[i] -= 1
                if len(slots) == n_slots:
                    return slots
    return slots

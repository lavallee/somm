"""Data-boundary enforcement for somm's egress points.

The classifier and the tier scale live in :mod:`somm_core.boundary`. This
module applies them to provider objects: what tier a configured provider
reaches for a given model, which providers of a chain are admissible, and the
refusal that is raised when none are. Every enforcement point (router,
pinned calls, streams, the OpenRouter roster walk, workload handlers, shadow
eval, judges, the proxy, MCP compare, harnesses) goes through
:func:`somm_core.boundary.admit`, by way of the helpers here.
"""

from __future__ import annotations

from typing import Any

from somm_core.boundary import (
    DataTier,
    ResolvedBoundary,
    RouteTier,
    admit,
    classify,
    load_policy,
    parse_tier,
    refusal_message,
    resolve_boundary,
)

from somm.errors import SommPrivacyViolation

__all__ = [
    "DataTier",
    "ResolvedBoundary",
    "RouteTier",
    "admissible_providers",
    "base_url_of",
    "check_route",
    "admit",
    "classify",
    "parse_tier",
    "provider_routes",
    "provider_tier",
    "refuse",
    "resolve_boundary",
    "served_tier",
]


def _inner(provider: Any) -> Any:
    """Unwrap per-call wrappers (workload-policy providers) to the adapter."""
    seen = 0
    while hasattr(provider, "_provider") and seen < 5:
        provider = provider._provider
        seen += 1
    return provider


def base_url_of(provider: Any) -> str | None:
    value = getattr(_inner(provider), "base_url", None)
    return value if isinstance(value, str) and value else None


def provider_routes(
    provider: Any, model: str | None = None, *, roster: bool = True
) -> list[RouteTier]:
    """The routes a provider could take for ``model``.

    An explicit model is one route. With no model, a provider that walks its
    own roster (OpenRouter) has one route per roster model, because a
    ``:free`` model and a paid one can sit in different tiers; any other
    provider uses its default model. ``roster=False`` (streams, which do
    not walk a roster) classifies the provider's default model only.
    """
    name = getattr(provider, "name", "") or ""
    base_url = base_url_of(provider)
    policy = load_policy()
    if model:
        return [policy.classify(name, model, base_url=base_url)]
    explicit = getattr(provider, "default_model", None)
    inner = _inner(provider)
    members = getattr(inner, "roster", None)
    if roster and not explicit and isinstance(members, list) and members:
        return [policy.classify(name, m, base_url=base_url) for m in members]
    default = explicit or getattr(inner, "default_model", "") or ""
    return [policy.classify(name, default, base_url=base_url)]


def provider_tier(
    provider: Any, model: str | None = None, *, roster: bool = True
) -> RouteTier:
    """The strictest route a provider offers for ``model`` (what it can be
    admitted on)."""
    routes = provider_routes(provider, model, roster=roster)
    return min(routes, key=lambda r: r.tier.rank)


def admissible_providers(
    providers: list[Any], model: str | None, boundary: DataTier | str, *, roster: bool = True
) -> tuple[list[Any], list[RouteTier]]:
    """Split a chain into (admissible providers, refused routes)."""
    ok: list[Any] = []
    refused: list[RouteTier] = []
    for provider in providers:
        route = provider_tier(provider, model, roster=roster)
        if admit(route, boundary):
            ok.append(provider)
        else:
            refused.append(route)
    return ok, refused


def served_tier(provider: Any | None, provider_name: str, model: str | None) -> str | None:
    """Tier string to record in ``calls.data_tier`` for a served call."""
    if not provider_name:
        return None
    if provider is not None:
        return load_policy().classify(provider_name, model, base_url=base_url_of(provider)).tier.value
    return classify(provider_name, model).tier.value


def refuse(
    *,
    workload: str | None,
    boundary: ResolvedBoundary,
    refused: list[RouteTier],
    context: str = "",
) -> SommPrivacyViolation:
    """Build (not raise) the canonical refusal."""
    return SommPrivacyViolation(
        refusal_message(workload=workload, boundary=boundary, refused=refused, context=context),
        workload=workload,
        data_boundary=boundary.tier.value,
        boundary_source=boundary.source,
        refused=[
            {
                "provider": r.provider,
                "model": r.model,
                "tier": r.tier.value,
                "basis": r.basis,
                "origin": r.origin,
            }
            for r in refused
        ],
    )


def check_route(
    *,
    provider: str,
    model: str | None,
    boundary: ResolvedBoundary,
    workload: str | None,
    base_url: str | None = None,
    context: str = "",
) -> RouteTier:
    """Classify one named route and raise :class:`SommPrivacyViolation` if
    it is outside ``boundary``. Returns the classification."""
    route = load_policy().classify(provider, model, base_url=base_url)
    if not admit(route, boundary.tier):
        raise refuse(workload=workload, boundary=boundary, refused=[route], context=context)
    return route


def parse_optional_tier(value: str | None) -> DataTier | None:
    return None if value in (None, "") else parse_tier(value)

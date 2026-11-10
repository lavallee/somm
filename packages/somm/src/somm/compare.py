"""Run one prompt through several explicitly named models, side by side.

The one implementation behind ``somm compare`` (CLI) and the MCP server's
``somm_compare``, which adds its token caps and marks ``text`` untrusted. Each model is a
(provider, model) pin: nothing is routed and nothing falls back, because an
answer from a substituted model is not a comparison. Every pin is checked
against the workload's data boundary before any call is made.

``compare_models`` returns plain dicts, one per requested model, in order:

* success: ``provider, model, text, tokens_in, tokens_out, latency_ms,
  cost_usd, outcome, call_id``
* failure: ``provider, model, error`` (and ``refused: True`` when the data
  boundary or a privacy rule stopped the call; boundary refusals also carry
  ``tier`` and ``data_boundary``).

``text`` is the recorded model output, untrusted. A caller that hands it to
an agent should mark it as such.
"""

from __future__ import annotations

from typing import Any

from somm.providers.registry import PINNED_ONLY_PROVIDER_NAMES


def parse_model_specs(raw: list[str] | None) -> list[tuple[str, str]]:
    """Parse ``provider/model`` or ``provider:model`` specs.

    Accepts comma-separated items and repeated flags; a bare ``provider``
    yields an empty model; a spec with no provider is dropped.
    """
    specs: list[tuple[str, str]] = []
    for item in raw or []:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue
            if "/" in token:
                p, _, m = token.partition("/")
            elif ":" in token:
                p, _, m = token.partition(":")
            else:
                p, m = token, ""
            if p:
                specs.append((p, m))
    return specs


def compare_models(
    llm: Any,
    specs: list[tuple[str, str]],
    *,
    prompt: str,
    workload: str,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Call each ``(provider, model)`` in ``specs`` with ``prompt``; one dict each.

    ``llm`` is a ``SommLLM`` whose provider chain holds the providers named
    in ``specs``. The caller owns ``llm`` (and closes it) and any limits on
    how many models or tokens it is willing to spend.
    """
    from somm.boundary import admit, provider_tier
    from somm.errors import SommPrivacyViolation

    by_name = {p.name: p for p in llm.providers}
    results: list[dict] = []
    for provider_name, model in specs:
        provider = by_name.get(provider_name)
        if provider is None:
            results.append(
                {
                    "provider": provider_name,
                    "model": model,
                    "error": (
                        f"provider {provider_name!r} not configured (missing env key?)"
                        if provider_name not in PINNED_ONLY_PROVIDER_NAMES
                        else f"provider {provider_name!r} is pinned-only; pass it with "
                        "--provider on `somm generate`, not in a bench spec"
                    ),
                }
            )
            continue
        try:
            boundary = llm._resolve_boundary(
                llm.repo.workload_by_name(workload, llm.config.project), None
            )
            route = provider_tier(
                provider,
                model or None,
                boundary=None if boundary.is_default else boundary.tier,
            )
            if not boundary.is_default and not admit(route, boundary.tier):
                results.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "refused": True,
                        "tier": route.tier.value,
                        "data_boundary": boundary.tier.value,
                        "error": (
                            f"REFUSED (SOMM_PRIVACY_VIOLATION): route tier {route.tier.value} "
                            "is outside data boundary "
                            f"{boundary.tier.value} (from {boundary.source})"
                        ),
                    }
                )
                continue
            r = llm.generate(
                prompt=prompt,
                workload=workload,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                provider=provider_name,
                allow_fallback=False,
            )
            results.append(
                {
                    "provider": provider_name,
                    "model": model,
                    "text": r.text,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "outcome": r.outcome.value,
                    "call_id": r.call_id,
                }
            )
        except Exception as e:
            entry = {
                "provider": provider_name,
                "model": model,
                "error": f"{type(e).__name__}: {e}",
            }
            if isinstance(e, SommPrivacyViolation):
                entry["refused"] = True
            results.append(entry)
    return results

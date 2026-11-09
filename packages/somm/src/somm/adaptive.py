"""Public adaptive routing policy and pure decision API."""
from somm_core.adaptive import (
    AdaptivePolicy,
    RouteDecision,
    RouteTarget,
    WeeklyQuota,
    choose_route,
    human_turns,
)

__all__ = ["AdaptivePolicy", "RouteDecision", "RouteTarget", "WeeklyQuota", "choose_route", "human_turns"]


from somm.codex_account import WeeklyQuotaProbe
from somm.errors import SommDeferred

_QUOTA = WeeklyQuotaProbe()


def decide_workload(workload, messages: list[dict]) -> RouteDecision:
    """Resolve an opt-in workload policy using fresh, first-party quota metadata."""
    settings = (workload.policy or {}).get("adaptive")
    if settings is None:
        raise ValueError("somm/auto requires an adaptive workload policy")
    policy = AdaptivePolicy.parse(settings)
    decision = choose_route(policy, messages, _QUOTA.read())
    if decision.deferred:
        raise SommDeferred(decision.as_dict())
    return decision


def fallback_messages(messages: list[dict]) -> list[dict]:
    """Keep text/tools while removing Responses-only opaque replay for other APIs."""
    result = []
    for message in messages:
        copy = dict(message)
        if isinstance(copy.get("reasoning_details"), list):
            details = [d for d in copy["reasoning_details"]
                       if not isinstance(d, dict) or not str(d.get("type", "")).startswith("openai.responses.")]
            if details:
                copy["reasoning_details"] = details
            else:
                copy.pop("reasoning_details", None)
        result.append(copy)
    return result

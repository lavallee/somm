"""Pace-aware routing decisions from billing plans.

Translates metered-plan pacing (somm_core.plans) into a per-provider
routing decision the Router consults on every dispatch:

- ``ok``    — route normally.
- ``defer`` — deprioritize: try only after every in-pace provider has
  failed. Fires when a limit is meaningfully into its quota (past the
  plan's soft target) AND burning faster than the window is passing,
  or when a limit is exhausted but the plan isn't enforced.
- ``block`` — skip entirely: a limit is exhausted and the plan says
  ``enforce = true``.

Decisions are TTL-cached (default 60s) because computing them means
read-only scans of every fleet database — cheap, but not per-call
cheap. PAYG and free providers are always ``ok``; their constraint is
money or rate limits, which the budget gate and cooldowns already
handle.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from somm_core.plans import LimitStatus, Plan, limit_statuses, plan_for

Decision = str  # 'ok' | 'defer' | 'block'


class PlanGovernor:
    def __init__(
        self,
        plans: dict[str, Plan],
        db_paths_fn: Callable[[], list[Path]],
        ttl_s: float = 60.0,
    ) -> None:
        self.plans = plans
        self._db_paths_fn = db_paths_fn
        self.ttl_s = ttl_s
        self._cache: dict[str, tuple[float, Decision]] = {}

    def has_metered_limits(self) -> bool:
        return any(p.mode == "metered" and p.limits for p in self.plans.values())

    def statuses(self, provider: str | None = None) -> list[LimitStatus]:
        providers = [provider] if provider else None
        try:
            return limit_statuses(self._db_paths_fn(), self.plans, providers=providers)
        except Exception:
            return []

    def decision(self, provider: str) -> Decision:
        now = time.monotonic()
        cached = self._cache.get(provider)
        if cached and cached[0] > now:
            return cached[1]
        result = self._decide(provider)
        self._cache[provider] = (now + self.ttl_s, result)
        return result

    def _decide(self, provider: str) -> Decision:
        plan = plan_for(provider, self.plans)
        if plan.mode != "metered" or not plan.limits:
            return "ok"
        worst: Decision = "ok"
        for st in self.statuses(provider):
            if st.state == "exhausted":
                return "block" if plan.enforce else "defer"
            if st.state == "over_pace":
                worst = "defer"
        return worst

"""OpenRouter provider — the OpenAI-compatible base plus a free-tier roster.

OpenRouter speaks the OpenAI chat-completions wire format, so request shaping
(messages, tools, tool_choice, response_format, reasoning_effort), status
classification, think-stripping, and usage parsing all come from
:class:`OpenAICompatProvider`. This adapter adds what is specific to
OpenRouter:

- **Roster cycling.** The free tier is heavily rate-limited and models come
  and go. With no explicit model, each call walks ``roster`` in order,
  skipping models still in cooldown, and cools the ones that fail. Cooldowns
  live in the shared ProviderHealthTracker so they survive restarts.
- **Reported cost.** ``usage: {"include": true}`` asks OpenRouter to state
  what the call cost; the base parses ``usage.cost`` into
  ``SommResponse.cost_usd`` so the ledger records an invoice figure rather
  than an estimate.
- **Attribution headers.** ``HTTP-Referer`` and ``X-Title``, which OpenRouter
  recommends.
- **Tiers from evidence.** Under a boundary stricter than ``unvetted``, a
  model whose static tier is refused can still be admitted as ``zdr`` when
  OpenRouter's own ZDR list names qualifying upstreams (see
  :mod:`somm.providers.openrouter_evidence`). The request is then pinned to
  exactly those endpoints with fallbacks off, and the response is marked so
  the call records ``data_tier = 'zdr'``. Under the default boundary nothing
  here runs and the request is unchanged.

When every roster model is cooling or failed, the adapter raises a
provider-level ``SommTransientError`` (``model=""``) and the router falls
through to the next provider. Earlier versions put sentinel names such as
``*all_cooled`` in the model field; those leaked into per-model rollups as if
they were models.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from somm.errors import SommModelNotFound, SommTransientError
from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.base import SommModel, SommRequest, SommResponse

if TYPE_CHECKING:
    from somm.routing import ProviderHealthTracker


# Free-tier roster seed, used by every project that sets no
# SOMM_OPENROUTER_ROSTER. Keep it short and keep it alive: a withdrawn model
# left here is retried by every such project until its cooldown escalates.
# Pruned 2026-10 — openrouter/elephant-alpha (withdrawn April 2026),
# google/gemma-3-27b-it:free and meta-llama/llama-3.3-70b-instruct:free (no
# successful call since July) had together failed 100,000 times in one ledger.
# SommRequest.metadata key carrying the upstream endpoints to pin one attempt to.
_PIN_KEY = "openrouter_only"

DEFAULT_FREE_ROSTER: list[str] = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]


class OpenRouterProvider(OpenAICompatProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        roster: list[str] | None = None,
        tracker: ProviderHealthTracker | None = None,
        timeout: float = 60.0,
        referer: str = "https://github.com/lavallee/somm",
        app_title: str = "somm",
    ) -> None:
        super().__init__(api_key=api_key, timeout=timeout)
        self.roster = list(roster) if roster else list(DEFAULT_FREE_ROSTER)
        self.tracker = tracker
        self.referer = referer
        self.app_title = app_title

    # ------------------------------------------------------------------

    def extra_headers(self) -> dict[str, str]:
        return {"HTTP-Referer": self.referer, "X-Title": self.app_title}

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        payload = super()._build_payload(request, model)
        # OpenRouter now returns usage (with `cost`) on every response; the
        # flag is a no-op there but still turns it on for older gateways that
        # speak the same wire format. The base records `usage.cost` as a
        # reported cost.
        payload["usage"] = {"include": True}
        only = (request.metadata or {}).get(_PIN_KEY)
        if only:
            # `zdr` and `data_collection` are belt and braces: `only` alone
            # already names strict-ZDR endpoints, and no fallback may widen it.
            payload["provider"] = {
                "only": list(only),
                "allow_fallbacks": False,
                "zdr": True,
                "data_collection": "deny",
            }
        return payload

    def zdr_only(self, model: str) -> list[str]:
        """Endpoint tags that carry ``model`` under strict zero retention, by
        OpenRouter's live ZDR list and the owner's ``zdr_headquarters``."""
        from somm_core.boundary import load_policy

        from somm.providers.openrouter_evidence import load_evidence, qualifying_upstreams

        evidence = load_evidence()
        if not evidence:
            return []
        return qualifying_upstreams(
            model, evidence["zdr"], evidence["providers"], load_policy().zdr_headquarters
        )

    def generate(self, request: SommRequest) -> SommResponse:
        # Explicit model: a pin is a pin — no roster, no substitution.
        if request.model:
            return self._call_single(request, request.model)

        roster = self.roster
        if request.data_boundary:
            # A :free model and a paid one can sit in different tiers, so the
            # boundary is checked per roster model, before any is tried.
            from somm.boundary import (
                ResolvedBoundary,
                admit,
                parse_tier,
                provider_routes,
                refuse,
            )

            routes = provider_routes(self, boundary=request.data_boundary)
            roster = [r.model for r in routes if admit(r, request.data_boundary)]
            if not roster:
                raise refuse(
                    workload=None,
                    boundary=ResolvedBoundary(parse_tier(request.data_boundary), "request"),
                    refused=routes,
                    context="the openrouter roster",
                )

        last_exc: Exception | None = None
        tried_any = False
        for model in roster:
            if self.tracker and self.tracker.get(self.name, model).is_cooling():
                continue
            tried_any = True
            try:
                return self._call_single(request, model)
            except (SommTransientError, SommModelNotFound) as e:
                # A withdrawn free model answers 404 ("No endpoints found").
                # That used to be fatal here: it aborted the whole chain and,
                # never cooled, was retried on every call. Now it cools on the
                # same per-model ladder as a rate limit and the walk moves on.
                if self.tracker:
                    self.tracker.mark_failure(self.name, model, cooldown_s=e.cooldown_s)
                last_exc = e
                continue

        if not tried_any:
            raise SommTransientError(
                "all openrouter roster models in cooldown",
                cooldown_s=_estimated_roster_cooldown(self, default=120.0),
            )
        raise SommTransientError(
            f"all openrouter roster models failed this round (last: {last_exc})",
            cooldown_s=120.0,
        )

    def _call_single(self, request: SommRequest, model: str) -> SommResponse:
        # The base class raises on any non-2xx or in-body error and returns a
        # fully parsed response. Pin the model for this attempt without
        # mutating the caller's request.
        pinned = replace(request, model=model)
        only: list[str] = []
        if request.data_boundary:
            from somm.boundary import ResolvedBoundary, admit, evidence_route, parse_tier, refuse

            # Decided once per attempt: what goes on the wire is what is recorded.
            route, only = evidence_route(self, model, request.data_boundary)
            if not admit(route, request.data_boundary):
                raise refuse(
                    workload=(request.metadata or {}).get("somm_workload"),
                    boundary=ResolvedBoundary(parse_tier(request.data_boundary), "request"),
                    refused=[route],
                    context=f"openrouter model {model!r}",
                )
            if only:
                pinned = replace(pinned, metadata={**(request.metadata or {}), _PIN_KEY: only})
        resp = super().generate(pinned)
        if only:
            served = resp.raw.get("provider") if isinstance(resp.raw, dict) else None
            resp.raw = {
                **(resp.raw or {}),
                "somm_route": {
                    "data_tier": "zdr",
                    "only": only,
                    # Undocumented but present on OpenRouter responses: the
                    # upstream's display name. Kept only when it is a string.
                    "served_by": served if isinstance(served, str) else None,
                },
            }
        if self.tracker:
            # Best-effort: a locked ledger must not turn this paid answer
            # into a failure the router would retry elsewhere.
            from somm.routing import _record_health

            _record_health(self.tracker.mark_ok, self.name, model)
        return resp

    def models(self):
        # /models unreachable → the roster is still the honest answer.
        return super().models() or [SommModel(name=m) for m in self.roster]

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        from somm_core.parse import estimate_prompt_tokens

        return estimate_prompt_tokens(text, image_token_cost=1200)


# ---------------------------------------------------------------------------


def _estimated_roster_cooldown(provider: OpenRouterProvider, default: float) -> float:
    """Peek the minimum cooldown across the roster; used when all are cooling."""
    if not provider.tracker:
        return default
    soonest: float = default
    now = datetime.now(UTC)
    for model in provider.roster:
        rec = provider.tracker.get(provider.name, model)
        if rec.cooldown_until:
            remaining = (rec.cooldown_until - now).total_seconds()
            if remaining > 0 and remaining < soonest:
                soonest = remaining
    return max(5.0, soonest)

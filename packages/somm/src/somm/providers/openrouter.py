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

from somm.errors import SommTransientError
from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.base import SommModel, SommRequest, SommResponse

if TYPE_CHECKING:
    from somm.routing import ProviderHealthTracker


# Free-tier roster seed. The model-intel worker refreshes availability from
# OpenRouter's /models endpoint; keep this short — it is a seed, not the
# source of truth.
DEFAULT_FREE_ROSTER: list[str] = [
    "openrouter/elephant-alpha",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
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
        # Ask for the invoice figure; the base records it as reported cost.
        payload["usage"] = {"include": True}
        return payload

    def generate(self, request: SommRequest) -> SommResponse:
        # Explicit model: a pin is a pin — no roster, no substitution.
        if request.model:
            return self._call_single(request, request.model)

        last_exc: Exception | None = None
        tried_any = False
        for model in self.roster:
            if self.tracker and self.tracker.get(self.name, model).is_cooling():
                continue
            tried_any = True
            try:
                return self._call_single(request, model)
            except SommTransientError as e:
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
        resp = super().generate(pinned)
        if self.tracker:
            self.tracker.mark_ok(self.name, model)
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

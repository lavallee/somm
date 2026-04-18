"""OpenRouter provider — free-roster + per-model cooldown pattern.

OpenRouter's free tier is heavily subject to rate limits + ephemeral
availability. Internal roster cycling with per-model cooldowns is the
pattern battle-tested in prior LLM-heavy codebases (see PLAN.md §0B).

- Iterates through `roster` on each call, skipping models still cooling.
- Uses the shared ProviderHealthTracker so cooldowns survive restarts.
- Raises SommTransientError when every model in the roster is cooled —
  the Router then falls through to the next provider.
- Strips `<think>` blocks post-hoc (models with reasoning leak them).
- Honors HTTP-Referer + X-Title headers (recommended by OpenRouter).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
from somm_core.parse import strip_think_block

from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommRateLimited,
    SommTimeout,
    SommTransientError,
    SommUpstream5xx,
)
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)

if TYPE_CHECKING:
    from somm.routing import ProviderHealthTracker


# Free-tier roster as of Apr 2026. Tested live — 404'd models removed.
# Model-intel worker (D3) will refresh this from OpenRouter's /models
# endpoint. Keep this short — it's a seed, not the source of truth.
DEFAULT_FREE_ROSTER: list[str] = [
    "openrouter/elephant-alpha",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


class OpenRouterProvider:
    name = "openrouter"

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        roster: list[str] | None = None,
        tracker: ProviderHealthTracker | None = None,
        timeout: float = 60.0,
        referer: str = "https://github.com/somm-dev/somm",
        app_title: str = "somm",
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouterProvider requires an api_key")
        self.api_key = api_key
        self.roster = list(roster) if roster else list(DEFAULT_FREE_ROSTER)
        self.tracker = tracker
        self.timeout = timeout
        self.referer = referer
        self.app_title = app_title

    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.app_title,
        }

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

    def generate(self, request: SommRequest) -> SommResponse:
        """Try each model in roster; return first success.

        Raises SommTransientError when all roster models are cooled (router
        falls through) or SommFatalError (auth, bad request) immediately.
        """
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
                model="*all_cooled",
            )
        raise SommTransientError(
            f"all openrouter roster models failed this round (last: {last_exc})",
            cooldown_s=120.0,
            model="*roster_exhausted",
        )

    def _call_single(self, request: SommRequest, model: str) -> SommResponse:
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.API_URL,
                    headers=self._headers(),
                    json=self._build_payload(request, model),
                )
        except httpx.TimeoutException as e:
            raise SommTimeout(f"openrouter timeout on {model}: {e}", cooldown_s=60.0, model=model) from e
        except httpx.RequestError as e:
            raise SommTransientError(f"network error on {model}: {e}", cooldown_s=30.0, model=model) from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        if resp.status_code == 401 or resp.status_code == 403:
            raise SommAuthError(
                f"openrouter rejected credentials ({resp.status_code}): {resp.text[:200]}"
            )
        if resp.status_code == 400:
            raise SommBadRequest(f"openrouter 400 on {model}: {resp.text[:200]}")
        if resp.status_code == 429:
            retry = _retry_after(resp) or 120.0
            raise SommRateLimited(f"openrouter 429 on {model}", retry_after_s=retry)
        if 500 <= resp.status_code < 600:
            raise SommUpstream5xx(f"openrouter {resp.status_code} on {model}", cooldown_s=30.0, model=model)
        if resp.status_code != 200:
            raise SommTransientError(
                f"openrouter unexpected {resp.status_code} on {model}: {resp.text[:200]}",
                cooldown_s=30.0, model=model,
            )

        data = resp.json()
        # OpenRouter returns error *inside* a 200 response sometimes.
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            if code == 429 or "rate" in msg.lower():
                raise SommRateLimited(f"openrouter body-error: {msg}", retry_after_s=120.0)
            raise SommTransientError(f"openrouter body-error on {model}: {msg}", cooldown_s=60.0, model=model)

        choices = data.get("choices") or []
        if not choices:
            raise SommTransientError(f"openrouter: no choices on {model}", cooldown_s=15.0)
        raw_text = choices[0].get("message", {}).get("content", "")
        text = strip_think_block(raw_text)

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))

        if self.tracker:
            self.tracker.mark_ok(self.name, model)

        return SommResponse(
            text=text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw=data,
        )

    # ------------------------------------------------------------------

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:  # pragma: no cover
        # D3 delivers full streaming with <think> buffered strip.
        resp = self.generate(request)
        yield SommChunk(text=resp.text, done=True)

    def health(self) -> ProviderHealth:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                r.raise_for_status()
            return ProviderHealth(available=True, detail="openrouter.ai reachable")
        except Exception as e:
            return ProviderHealth(available=False, detail=str(e))

    def models(self) -> list[SommModel]:
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                r.raise_for_status()
                data = r.json()
        except Exception:
            return [SommModel(name=m) for m in self.roster]
        out: list[SommModel] = []
        for m in data.get("data", []):
            name = m.get("id")
            if not name:
                continue
            out.append(
                SommModel(
                    name=name,
                    context_window=m.get("context_length"),
                )
            )
        return out

    def estimate_tokens(self, text: str, model: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------


def _retry_after(resp: httpx.Response) -> float | None:
    """Parse Retry-After (seconds, or HTTP-date)."""
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            then = parsedate_to_datetime(raw)
            now = datetime.now(UTC)
            if then.tzinfo is None:
                then = then.replace(tzinfo=UTC)
            return max(0.0, (then - now).total_seconds())
        except Exception:
            return None


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

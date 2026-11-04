"""Minimax provider — OpenAI-compatible endpoint at api.minimax.io.

Minimax hosts strong open-ish models (MiniMax-M3, M2.7, M2.5, etc.) at
competitive free/paid pricing via Token Plan keys (sk-cp-* prefix).
The `/v1/chat/completions` endpoint follows OpenAI's wire format, with
quirks:
- Reasoning models emit `<think>...</think>` blocks; stripped
  via the OpenAI-compat base's `strip_think_block` call.
- Rate limits are aggressive — cooldowns per model.
- NOTE: the old domain api.minimaxi.com is dead. Use api.minimax.io.
"""

from __future__ import annotations

from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.base import SommRequest


class MinimaxProvider(OpenAICompatProvider):
    name = "minimax"
    base_url = "https://api.minimax.io/v1"
    default_model = "MiniMax-M3"

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        # MiniMax reasoning models use thinking tokens that consume budget before the
        # actual response. 3x headroom with 1024 floor.
        payload = super()._build_payload(request, model)
        payload["max_tokens"] = max(request.max_tokens * 3, 1024)
        return payload


PLAN_REMAINS_URL = "https://api.minimax.io/v1/token_plan/remains"


def plan_remains(api_key: str, *, timeout: float = 10.0) -> dict | None:
    """What MiniMax says is left on the plan: the vendor's own number.

    Returns {"interval_used_pct", "weekly_used_pct"} for text models, or None if it
    can't be read. MiniMax publishes percentages rather than token quotas, so this is
    the only authoritative view; ``plans.toml`` limits are estimates beside it.
    """
    import json
    import urllib.request

    req = urllib.request.Request(PLAN_REMAINS_URL, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        general = next(m for m in data["model_remains"] if m.get("model_name") == "general")
        return {
            "interval_used_pct": 100 - int(general["current_interval_remaining_percent"]),
            "weekly_used_pct": 100 - int(general["current_weekly_remaining_percent"]),
        }
    except Exception:  # noqa: BLE001 - absent or changed API: the caller reports "unknown"
        return None

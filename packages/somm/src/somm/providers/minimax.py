"""Minimax provider — OpenAI-compatible endpoint at api.minimax.io.

Minimax hosts strong open-ish models (MiniMax-M2.7, M2.5, etc.) at
competitive free/paid pricing via Token Plan keys (sk-cp-* prefix).
The `/v1/chat/completions` endpoint follows OpenAI's wire format, with
quirks:
- Reasoning models (M2.7) emit `<think>...</think>` blocks; stripped
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
    default_model = "MiniMax-M2.7"

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        # M2.7 uses thinking tokens that consume budget before the
        # actual response. 3x headroom with 1024 floor.
        payload = super()._build_payload(request, model)
        payload["max_tokens"] = max(request.max_tokens * 3, 1024)
        return payload

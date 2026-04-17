"""Minimax provider — OpenAI-compatible endpoint at api.minimaxi.com.

Minimax hosts strong open-ish models (MiniMax-M2, Text-01, etc.) at
competitive free/paid pricing. The `/v1/chat/completions` endpoint
follows OpenAI's wire format, with quirks:
- Reasoning models emit `<think>...</think>` blocks; stripped via
  the OpenAI-compat base's `strip_think_block` call.
- Rate limits are aggressive on free tier — cooldowns per model.
"""

from __future__ import annotations

from somm.providers._openai_compat import OpenAICompatProvider


class MinimaxProvider(OpenAICompatProvider):
    name = "minimax"
    base_url = "https://api.minimaxi.com/v1"
    default_model = "MiniMax-M2"

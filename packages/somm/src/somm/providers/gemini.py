"""Gemini provider — Google's OpenAI-compatible endpoint.

Google exposes Gemini at `generativelanguage.googleapis.com/v1beta/openai/`
with the same wire format as OpenAI's `/v1/chat/completions`. That lets
us reuse the OpenAICompatProvider base directly.

Model IDs use Google's naming (`gemini-2.5-pro`, `gemini-2.5-flash`,
`gemini-2.0-flash`, etc.). Auth is a bearer token with the Gemini API
key — no project/location needed for this endpoint variant.

## Thinking-budget quirk

Gemini 2.5 Pro is a reasoning model. Its internal thinking consumes
output-token budget but does NOT appear in `completion_tokens`. A
request with `max_tokens=200` will often return `completion_tokens=0`
and `finish_reason="length"` — Gemini burned all 200 tokens on
reasoning before emitting any assistant text.

We handle this the way somm handles MiniMax M2.7's `<think>` blocks:
- Default: multiply caller's `max_tokens` by 4x (floor 2048) so
  thinking + completion both fit.
- Reasoning effort: pass `reasoning_effort: "low"` to minimize
  thinking on simple requests (Gemini supports this on the
  OAI-compat endpoint).
- Flash models (`gemini-2.5-flash`, `gemini-2.0-flash`) have much
  lower thinking overhead — the 4x scale still applies but rarely bites.

Note: the OAI-compat endpoint is a subset of Gemini's native
capabilities. For structured outputs, function calling, or multimodal
tool use beyond plain image input, use the native
`/v1beta/models/{model}:generateContent` endpoint instead.
"""

from __future__ import annotations

from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.base import SommRequest


class GeminiProvider(OpenAICompatProvider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    default_model = "gemini-2.5-pro"

    # Default reasoning_effort for the OAI-compat endpoint. "low" keeps
    # thinking cheap for straightforward prompts; callers can override
    # per-request via request.metadata["reasoning_effort"].
    default_reasoning_effort: str = "low"

    # Multiply caller's max_tokens to leave room for thinking. Gemini
    # 2.5 Pro reasoning can burn 1–4K tokens before emitting text;
    # Flash models need less but the headroom is harmless.
    max_tokens_multiplier: int = 4
    max_tokens_floor: int = 2048

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        payload = super()._build_payload(request, model)
        payload["max_tokens"] = max(
            request.max_tokens * self.max_tokens_multiplier, self.max_tokens_floor
        )
        effort = request.metadata.get("reasoning_effort") or self.default_reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort
        return payload

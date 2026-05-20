"""Perplexity provider — search-grounded `sonar` models at api.perplexity.ai.

Perplexity exposes an OpenAI-compatible `/chat/completions` endpoint, but the
models are search-grounded: each response carries the synthesized answer plus
the web sources it cited. Those extra fields (`citations`,
`related_questions`) live at the top level of the response body, which the
OpenAI-compat base preserves on `SommResponse.raw` — callers read them from
there.

Models:
- ``sonar`` — fast, ~5-10 citations per query. Good for breadth.
- ``sonar-pro`` — stronger synthesis, more citations.
- ``sonar-deep-research`` — slow (30-120s+), 20-50+ citations with a full
  report. It rejects system messages and `temperature`, so we strip those.

Because the answer is only useful *with* its citations, callers should pin
this provider with ``no_fallback=True`` — a silent fallback to a non-grounded
model would return prose with no sources for the pipeline to extract.
"""

from __future__ import annotations

from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.base import SommRequest


class PerplexityProvider(OpenAICompatProvider):
    name = "perplexity"
    base_url = "https://api.perplexity.ai"
    default_model = "sonar"

    def __init__(
        self,
        api_key: str,
        default_model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        # sonar-deep-research can run 30-120s+; the timeout is a ceiling, so a
        # generous default is safe for the fast sonar models too.
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            timeout=timeout,
        )

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        payload = super()._build_payload(request, model)
        # Search-grounded extras — surfaced on SommResponse.raw.
        payload["return_citations"] = True
        payload["return_related_questions"] = True
        # Deep research mode doesn't accept system prompts or sampling params.
        if "deep-research" in model.lower():
            payload["messages"] = [
                m for m in payload["messages"] if m.get("role") != "system"
            ]
            payload.pop("temperature", None)
            payload.pop("max_tokens", None)
            payload.pop("max_completion_tokens", None)
        return payload

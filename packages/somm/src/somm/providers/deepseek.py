"""DeepSeek provider — OpenAI-compatible endpoint at api.deepseek.com.

DeepSeek hosts production models at low-cost paid pricing. The
`/v1/chat/completions` endpoint follows OpenAI's wire format. Notes:

- ``deepseek-flash`` (V4.1-Flash) is the default; ``deepseek-v4-pro`` is
  the larger tier. The older ``deepseek-chat`` / ``deepseek-reasoner``
  aliases were retired in July 2026.
- Thinking is a per-request dial (``reasoning_effort``: none|low|high|max);
  ``reasoning_content`` must be echoed back on multi-turn tool use.
- Reasoning calls can take a while; the per-call ``timeout`` defaults
  to 180s for this provider so longer thinking responses succeed.
- Rate limits are model-dependent; transient 429s cool down per the
  base class.
"""

from __future__ import annotations

from somm.providers._openai_compat import OpenAICompatProvider
from somm.providers.base import SommRequest


class DeepSeekProvider(OpenAICompatProvider):
    name = "deepseek"
    base_url = "https://api.deepseek.com/v1"
    default_model = "deepseek-flash"
    # DeepSeek accepts JSON mode but 400s on a json_schema response_format
    # ("This response_format type is unavailable now"). The schema rides in the
    # system prompt and generate_structured validates, so JSON mode is enough.
    structured_output_mode = "json_object"

    def __init__(self, api_key: str, default_model: str | None = None,
                 timeout: float = 180.0) -> None:
        # DeepSeek thinking calls (deepseek-reasoner) and large
        # structured-output requests can run >60s. Default to 180s.
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            timeout=timeout,
        )

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        payload = super()._build_payload(request, model)
        # Reasoner needs more headroom (thinking tokens consume budget
        # before the actual response). Mirror the minimax 3x heuristic.
        if "reasoner" in model.lower():
            payload["max_tokens"] = max(request.max_tokens * 3, 1024)
        return payload

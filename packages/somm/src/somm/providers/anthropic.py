"""Anthropic provider — Messages API.

Not OpenAI-compatible — distinct request/response shape. Adapter normalizes
to the somm SommRequest/SommResponse envelope so the Router treats it
uniformly with other providers.

Keeps the system prompt out of the `messages` array (Anthropic puts it
in a top-level `system` field).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

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
    ToolCall,
)

DEFAULT_ANTHROPIC_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


class AnthropicProvider:
    name = "anthropic"
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-haiku-4-5-20251001",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("AnthropicProvider requires an api_key")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        # `messages` (multi-turn) overrides `prompt` (single-user-turn) when set.
        # Anthropic accepts the somm-neutral message shape directly because
        # somm's format mirrors Anthropic's Messages API.
        if request.messages is not None:
            messages = request.messages
        else:
            messages = [{"role": "user", "content": request.prompt}]
        payload: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if request.system:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = [_translate_tool(t) for t in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = _translate_tool_choice(request.tool_choice)
        return payload

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.API_URL,
                    headers=self._headers(),
                    json=self._build_payload(request, model),
                )
        except httpx.TimeoutException as e:
            raise SommTimeout(f"anthropic timeout on {model}: {e}", cooldown_s=60.0) from e
        except httpx.RequestError as e:
            raise SommTransientError(
                f"anthropic network error on {model}: {e}", cooldown_s=30.0
            ) from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        self._classify_status(resp, model)
        data = resp.json()

        # Anthropic: content is a list of blocks. Text blocks concatenate
        # into `text`; tool_use blocks become ToolCall entries; thinking
        # blocks are intentionally ignored (not billed as output tokens in
        # the somm view, even though Anthropic may bill them).
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        # Anthropic returns parsed JSON in `input` already.
                        arguments=block.get("input") or {},
                    )
                )
            # "thinking" blocks are intentionally ignored here.
        text = strip_think_block("".join(text_parts))

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)

        return SommResponse(
            text=text,
            model=data.get("model") or model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw=data,
            tool_calls=tool_calls,
            stop_reason=str(data.get("stop_reason") or ""),
        )

    def _classify_status(self, resp: httpx.Response, model: str) -> None:
        sc = resp.status_code
        if sc == 200:
            return
        body = resp.text[:200]
        if sc in (401, 403):
            raise SommAuthError(f"anthropic auth failed ({sc}): {body}")
        if sc == 400:
            raise SommBadRequest(f"anthropic 400 on {model}: {body}")
        if sc == 404:
            raise SommBadRequest(f"anthropic 404 — model {model!r} not found: {body}")
        if sc == 429:
            retry = _retry_after(resp) or 120.0
            raise SommRateLimited(f"anthropic 429 on {model}: {body}", retry_after_s=retry)
        if sc == 529:  # overloaded (Anthropic-specific)
            raise SommTransientError(f"anthropic overloaded on {model}", cooldown_s=60.0)
        if 500 <= sc < 600:
            raise SommUpstream5xx(f"anthropic {sc} on {model}", cooldown_s=30.0)
        raise SommTransientError(f"anthropic unexpected {sc} on {model}: {body}", cooldown_s=30.0)

    # ------------------------------------------------------------------

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:  # pragma: no cover
        resp = self.generate(request)
        yield SommChunk(text=resp.text, done=True)

    def health(self) -> ProviderHealth:
        # Anthropic has no "list models" endpoint — a cheap probe is a
        # tokens-count-like POST, but that's overkill. Optimistically
        # assume reachable when key is set; failures surface on first call.
        return ProviderHealth(available=bool(self.api_key), detail="api-key present")

    def models(self) -> list[SommModel]:
        return [SommModel(name=m) for m in DEFAULT_ANTHROPIC_MODELS]

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        from somm_core.parse import estimate_prompt_tokens

        # Anthropic bills ~1568 tokens for a 1092×1092 image.
        return estimate_prompt_tokens(text, image_token_cost=1568)


def _translate_tool(tool: dict) -> dict:
    """somm-neutral → Anthropic tool shape. Rename `parameters` → `input_schema`."""
    out = {"name": tool["name"], "description": tool.get("description", "")}
    if "parameters" in tool:
        out["input_schema"] = tool["parameters"]
    elif "input_schema" in tool:
        # Tolerate Anthropic-native form passed through unchanged.
        out["input_schema"] = tool["input_schema"]
    return out


def _translate_tool_choice(choice: str | dict) -> dict:
    """somm-neutral → Anthropic tool_choice."""
    if choice == "auto":
        return {"type": "auto"}
    if choice == "any":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return {"type": "tool", "name": choice["name"]}
    raise ValueError(f"unrecognized tool_choice: {choice!r}")


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None

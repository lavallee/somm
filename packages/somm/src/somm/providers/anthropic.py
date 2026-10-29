"""Anthropic provider — Messages API.

Not OpenAI-compatible — distinct request/response shape. Adapter normalizes
to the somm SommRequest/SommResponse envelope so the Router treats it
uniformly with other providers.

Keeps the system prompt out of the `messages` array (Anthropic puts it
in a top-level `system` field).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

import httpx
from somm_core.parse import (
    anthropic_prompt_tokens,
    reasoning_tokens_from_usage,
    strip_think_block,
)

from somm.errors import (
    SommTimeout,
    SommTransientError,
    classify_http_status,
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
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
]

_VERSIONED = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?")


def accepts_temperature(model: str) -> bool:
    """Whether this model still takes sampling parameters.

    Claude 4.7 and later reject ``temperature`` (Fable/Mythos and Opus 4.7+
    with a 400 on any value; Sonnet 5.5 and Haiku 5.5 on any non-default
    value). Sending it would fail every call on the current lineup, so it is
    omitted there and kept for the 4.6-and-earlier models that honor it.
    Unknown names send it: a 400 is louder than a silently ignored dial.
    """
    name = model.lower()
    if name.startswith(("claude-fable", "claude-mythos")):
        return False
    m = _VERSIONED.search(name)
    if not m:
        return True
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return (major, minor) < (4, 7)


class AnthropicProvider:
    name = "anthropic"
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-haiku-5-5",
        timeout: float = 60.0,
        auto_cache: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("AnthropicProvider requires an api_key")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        # Top-level automatic prompt caching: the API caches the last cacheable
        # prefix block (tools → system → messages) so a repeated system prompt
        # is read at 0.025–0.1x. Prefixes under the model's minimum are simply
        # not cached, so the flag costs nothing on short prompts.
        self.auto_cache = auto_cache

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
            # Strip OpenAI-compat-only keys (e.g. `reasoning_content`, which the
            # somm_langchain adapter attaches for DeepSeek thinking models).
            # Anthropic rejects unknown message keys; it has no use for them.
            messages = [
                {
                    k: (_content_to_anthropic(v) if k == "content" else v)
                    for k, v in m.items()
                    if k not in ("reasoning_content",)
                }
                for m in request.messages
            ]
        else:
            messages = [{"role": "user", "content": _content_to_anthropic(request.prompt)}]
        payload: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if accepts_temperature(model):
            payload["temperature"] = request.temperature
        if request.system:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = [_translate_tool(t) for t in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = _translate_tool_choice(request.tool_choice)
        output_config: dict = {}
        if request.reasoning_effort:
            # GA: low | medium | high | xhigh | max. Thinking itself is adaptive
            # and on by default on current models; effort is the dial.
            output_config["effort"] = request.reasoning_effort
        schema = _json_schema_from_response_format(request.response_format)
        if schema is not None:
            # Structured outputs (GA): the first content block is text holding
            # JSON that validates against the schema.
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            payload["output_config"] = output_config
        if self.auto_cache:
            payload["cache_control"] = {"type": "ephemeral"}
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
        # blocks are not surfaced (their token count is, via
        # usage.output_tokens_details.thinking_tokens → reasoning_tokens).
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
        # The WHOLE prompt, not just the uncached slice. Anthropic splits it
        # across input_tokens / cache_read_input_tokens / cache_creation_input_tokens;
        # the OpenAI-compatible providers report one `prompt_tokens` that
        # already includes their cached portion. Reading input_tokens alone
        # made the same prompt report different sizes depending on which
        # provider served it, which breaks the cross-provider comparison somm
        # exists to do. cache_tokens_in/out still record the split separately.
        tokens_in = anthropic_prompt_tokens(usage)
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
            reasoning_tokens=reasoning_tokens_from_usage(usage),
        )

    def _classify_status(self, resp: httpx.Response, model: str) -> None:
        if 200 <= resp.status_code < 300:
            return
        # 529 is Anthropic's "overloaded": transient, retry shortly.
        classify_http_status(
            resp.status_code,
            resp.text,
            provider=self.name,
            model=model,
            headers=resp.headers,
            overloaded_statuses=(529,),
        )

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


def _json_schema_from_response_format(response_format: dict | None) -> dict | None:
    """The schema inside an OpenAI-style ``response_format``, if it carries one.

    ``json_object`` (JSON mode without a schema) has no Anthropic equivalent;
    the system prompt already asks for JSON and ``generate_structured``
    validates, so it is simply not forwarded.
    """
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return None
    inner = response_format.get("json_schema")
    schema = inner.get("schema") if isinstance(inner, dict) else None
    return schema if isinstance(schema, dict) else None


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


def _content_to_anthropic(content):
    """Accept OpenAI-shaped `image_url` blocks on the native endpoint.

    somm's neutral image block is Anthropic's own shape, so it passes through.
    A caller that built `{"type": "image_url", ...}` for an OpenAI-compatible
    provider (issue #70's downstream workaround) would otherwise 400 here, so
    data URLs become base64 sources and other URLs become url sources.
    """
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            image_url = block.get("image_url")
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url or "")
            if url.startswith("data:") and ";base64," in url:
                header, data = url[len("data:") :].split(";base64,", 1)
                source = {"type": "base64", "media_type": header or "image/png", "data": data}
            else:
                source = {"type": "url", "url": url}
            out.append({"type": "image", "source": source})
        else:
            out.append(block)
    return out

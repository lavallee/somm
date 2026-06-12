"""Shared HTTP call logic for OpenAI-compatible /v1/chat/completions endpoints.

Used by OpenAIProvider (api.openai.com), MinimaxProvider (api.minimaxi.com),
and any future third-party provider that implements the same wire format
(Groq, Together, Fireworks, vLLM, LM Studio, custom internal gateways).

Classifies HTTP status codes into the SommError hierarchy so the router
treats every provider uniformly. Each subclass just overrides `name`,
`base_url`, and optionally `extra_headers()`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
from somm_core.parse import strip_think_block

from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommInsufficientCredit,
    SommRateLimited,
    SommTimeout,
    SommTransientError,
    SommUpstream5xx,
    looks_like_insufficient_credit,
)
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class OpenAICompatProvider:
    """Base class for OpenAI-compatible chat-completion endpoints.

    Subclasses set: `name`, `base_url`, `default_model`, and (optionally)
    `extra_headers()`. This class handles request shape, status-code
    classification, think-stripping, and token/usage parsing.
    """

    name: str = "openai-compat"
    base_url: str = "https://api.example.com/v1"
    default_model: str = ""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(f"{self.name} requires an api_key")
        self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if default_model is not None:
            self.default_model = default_model
        self.timeout = timeout

    # ------------------------------------------------------------------

    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _models_url(self) -> str:
        return f"{self.base_url}/models"

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def extra_headers(self) -> dict[str, str]:
        """Subclasses override to add provider-specific headers (e.g. openrouter referer)."""
        return {}

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(self._auth_header())
        h.update(self.extra_headers())
        return h

    def _build_payload(self, request: SommRequest, model: str) -> dict:
        # Multi-turn `messages` overrides single-turn `prompt`. The somm-neutral
        # message shape mirrors Anthropic; here we translate to OpenAI's
        # tool_calls / tool message conventions.
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if request.messages is not None:
            messages.extend(_translate_messages_to_openai(request.messages))
        else:
            messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if _uses_max_completion_tokens(model):
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        if request.tools:
            payload["tools"] = [_translate_tool_to_openai(t) for t in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = _translate_tool_choice_to_openai(request.tool_choice)
        return payload

    # ------------------------------------------------------------------

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model
        if not model:
            raise SommBadRequest(f"{self.name}: no model configured or requested")

        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self._chat_url(),
                    headers=self._headers(),
                    json=self._build_payload(request, model),
                )
        except httpx.TimeoutException as e:
            raise SommTimeout(f"{self.name} timeout on {model}: {e}", cooldown_s=60.0) from e
        except httpx.RequestError as e:
            raise SommTransientError(
                f"{self.name} network error on {model}: {e}", cooldown_s=30.0
            ) from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        self._classify_status(resp, model)

        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            self._raise_body_error(data["error"], model)

        choices = data.get("choices") or []
        if not choices:
            raise SommTransientError(f"{self.name}: no choices on {model}", cooldown_s=15.0)
        choice = choices[0]
        message = choice.get("message") or {}
        raw_text = message.get("content") or ""
        text = strip_think_block(raw_text)

        tool_calls = _parse_openai_tool_calls(message.get("tool_calls") or [])
        stop_reason = _normalize_finish_reason(choice.get("finish_reason"))
        # Thinking models (DeepSeek v4) return chain-of-thought here; it must be
        # echoed back on the assistant turn in subsequent calls (see
        # _translate_messages_to_openai), or DeepSeek 400s on turn 2.
        reasoning_content = message.get("reasoning_content") or ""

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)

        return SommResponse(
            text=text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw=data,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            reasoning_content=reasoning_content,
        )

    def _classify_status(self, resp: httpx.Response, model: str) -> None:
        sc = resp.status_code
        if sc == 200:
            return
        body = resp.text[:200]
        if looks_like_insufficient_credit(resp.text):
            # Out of money/quota — a billing state on this provider, not a bad
            # request or bad key. Transient so the router falls through to the
            # next provider instead of aborting the chain.
            raise SommInsufficientCredit(f"{self.name} out of credit on {model}: {body}")
        if sc in (401, 403):
            raise SommAuthError(f"{self.name} auth failed ({sc}): {body}")
        if sc == 400:
            raise SommBadRequest(f"{self.name} 400 on {model}: {body}")
        if sc == 404:
            raise SommBadRequest(f"{self.name} 404 — model {model!r} not found: {body}")
        if sc == 429:
            retry = _retry_after(resp) or 120.0
            raise SommRateLimited(f"{self.name} 429 on {model}: {body}", retry_after_s=retry)
        if 500 <= sc < 600:
            raise SommUpstream5xx(f"{self.name} {sc} on {model}", cooldown_s=30.0)
        raise SommTransientError(f"{self.name} unexpected {sc} on {model}: {body}", cooldown_s=30.0)

    def _raise_body_error(self, err: dict | str, model: str) -> None:
        if isinstance(err, dict):
            msg = err.get("message", "")
            code = err.get("code")
            typ = err.get("type", "")
            if looks_like_insufficient_credit(f"{code} {typ} {msg}"):
                raise SommInsufficientCredit(
                    f"{self.name} out of credit on {model}: {msg}"
                )
            if code == 429 or "rate" in str(msg).lower():
                raise SommRateLimited(
                    f"{self.name} body-429 on {model}: {msg}", retry_after_s=120.0
                )
            if "auth" in str(typ).lower() or "auth" in str(msg).lower():
                raise SommAuthError(f"{self.name} body-auth on {model}: {msg}")
            raise SommTransientError(f"{self.name} body-error on {model}: {msg}", cooldown_s=60.0)
        raise SommTransientError(f"{self.name} body-error on {model}: {err}", cooldown_s=60.0)

    # ------------------------------------------------------------------

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        """SSE-based streaming for OpenAI-compatible endpoints.

        Parses `data: {...}\\n\\ndata: [DONE]\\n\\n` frames. `<think>` stripping
        is the library's concern (see SommLLM.stream).
        """
        import json

        model = request.model or self.default_model
        if not model:
            raise SommBadRequest(f"{self.name}: no model configured or requested")

        payload = self._build_payload(request, model)
        payload["stream"] = True

        with (
            httpx.Client(timeout=self.timeout) as client,
            client.stream(
                "POST",
                self._chat_url(),
                headers=self._headers(),
                json=payload,
            ) as resp,
        ):
            if resp.status_code != 200:
                try:
                    text = resp.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    text = ""
                fake = httpx.Response(resp.status_code, text=text)
                self._classify_status(fake, model)
                return

            for raw_line in resp.iter_lines():
                line = raw_line.strip() if isinstance(raw_line, str) else raw_line
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    yield SommChunk(text="", done=True)
                    return
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    yield SommChunk(text=piece, done=False)
            yield SommChunk(text="", done=True)

    def health(self) -> ProviderHealth:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(self._models_url(), headers=self._auth_header())
                r.raise_for_status()
            return ProviderHealth(available=True, detail=f"{self.base_url} reachable")
        except Exception as e:
            return ProviderHealth(available=False, detail=str(e))

    def models(self) -> list[SommModel]:
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(self._models_url(), headers=self._auth_header())
                r.raise_for_status()
                data = r.json()
        except Exception:
            return [SommModel(name=self.default_model)] if self.default_model else []
        out: list[SommModel] = []
        for m in data.get("data", []):
            name = m.get("id") or m.get("model")
            if name:
                out.append(SommModel(name=name, context_window=m.get("context_length")))
        return out

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        from somm_core.parse import estimate_prompt_tokens

        # OpenAI: ~85 for low-res image + tiles for hi-res. Use a middling
        # estimate; a precise tokenizer lives behind `somm[tokenizers]` later.
        return estimate_prompt_tokens(text, image_token_cost=700)


def _translate_tool_to_openai(tool: dict) -> dict:
    """somm-neutral → OpenAI tool wrapping."""
    function: dict = {"name": tool["name"]}
    if "description" in tool:
        function["description"] = tool["description"]
    if "parameters" in tool:
        function["parameters"] = tool["parameters"]
    return {"type": "function", "function": function}


def _translate_tool_choice_to_openai(choice: str | dict) -> str | dict:
    """somm-neutral → OpenAI tool_choice."""
    if choice == "auto":
        return "auto"
    if choice == "any":
        return "required"
    if choice == "none":
        return "none"
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    raise ValueError(f"unrecognized tool_choice: {choice!r}")


def _translate_messages_to_openai(messages: list[dict]) -> list[dict]:
    """somm-neutral (Anthropic-shaped) → OpenAI messages.

    Translation rules:
    - Assistant message with text + tool_use blocks → single assistant
      message with `content` (concatenated text) and `tool_calls` array.
      OpenAI accepts an empty/None content when tool_calls are present.
    - User message with tool_result blocks → one `{role:"tool", ...}`
      message per tool_result. Any non-tool_result blocks in the same
      message are emitted as a separate user message that follows
      (preserves ordering).
    """
    import json as _json

    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls_payload: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    args = block.get("input") or {}
                    tool_calls_payload.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": _json.dumps(args),
                            },
                        }
                    )
            assistant_msg: dict = {"role": "assistant"}
            joined = "".join(text_parts)
            # OpenAI requires either content or tool_calls; allow content=None
            # when only tool_calls are present (matches OpenAI's docs).
            assistant_msg["content"] = joined if joined else None
            if tool_calls_payload:
                assistant_msg["tool_calls"] = tool_calls_payload
            # Thinking models (DeepSeek v4) require the prior turn's
            # reasoning_content echoed back, carried as a top-level key on the
            # somm-neutral assistant message by the somm_langchain adapter.
            if msg.get("reasoning_content"):
                assistant_msg["reasoning_content"] = msg["reasoning_content"]
            out.append(assistant_msg)
            continue

        if role == "user" and isinstance(content, list):
            # Split tool_results into separate `role:tool` messages; keep
            # any other blocks in a trailing user message.
            other_blocks: list[dict] = []
            for block in content:
                if block.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _stringify_tool_result(block.get("content", "")),
                        }
                    )
                else:
                    other_blocks.append(block)
            if other_blocks:
                out.append({"role": "user", "content": other_blocks})
            continue

        # Plain text content or system message — forward unchanged.
        out.append(msg)
    return out


def _stringify_tool_result(content: str | list | dict) -> str:
    """OpenAI's tool message wants a string; somm-neutral allows blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    import json as _json

    return _json.dumps(content)


def _parse_openai_tool_calls(raw_tool_calls: list[dict]) -> list[ToolCall]:
    """OpenAI tool_calls → somm ToolCall list.

    OpenAI sends `arguments` as a JSON string; parse it. Malformed JSON
    surfaces as `arguments={}` with `arguments_raw` populated so the
    caller can repair or re-prompt rather than silently corrupt the loop.
    """
    import json as _json

    out: list[ToolCall] = []
    for entry in raw_tool_calls:
        fn = entry.get("function") or {}
        raw_args = fn.get("arguments") or ""
        try:
            args = _json.loads(raw_args) if raw_args else {}
            arguments_raw = ""
        except _json.JSONDecodeError:
            args = {}
            arguments_raw = raw_args
        out.append(
            ToolCall(
                id=str(entry.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=args,
                arguments_raw=arguments_raw,
            )
        )
    return out


_OPENAI_FINISH_REASON_TO_SOMM: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",  # legacy single-call form
    "length": "max_tokens",
    "content_filter": "content_filter",
}


def _normalize_finish_reason(reason: str | None) -> str:
    if not reason:
        return ""
    return _OPENAI_FINISH_REASON_TO_SOMM.get(reason, reason)


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            then = parsedate_to_datetime(raw)
            if then.tzinfo is None:
                then = then.replace(tzinfo=UTC)
            return max(0.0, (then - datetime.now(UTC)).total_seconds())
        except Exception:
            return None


def _uses_max_completion_tokens(model: str) -> bool:
    """Newer OpenAI reasoning/chat models reject legacy `max_tokens`."""
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))

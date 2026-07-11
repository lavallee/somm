"""Ollama provider — local first-party adapter.

D1 target: `generate()` works against a live local ollama. Streaming is stubbed.
`<think>` block stripping applies before returning `.text`; tokens_out counts
only non-think content.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
from somm_core.parse import strip_think_block

from somm.errors import SommBadRequest
from somm.providers._openai_compat import (
    _translate_messages_to_openai,
    _translate_tool_to_openai,
)
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommEmbedRequest,
    SommEmbedResponse,
    SommModel,
    SommRequest,
    SommResponse,
    ToolCall,
)


class OllamaProvider:
    name = "ollama"

    DEFAULT_EMBED_MODEL = "nomic-embed-text"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "gemma4:e4b",
        default_embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = 120.0,
        enable_think: bool = False,
        keep_alive: str = "30m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.default_embed_model = default_embed_model
        self.timeout = timeout
        # When True, sets `"think": true` on the ollama request. Ollama 0.5+
        # uses this to opt reasoning-capable models (qwen3, deepseek-r1, etc.)
        # into their native thinking mode. Post-hoc <think> stripping still
        # runs regardless.
        self.enable_think = enable_think
        # Pinned resident window for the model. Ollama's default (~5 min)
        # evicts weights from GPU between sparse calls, paying the model-
        # load cost on the next request. For workload chains that
        # interleave many small calls (classifier+synthesis+evaluator) this
        # shows up as slow outliers — a single call in the middle takes
        # 10x its median because the model just got loaded back into VRAM.
        # Use "0" to opt out, or e.g. "1h" for long-running batches.
        self.keep_alive = keep_alive

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model

        # Ollama has no tool_choice knob (as of 0.20.x — it always lets the
        # model decide). Raise rather than silently drop a forced-tool
        # request, matching the spec's no-silent-drop rule
        # (docs/tool-calling.md): the router then falls through to a
        # provider that can honor it.
        if request.tool_choice is not None:
            raise SommBadRequest(
                "ollama does not support tool_choice; omit it or route to a "
                "provider that does (anthropic, openai-compat)"
            )

        # Multi-turn `messages` overrides single-turn `prompt`. Ollama's
        # /api/chat mirrors OpenAI's message shape (assistant `tool_calls`,
        # `role:tool` results), so the OpenAI translator applies unchanged.
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if request.messages is not None:
            messages.extend(_translate_messages_to_openai(request.messages))
        else:
            messages.append({"role": "user", "content": request.prompt})

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                # Thinking models (gemma4, etc.) wrap reasoning in
                # <think>...</think> blocks that consume budget before
                # the actual response. 3x headroom with a 1024 floor
                # prevents empty responses after think-block stripping.
                "num_predict": max(request.max_tokens * 3, 1024),
            },
        }
        if self.enable_think:
            payload["think"] = True
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if request.tools:
            # Ollama 0.4+ accepts OpenAI-shaped tool declarations on
            # /api/chat. The box runs 0.20.x, well past that floor.
            payload["tools"] = [_translate_tool_to_openai(t) for t in request.tools]
        if request.response_format is not None:
            ollama_format = _ollama_response_format(request.response_format)
            if ollama_format is not None:
                payload["format"] = ollama_format

        t0 = time.monotonic()
        with self._client() as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.monotonic() - t0) * 1000)

        message = data.get("message") or {}
        raw_text = message.get("content") or ""
        clean_text = strip_think_block(raw_text)

        tool_calls = _parse_ollama_tool_calls(message.get("tool_calls") or [])
        stop_reason = _normalize_ollama_done_reason(
            data.get("done_reason"), has_tool_calls=bool(tool_calls)
        )

        # Ollama returns prompt_eval_count (tokens_in) + eval_count (tokens_out).
        tokens_in = int(data.get("prompt_eval_count", 0))
        tokens_out = int(data.get("eval_count", 0))

        return SommResponse(
            text=clean_text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            raw=data,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        """Native ollama streaming. Emits SommChunk per line of JSONL output.

        Each chunk is a text delta; `done=True` fires when ollama signals end.
        `<think>` stripping is the library's concern (see SommLLM.stream).
        """
        import json

        model = request.model or self.default_model
        payload = {
            "model": model,
            "messages": [],
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": max(request.max_tokens * 3, 1024),
            },
        }
        if self.enable_think:
            payload["think"] = True
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if request.system:
            payload["messages"].append({"role": "system", "content": request.system})
        payload["messages"].append({"role": "user", "content": request.prompt})

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = data.get("message", {}).get("content", "") or ""
                    done = bool(data.get("done"))
                    if text:
                        yield SommChunk(text=text, done=False)
                    if done:
                        yield SommChunk(text="", done=True)
                        return

    def embed(self, request: SommEmbedRequest) -> SommEmbedResponse:
        """POST /api/embeddings. Single-string input only — batch is the
        caller's concern (loop with caching)."""
        model = request.model or self.default_embed_model
        payload = {"model": model, "prompt": request.text}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        t0 = time.monotonic()
        with self._client() as client:
            resp = client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.monotonic() - t0) * 1000)

        embedding = data.get("embedding") or []
        if not isinstance(embedding, list) or not embedding:
            # Caller surfaces as Outcome.UPSTREAM_ERROR; SommLLM.embed handles.
            raise ValueError(f"ollama returned no embedding: {str(data)[:200]}")
        # Approximate input tokens — ollama's embed endpoint doesn't report
        # them, but we want a non-zero signal in calls.tokens_in.
        from somm_core.parse import estimate_prompt_tokens

        tokens_in = estimate_prompt_tokens(request.text, image_token_cost=0)
        return SommEmbedResponse(
            embedding=embedding,
            model=model,
            tokens_in=tokens_in,
            latency_ms=latency_ms,
            raw=data,
        )

    def health(self) -> ProviderHealth:
        try:
            with self._client() as client:
                r = client.get(f"{self.base_url}/api/tags", timeout=2.0)
                r.raise_for_status()
            return ProviderHealth(available=True, detail=f"{self.base_url} ok")
        except Exception as e:
            return ProviderHealth(available=False, detail=str(e))

    def models(self) -> list[SommModel]:
        try:
            with self._client() as client:
                r = client.get(f"{self.base_url}/api/tags", timeout=5.0)
                r.raise_for_status()
                data = r.json()
        except Exception:
            return []
        out: list[SommModel] = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                out.append(SommModel(name=name))
        return out

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        # 4-char-per-token approximation. Real tokenizers live behind
        # `somm[tokenizers]`. Vision models (llava, etc.) get a rough
        # per-image addend — good enough for latency/cost tracking.
        from somm_core.parse import estimate_prompt_tokens

        return estimate_prompt_tokens(text, image_token_cost=1000)


def _ollama_response_format(response_format: dict) -> dict | str | None:
    if response_format.get("type") == "json_object":
        return "json"
    if response_format.get("type") != "json_schema":
        return None
    schema = response_format.get("json_schema", {}).get("schema")
    return schema if isinstance(schema, dict) else None


def _parse_ollama_tool_calls(raw_tool_calls: list[dict]) -> list[ToolCall]:
    """Ollama tool_calls → somm ToolCall list.

    Unlike OpenAI (which sends `arguments` as a JSON string), Ollama returns
    `function.arguments` already parsed as a dict — so no json.loads and no
    `arguments_raw` repair path. Ollama also doesn't assign tool-call ids, so
    `id` is usually empty; the tool-use loop correlates by name/order.
    """
    out: list[ToolCall] = []
    for entry in raw_tool_calls:
        fn = entry.get("function") or {}
        args = fn.get("arguments")
        out.append(
            ToolCall(
                id=str(entry.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=args if isinstance(args, dict) else {},
            )
        )
    return out


_OLLAMA_DONE_REASON_TO_SOMM: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
}


def _normalize_ollama_done_reason(reason: str | None, *, has_tool_calls: bool) -> str:
    """Ollama `done_reason` → somm-neutral stop_reason.

    Ollama reports `done_reason="stop"` even when the turn is a tool call, so
    presence of tool_calls is the authoritative signal for `tool_use`.
    """
    if has_tool_calls:
        return "tool_use"
    if not reason:
        return ""
    return _OLLAMA_DONE_REASON_TO_SOMM.get(reason, reason)

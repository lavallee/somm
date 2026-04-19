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

from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "gemma4:e4b",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model
        payload = {
            "model": model,
            "messages": [],
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
        if request.system:
            payload["messages"].append({"role": "system", "content": request.system})
        payload["messages"].append({"role": "user", "content": request.prompt})

        t0 = time.monotonic()
        with self._client() as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.monotonic() - t0) * 1000)

        raw_text = data.get("message", {}).get("content", "")
        clean_text = strip_think_block(raw_text)

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

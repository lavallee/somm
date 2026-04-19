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
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

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
        raw_text = choices[0].get("message", {}).get("content", "") or ""
        text = strip_think_block(raw_text)

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
        )

    def _classify_status(self, resp: httpx.Response, model: str) -> None:
        sc = resp.status_code
        if sc == 200:
            return
        body = resp.text[:200]
        if sc in (401, 403):
            raise SommAuthError(f"{self.name} auth failed ({sc}): {body}")
        if sc == 400:
            raise SommBadRequest(f"{self.name} 400 on {model}: {body}")
        if sc == 404:
            raise SommBadRequest(f"{self.name} 404 — model {model!r} not found: {body}")
        if sc == 429:
            retry = _retry_after(resp) or 120.0
            raise SommRateLimited(f"{self.name} 429 on {model}", retry_after_s=retry)
        if 500 <= sc < 600:
            raise SommUpstream5xx(f"{self.name} {sc} on {model}", cooldown_s=30.0)
        raise SommTransientError(f"{self.name} unexpected {sc} on {model}: {body}", cooldown_s=30.0)

    def _raise_body_error(self, err: dict | str, model: str) -> None:
        if isinstance(err, dict):
            msg = err.get("message", "")
            code = err.get("code")
            typ = err.get("type", "")
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

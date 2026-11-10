"""TypeSafe System One: a decision model behind ``SommLLM.decide``.

Not a generation provider. It answers one typed question about a ``state``
and never joins the routing chain. Wire shape, from the vendor's API page
(https://docs.typesafe.ai/api/system-one.md, read 2026-11-09):

- ``POST https://api.typesafe.ai/v1/systemone`` with ``Authorization: Bearer``.
- Body ``{"model", "state", "questions": {name: question}}``; a question is
  ``{"type": "noul" | "choice" | "score", "instructions", "criteria"}``.
- Response ``{"model", "answers": {name: answer}, "usage": {"input_tokens",
  "output_tokens"}}``. A noul answer carries ``noul`` (P(yes)); choice carries
  ``choice``, ``confidence`` and ``probabilities``; score carries ``score``,
  ``confidence``, ``legend`` and ``probabilities`` keyed by level index.

Status codes (https://docs.typesafe.ai/api/errors.md): 400, 413, 415 and 422
are the caller's request; 401 a bad key; 429 rate limit; 529 overloaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from somm.errors import (
    SommBadRequest,
    SommTimeout,
    SommTransientError,
    classify_http_status,
)

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
# Statuses the docs define as a malformed request; classify_http_status would
# read the unfamiliar ones as transient.
_BAD_REQUEST_STATUSES = (400, 413, 415, 422)


@dataclass(slots=True)
class DecisionResponse:
    answer: dict[str, Any]
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)


class TypeSafeProvider:
    name = "typesafe"
    base_url = API_URL
    default_model = DEFAULT_MODEL

    def __init__(self, api_key: str, timeout: float = 20.0, url: str = API_URL) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.url = url

    def decide(self, state: Any, question: dict, model: str | None = None) -> DecisionResponse:
        """Ask one question about ``state``; return its answer and usage."""
        model = model or self.default_model
        body = {"model": model, "state": state, "questions": {"q": question}}
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                )
        except httpx.TimeoutException as e:
            raise SommTimeout(f"typesafe timeout on {model}: {e}", model=model) from e
        except httpx.RequestError as e:
            raise SommTransientError(f"typesafe network error on {model}: {e}", model=model) from e
        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code in _BAD_REQUEST_STATUSES:
            raise SommBadRequest(f"typesafe {resp.status_code} on {model}: {resp.text[:200]}")
        classify_http_status(
            resp.status_code, resp.text, provider="typesafe", model=model, headers=resp.headers
        )
        try:
            data = resp.json()
            answer = data["answers"]["q"]
            usage = data.get("usage") or {}
        except (ValueError, KeyError, TypeError) as e:
            raise SommTransientError(f"typesafe malformed response on {model}: {e}", model=model) from e
        return DecisionResponse(
            answer=answer,
            model=data.get("model") or model,
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
            raw=data,
        )

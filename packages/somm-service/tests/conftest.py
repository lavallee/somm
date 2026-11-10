"""The proxy's test seam: a fake provider handed to ``create_app(providers=...)``.

The proxy dispatches through ``SommLLM``, so tests fake the provider the
client calls, as the library's own tests do, and read the ledger the client
writes. No network, no provider keys.
"""

from __future__ import annotations

import time

import pytest
from somm.providers.base import ProviderHealth, SommChunk, SommResponse


class FakeProvider:
    def __init__(
        self,
        name: str = "anthropic",
        *,
        text: str = "hello back",
        model: str | None = None,
        tokens_in: int = 7,
        tokens_out: int = 3,
        tool_calls: list | None = None,
        raw: dict | None = None,
        stop_reason: str = "",
        error: Exception | None = None,
        delay: float = 0.0,
        base_url: str | None = None,
    ) -> None:
        self.name = name
        self.text, self.model, self.raw = text, model, raw
        self.tokens_in, self.tokens_out = tokens_in, tokens_out
        self.tool_calls = list(tool_calls or [])
        self.stop_reason, self.error, self.delay = stop_reason, error, delay
        self.default_model = "fake-default"
        if base_url is not None:
            self.base_url = base_url
        self.requests: list = []

    def generate(self, request):
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SommResponse(
            text=self.text,
            model=self.model or request.model or self.default_model,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_ms=1,
            raw=self.raw,
            tool_calls=list(self.tool_calls),
            stop_reason=self.stop_reason,
        )

    def stream(self, request):
        # Answers in one piece, as the CLI seats and ChatGPT do; the proxy's
        # streaming routes are exercised through it.
        resp = self.generate(request)
        yield SommChunk(text=resp.text, done=True, model=resp.model, response=resp)

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


@pytest.fixture
def fake_provider():
    return FakeProvider


@pytest.fixture
def ledger():
    """Query the app's ledger after the proxy's clients have flushed their rows."""

    def read(app, sql: str, params: tuple = ()):
        for client in app.state.llm_clients.values():
            client._writer.flush()
        with app.state.repo._open() as conn:
            return conn.execute(sql, params).fetchall()

    return read

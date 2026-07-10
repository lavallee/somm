"""Tests for shared HTTP ingress limit helpers."""

from __future__ import annotations

import pytest
from somm_service.http_limits import PayloadTooLarge, read_bounded_json


class FakeRequest:
    def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self._chunks = chunks

    async def stream(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_read_bounded_json_rejects_streaming_body_over_cap():
    req = FakeRequest([b'{"a"', b": 1}"])

    with pytest.raises(PayloadTooLarge):
        await read_bounded_json(req, max_bytes=5)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_read_bounded_json_checks_content_length_before_streaming():
    req = FakeRequest([], {"content-length": "99"})

    with pytest.raises(PayloadTooLarge):
        await read_bounded_json(req, max_bytes=5)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_read_bounded_json_parses_body_within_cap():
    req = FakeRequest([b'{"a"', b": 1}"])

    assert await read_bounded_json(req, max_bytes=8) == {"a": 1}  # type: ignore[arg-type]

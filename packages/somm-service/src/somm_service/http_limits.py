"""Small request-body guards shared by service JSON ingress paths."""

from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request


class PayloadTooLarge(ValueError):
    """Raised when a request body exceeds the configured byte cap."""


async def read_bounded_json(request: Request, *, max_bytes: int) -> Any:
    """Read and parse a JSON body, rejecting bodies over ``max_bytes``.

    ``Content-Length`` is checked before reading when present. Chunked or
    lengthless requests are read incrementally and rejected as soon as the cap
    is crossed, before JSON parsing.
    """
    max_bytes = max(1, int(max_bytes))
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            if int(raw_length) > max_bytes:
                raise PayloadTooLarge(f"request body exceeds {max_bytes} bytes")
        except ValueError as exc:
            if isinstance(exc, PayloadTooLarge):
                raise

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLarge(f"request body exceeds {max_bytes} bytes")
        chunks.append(chunk)
    body = b"".join(chunks)
    return json.loads(body)

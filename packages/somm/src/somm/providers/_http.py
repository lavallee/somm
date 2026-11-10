"""One reusable HTTP client per provider object.

Opening an ``httpx.Client`` per call paid TCP (and TLS) setup on every call
and every fallback hop. A provider now holds one ``SharedClient``; its
``httpx.Client`` is created on first use and reused, so keep-alive
connections carry the next call. httpx clients are safe to share between
threads for requests.

Timeouts stay per request (``timeout=request_timeout(...)`` on each call), so
the per-call ``timeout=`` and a workload's timeout keep working on a shared
client.

Fork: a child process must not write to sockets it inherited from its
parent. The client remembers the pid that made it; in another process it is
abandoned (not closed, which could disturb the parent's connections) and a
fresh one is made, as ``somm_core.repository`` does for its SQLite handle.
"""

from __future__ import annotations

import os
import threading

import httpx

# Cap on connection setup. A host that has not accepted a connection in 10 s
# is unreachable for practical purposes; the read limit is the configured
# value, since a model can legitimately think for minutes.
CONNECT_TIMEOUT_S = 10.0


def request_timeout(seconds: float) -> httpx.Timeout:
    """Per-request limits from the one configured value: connect within
    ``min(seconds, 10)``; read, write and waiting for a pooled connection get
    the whole value."""
    return httpx.Timeout(seconds, connect=min(seconds, CONNECT_TIMEOUT_S))


class SharedClient:
    """A lazily created ``httpx.Client``, one per process, closable and
    reopenable (a closed holder makes a new client on next use)."""

    def __init__(self) -> None:
        self._client: httpx.Client | None = None
        self._pid: int | None = None
        self._lock = threading.Lock()

    def get(self) -> httpx.Client:
        pid = os.getpid()
        client = self._client
        if client is not None and self._pid == pid:
            return client
        # Build outside the lock (an SSL context takes a few ms), so a fork
        # from another thread never inherits the lock held.
        fresh = httpx.Client()
        with self._lock:
            if self._client is None or self._pid != pid:
                self._client, self._pid = fresh, pid
                return fresh
            winner = self._client
        fresh.close()
        return winner

    def close(self) -> None:
        with self._lock:
            client, pid = self._client, self._pid
            self._client = self._pid = None
        if client is not None and pid == os.getpid():
            client.close()


def shared_client(provider: object) -> httpx.Client:
    """The provider's reusable client. Providers make their ``SharedClient``
    in ``__init__`` so a ``copy.copy`` of the provider shares it; one built
    without that (a subclass that skips ``__init__``) gets one here."""
    holder = getattr(provider, "_http", None)
    if holder is None:
        holder = SharedClient()
        provider._http = holder  # type: ignore[attr-defined]
    return holder.get()


def close_shared_client(provider: object) -> None:
    holder = getattr(provider, "_http", None)
    if holder is not None:
        holder.close()

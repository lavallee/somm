"""Subscription metadata via supported Codex RPC; Codex owns its credentials."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time

from somm_core.adaptive import WeeklyQuota

from somm.providers._cli_bin import resolve_cli_binary


def account_rpc(method: str, *, binary: str | None = None, timeout: float = 15) -> dict:
    if binary is None:
        lookup = resolve_cli_binary("codex", explicit=os.environ.get("SOMM_CODEX_BIN"), env_var="SOMM_CODEX_BIN")
        if not lookup.found:
            raise RuntimeError("Codex CLI is not installed")
        binary = lookup.path
    proc = subprocess.Popen([binary, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    received: queue.Queue = queue.Queue()

    def reader():
        for line in proc.stdout:
            try:
                received.put(json.loads(line))
            except ValueError:
                continue
        received.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout

    def send(payload):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def call(id_, name, params):
        send({"id": id_, "method": name, "params": params})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex account RPC timed out")
            try:
                result = received.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("Codex account RPC timed out") from exc
            if result is None:
                raise RuntimeError("Codex account RPC exited before replying")
            if isinstance(result, dict) and result.get("id") == id_:
                if "error" in result:
                    raise RuntimeError("Codex account RPC failed; run codex login and retry")
                return result.get("result", {})

    try:
        call(1, "initialize", {"clientInfo": {"name": "somm", "version": "1"}})
        send({"method": "initialized"})
        return call(2, method, {"limit": 100} if method == "model/list" else {})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for pipe in (proc.stdin, proc.stdout):
            if pipe:
                pipe.close()
        thread.join(timeout=1)


def weekly_quota(result: dict, *, now: float | None = None) -> WeeklyQuota:
    now = time.time() if now is None else now
    roots = []
    by_id = result.get("rateLimitsByLimitId") or {}
    if isinstance(by_id, dict) and isinstance(by_id.get("codex"), dict):
        roots.append(by_id["codex"])
    top = result.get("rateLimits")
    if isinstance(top, dict) and top.get("limitId") in (None, "codex"):
        roots.append(top)
    windows = []
    for root in roots:
        for name in ("primary", "secondary"):
            window = root.get(name)
            if not isinstance(window, dict) or window.get("windowDurationMins") != 10080:
                continue
            used, reset = window.get("usedPercent"), window.get("resetsAt")
            if (isinstance(used, (int, float)) and not isinstance(used, bool) and 0 <= used <= 100
                    and isinstance(reset, int) and not isinstance(reset, bool) and reset > now):
                windows.append((float(used), reset))
    if not windows:
        return WeeklyQuota(None, observed_at=now, error="weekly_window_unavailable")
    used, reset = max(windows)
    return WeeklyQuota(used, reset, now)


class WeeklyQuotaProbe:
    def __init__(self, *, ttl: float = 60):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._value: WeeklyQuota | None = None

    def read(self) -> WeeklyQuota:
        with self._lock:
            now = time.time()
            if self._value and 0 <= now - self._value.observed_at < self.ttl:
                return self._value
            try:
                self._value = weekly_quota(account_rpc("account/rateLimits/read"), now=now)
            except (OSError, RuntimeError, TimeoutError):
                self._value = WeeklyQuota(None, observed_at=now, error="quota_probe_failed")
            return self._value

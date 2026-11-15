"""What the vendors themselves say is left on a plan.

``somm plans`` meters usage from somm's own telemetry, which only sees calls that
went through somm. A subscription is shared with everything else on the account,
such as an interactive Codex session, so the vendor's own percentage is the
authority whenever one can be read. Readers here never make a call that spends
quota, and never refresh or rewrite anyone's credentials:

- **minimax**: ``GET /v1/token_plan/remains`` with ``MINIMAX_API_KEY``.
- **codex-cli**: the newest ``rate_limits`` event in Codex's own session logs
  (``$CODEX_HOME/sessions``). Passive and credential-free. It is only as fresh
  as the last Codex turn on this host, so every reading carries ``observed_at``.

Each reading is ``{provider, plan, window, used_pct, resets_at, observed_at,
source}`` plus provider extras. A reader that cannot see anything returns ``[]``.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

CODEX_FILES_TO_SCAN = 20            # newest session logs, by mtime
CODEX_TAIL_BYTES = 2 * 1024 * 1024  # rate_limits rides on every token_count event


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts else None


def _window_label(minutes: int | None) -> str:
    if not minutes:
        return "?"
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def minimax_readings(env: dict | None = None) -> list[dict]:
    from somm.providers.minimax import plan_remains

    key = (env if env is not None else os.environ).get("MINIMAX_API_KEY")
    live = plan_remains(key) if key else None
    if not live:
        return []
    now = datetime.now(UTC).isoformat()
    return [
        {"provider": "minimax", "plan": None, "window": window, "used_pct": float(pct),
         "resets_at": None, "observed_at": now, "source": "minimax token_plan/remains"}
        for window, pct in (("5h", live["interval_used_pct"]), ("7d", live["weekly_used_pct"]))
    ]


def _find_rate_limits(event: dict) -> dict | None:
    payload = event.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("rate_limits"), dict):
        return payload["rate_limits"]
    return None


def _last_rate_limits(path: Path) -> tuple[dict, dict] | None:
    """(event, rate_limits) for the newest rate_limits event in one session log."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - CODEX_TAIL_BYTES))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if '"rate_limits"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue            # the first line of a tail read may be partial
        limits = _find_rate_limits(event)
        if limits:
            return event, limits
    return None


def codex_readings(codex_home: str | os.PathLike | None = None) -> list[dict]:
    home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        files = sorted((home / "sessions").rglob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:CODEX_FILES_TO_SCAN]
    except OSError:
        return []
    for path in files:
        found = _last_rate_limits(path)
        if not found:
            continue
        event, limits = found
        observed = event.get("timestamp") or _iso(path.stat().st_mtime)
        credits = limits.get("credits") if isinstance(limits.get("credits"), dict) else None
        out = []
        for slot in ("primary", "secondary"):
            w = limits.get(slot)
            if not isinstance(w, dict) or w.get("used_percent") is None:
                continue
            out.append({
                "provider": "codex-cli", "plan": limits.get("plan_type"),
                "window": _window_label(w.get("window_minutes")),
                "used_pct": float(w["used_percent"]),
                "resets_at": _iso(w.get("resets_at")),
                "observed_at": observed, "source": "codex session log",
                "limit_reached": limits.get("rate_limit_reached_type"),
                "credits_balance": (credits or {}).get("balance"),
            })
        return out
    return []


def vendor_readings(env: dict | None = None) -> list[dict]:
    """Every vendor-reported window this host can see. Never raises."""
    out: list[dict] = []
    for reader in (lambda: minimax_readings(env), codex_readings):
        try:
            out += reader()
        except Exception:  # noqa: BLE001 - one vendor's change must not hide the others
            continue
    return out

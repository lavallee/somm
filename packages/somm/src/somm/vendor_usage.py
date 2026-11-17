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

``balances()`` reads prepaid pay-as-you-go accounts (DeepSeek, OpenRouter,
RunPod, Vast.ai) with each account's own API key, read-only. TypeSafe (Jev) and
SubQ publish no balance API: for them it reports whether the key works
(``balance`` is None, ``key_ok`` True), and ``somm_core.plans.credit_refusals``
reports calls refused for lack of credit.
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


# --- prepaid balances ---------------------------------------------------------------
# Pay-as-you-go accounts stop when the balance hits zero, which no metered
# budget sees coming. Each reader is one read-only GET/POST with the account's own
# API key from the environment, and returns [] when the key is absent.

def _request(url: str, key: str, *, body: dict | None = None, timeout: float = 10.0) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                 "Content-Type": "application/json", "User-Agent": "somm"},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs
        return json.loads(resp.read(1024 * 1024))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def deepseek_balance(env: dict) -> list[dict]:
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        return []
    data = _request("https://api.deepseek.com/user/balance", key)
    out = []
    for info in data.get("balance_infos") or []:
        out.append({
            "provider": "deepseek", "currency": info.get("currency"),
            "balance": float(info.get("total_balance") or 0),
            "granted": float(info.get("granted_balance") or 0),
            "topped_up": float(info.get("topped_up_balance") or 0),
            "available": bool(data.get("is_available")),
            "observed_at": _now(), "source": "deepseek /user/balance"})
    return out


def openrouter_balance(env: dict) -> list[dict]:
    key = env.get("OPENROUTER_API_KEY")
    if not key:
        return []
    data = (_request("https://openrouter.ai/api/v1/credits", key).get("data") or {})
    total, used = float(data.get("total_credits") or 0), float(data.get("total_usage") or 0)
    return [{"provider": "openrouter", "currency": "USD", "balance": round(total - used, 4),
             "purchased": total, "used": used, "observed_at": _now(),
             "source": "openrouter /api/v1/credits"}]


RUNPOD_QUERY = "query { myself { clientBalance currentSpendPerHr spendLimit } }"


def runpod_balance(env: dict) -> list[dict]:
    key = env.get("RUNPOD_API_KEY")
    if not key:
        return []
    me = ((_request("https://api.runpod.io/graphql", key, body={"query": RUNPOD_QUERY})
           .get("data") or {}).get("myself") or {})
    if "clientBalance" not in me:
        raise ValueError("runpod: no balance in response")
    return [{"provider": "runpod", "currency": "USD", "balance": float(me["clientBalance"] or 0),
             "spend_per_hr": float(me.get("currentSpendPerHr") or 0),
             "spend_limit": me.get("spendLimit"), "observed_at": _now(),
             "source": "runpod graphql myself"}]


def vast_balance(env: dict) -> list[dict]:
    key = env.get("VAST_API_KEY")
    if not key:
        return []
    user = _request("https://console.vast.ai/api/v0/users/current", key)
    # The response also carries the account's email and ssh key: keep only money.
    if "balance" not in user and "credit" not in user:
        raise ValueError("vast: no balance in response")
    balance = user.get("balance") if user.get("balance") is not None else user.get("credit")
    return [{"provider": "vast", "currency": "USD", "balance": float(balance or 0),
             "observed_at": _now(), "source": "vast /api/v0/users/current"}]


# Vendors with no balance API. A free, authenticated `GET /v1/models` proves the
# key works; credit exhaustion only shows up as refused calls (credit_refusals).
ACCESS_ONLY = {
    "typesafe": ("TYPESAFE_API_KEY", "https://api.typesafe.ai/v1/models"),
    "subq": ("SUBQ_API_KEY", "https://api.subq.ai/v1/models"),
}


def _access_reader(provider: str):
    env_key, url = ACCESS_ONLY[provider]

    def read(env: dict) -> list[dict]:
        key = env.get(env_key)
        if not key:
            return []
        data = _request(url, key)
        models = [m.get("id") for m in data.get("data") or [] if isinstance(m, dict)]
        return [{"provider": provider, "currency": None, "balance": None, "key_ok": True,
                 "models": [m for m in models if m][:20], "observed_at": _now(),
                 "source": f"{provider} GET /v1/models (no balance API)"}]

    read.__name__ = f"{provider}_balance"
    return read


BALANCE_READERS = (deepseek_balance, openrouter_balance, runpod_balance, vast_balance,
                   _access_reader("typesafe"), _access_reader("subq"))


def balances(env: dict | None = None) -> list[dict]:
    """Every prepaid balance this host has a key for. A failed read is reported,
    not hidden, so "no key" and "the API changed" stay distinguishable."""
    env = env if env is not None else dict(os.environ)
    out: list[dict] = []
    for reader in BALANCE_READERS:
        try:
            out += reader(env)
        except Exception as e:  # noqa: BLE001 - one account's failure must not hide the others
            out.append({"provider": reader.__name__.removesuffix("_balance"),
                        "error": f"{type(e).__name__}: {e}"[:200], "observed_at": _now()})
    return out


def vendor_readings(env: dict | None = None) -> list[dict]:
    """Every vendor-reported window this host can see. Never raises."""
    out: list[dict] = []
    for reader in (lambda: minimax_readings(env), codex_readings):
        try:
            out += reader()
        except Exception:  # noqa: BLE001 - one vendor's change must not hide the others
            continue
    return out

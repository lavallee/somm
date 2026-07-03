"""Billing plans — PAYG vs metered-quota accounting per provider.

Two fundamentally different billing shapes hide behind "provider":

- **payg** — pay-as-you-go per token. `calls.cost_usd` is real marginal
  dollars; the constraint is your wallet.
- **metered** — a subscription plan with recurring usage limits (a
  coding plan, a CLI seat). Marginal dollars are ~0 inside the quota;
  `cost_usd` is *notional* (list-price equivalent). The scarce resource
  is window headroom, and the operational question is **pacing**: at the
  current burn rate, does the quota survive the window?
- **free** — local or genuinely free (ollama, `:free` rosters).

Plans are declared machine-wide in ``~/.somm/plans.toml`` because a
metered quota is shared by every project using the same account/key —
pacing computed against one project's telemetry alone would understate
burn. Fleet-wide usage comes from the project registry (see
somm_core.registry).

Example ``~/.somm/plans.toml``::

    [minimax]
    mode = "metered"
    plan = "coding-pro"
    soft_target_pct = 80    # warn/deprioritize beyond this pace
    enforce = false         # true: hard-skip provider when a limit is exhausted

    [[minimax.limits]]
    window = "month"        # calendar month …
    anchor_day = 12         # … resetting on the 12th
    quota = 40.0
    unit = "usd_equiv"      # requests | tokens_in | tokens_out | tokens_total | usd_equiv

    [claude-cli]
    mode = "metered"
    plan = "max"
    [[claude-cli.limits]]
    window = "5h"           # rolling window
    quota = 200
    unit = "requests"

    [gemini]
    mode = "payg"

Providers not listed default to: ollama → free, claude-cli/codex-cli →
metered (unlabelled), everything else → payg.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

VALID_MODES = ("payg", "metered", "free")
VALID_UNITS = ("requests", "tokens_in", "tokens_out", "tokens_total", "usd_equiv")

_WINDOW_RE = re.compile(r"^(\d+)([hdw])$")

# Providers with a default mode when plans.toml doesn't mention them.
_DEFAULT_MODES = {
    "ollama": "free",
    "claude-cli": "metered",
    "codex-cli": "metered",
}


@dataclass(slots=True)
class PlanLimit:
    window: str  # "5h" | "3d" | "1w" (rolling) or "month" (calendar)
    quota: float
    unit: str = "requests"
    anchor_day: int = 1  # calendar-month reset day (1-28)

    def is_rolling(self) -> bool:
        return self.window != "month"

    def window_seconds(self) -> float:
        if self.window == "month":
            raise ValueError("calendar window has no fixed size")
        m = _WINDOW_RE.match(self.window)
        if not m:
            raise ValueError(f"bad window {self.window!r} (use e.g. '5h', '7d', '1w', 'month')")
        n, u = int(m.group(1)), m.group(2)
        return n * {"h": 3600, "d": 86400, "w": 7 * 86400}[u]

    def bounds(self, now: datetime) -> tuple[datetime, datetime]:
        """(start, end) of the window containing `now`."""
        if self.is_rolling():
            span = timedelta(seconds=self.window_seconds())
            return (now - span, now)
        day = min(max(self.anchor_day, 1), 28)
        anchor = now.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
        if now < anchor:
            # window began on the anchor day of the PREVIOUS month
            start = (anchor.replace(day=1) - timedelta(days=1)).replace(day=day)
            end = anchor
        else:
            start = anchor
            end = (anchor + timedelta(days=32)).replace(day=day)
        return (start, end)


@dataclass(slots=True)
class Plan:
    provider: str
    mode: str = "payg"
    name: str = ""
    soft_target_pct: float = 80.0
    enforce: bool = False
    limits: list[PlanLimit] = field(default_factory=list)


@dataclass(slots=True)
class LimitStatus:
    """One limit's usage inside its current window."""

    provider: str
    plan_name: str
    limit: PlanLimit
    used: float
    window_start: datetime
    window_end: datetime
    soft_target_pct: float = 80.0

    @property
    def used_pct(self) -> float:
        return 100.0 * self.used / self.limit.quota if self.limit.quota else 0.0

    @property
    def elapsed_pct(self) -> float:
        """How much of the window has passed. Rolling windows are always
        fully elapsed — usage there is a straight fraction of quota."""
        if self.limit.is_rolling():
            return 100.0
        total = (self.window_end - self.window_start).total_seconds()
        gone = (datetime.now(UTC) - self.window_start).total_seconds()
        return max(0.0, min(100.0, 100.0 * gone / total)) if total else 100.0

    @property
    def pace_ratio(self) -> float:
        """>1 = burning faster than the window is passing."""
        e = self.elapsed_pct
        return (self.used_pct / e) if e else 0.0

    @property
    def projected_pct(self) -> float:
        """Straight-line projection of used_pct at window end."""
        return self.pace_ratio * 100.0

    @property
    def state(self) -> str:
        """'exhausted' | 'over_pace' | 'ok'.

        Calendar windows go over-pace when burn outruns the calendar AND
        usage is past the soft target (an early-window burst with plenty
        of quota left is fine). Rolling windows have no calendar to
        outrun — utilization past the soft target IS the over-pace
        signal there."""
        if self.used_pct >= 100.0:
            return "exhausted"
        if self.limit.is_rolling():
            return "over_pace" if self.used_pct >= self.soft_target_pct else "ok"
        if self.pace_ratio > 1.0 and self.used_pct >= self.soft_target_pct:
            return "over_pace"
        return "ok"


def plans_path() -> Path:
    env = os.environ.get("SOMM_PLANS_PATH")
    return Path(env) if env else Path.home() / ".somm" / "plans.toml"


def load_plans(path: Path | None = None) -> dict[str, Plan]:
    """Parse plans.toml → {provider: Plan}. Missing file → {} (defaults
    still apply via plan_for). Malformed entries raise ValueError with the
    offending provider named — a silently ignored quota is worse than a
    loud config error."""
    path = path or plans_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    out: dict[str, Plan] = {}
    for provider, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"plans.toml [{provider}]: expected a table")
        mode = spec.get("mode", "payg")
        if mode not in VALID_MODES:
            raise ValueError(f"plans.toml [{provider}]: mode {mode!r} not in {VALID_MODES}")
        limits = []
        for lim in spec.get("limits", []):
            unit = lim.get("unit", "requests")
            if unit not in VALID_UNITS:
                raise ValueError(f"plans.toml [{provider}]: unit {unit!r} not in {VALID_UNITS}")
            limit = PlanLimit(
                window=str(lim.get("window", "month")),
                quota=float(lim["quota"]),
                unit=unit,
                anchor_day=int(lim.get("anchor_day", 1)),
            )
            if limit.is_rolling():
                limit.window_seconds()  # validate format now, loudly
            limits.append(limit)
        out[provider] = Plan(
            provider=provider,
            mode=mode,
            name=str(spec.get("plan", "")),
            soft_target_pct=float(spec.get("soft_target_pct", 80.0)),
            enforce=bool(spec.get("enforce", False)),
            limits=limits,
        )
    return out


def plan_for(provider: str, plans: dict[str, Plan]) -> Plan:
    """Declared plan, or the built-in default for the provider."""
    if provider in plans:
        return plans[provider]
    return Plan(provider=provider, mode=_DEFAULT_MODES.get(provider, "payg"))


_UNIT_SQL = {
    "requests": "COUNT(*)",
    "tokens_in": "COALESCE(SUM(tokens_in), 0)",
    "tokens_out": "COALESCE(SUM(tokens_out), 0)",
    "tokens_total": "COALESCE(SUM(tokens_in + tokens_out), 0)",
    "usd_equiv": "COALESCE(SUM(cost_usd), 0)",
}


def usage_in_window(
    db_paths: list[Path],
    provider: str,
    limit: PlanLimit,
    now: datetime | None = None,
) -> float:
    """Sum a limit's unit for one provider across the given DBs.

    Read-only; a missing or locked DB contributes 0 rather than failing —
    pacing is advisory and must never take the call path down."""
    now = now or datetime.now(UTC)
    start, _end = limit.bounds(now)
    total = 0.0
    for db in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                row = conn.execute(
                    f"SELECT {_UNIT_SQL[limit.unit]} FROM calls "
                    f"WHERE provider = ? AND ts >= ?",
                    (provider, start.isoformat()),
                ).fetchone()
                total += float(row[0] or 0)
            finally:
                conn.close()
        except Exception:
            continue
    return total


def limit_statuses(
    db_paths: list[Path],
    plans: dict[str, Plan],
    providers: list[str] | None = None,
    now: datetime | None = None,
) -> list[LimitStatus]:
    """Current-window status for every limit of every metered plan."""
    now = now or datetime.now(UTC)
    out: list[LimitStatus] = []
    for provider, plan in plans.items():
        if providers is not None and provider not in providers:
            continue
        if plan.mode != "metered":
            continue
        for limit in plan.limits:
            start, end = limit.bounds(now)
            used = usage_in_window(db_paths, provider, limit, now)
            out.append(
                LimitStatus(
                    provider=provider,
                    plan_name=plan.name,
                    limit=limit,
                    used=used,
                    window_start=start,
                    window_end=end,
                    soft_target_pct=plan.soft_target_pct,
                )
            )
    return out

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
    learned: bool = False
    source: str = ""
    last_observed: str = ""
    n_events: int = 0

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
    catalog_ref: str = ""  # "provider/plan-id" when limits came from the catalog
    # Subscription price (metered plans). Lets reporting show the value
    # multiple: notional list-price consumed ÷ what the plan costs.
    price_usd_month: float = 0.0


@dataclass(slots=True)
class CatalogEntry:
    """A known commercial plan and its published limits.

    The catalog (data/plan_catalog.toml, shipped in the somm-core wheel)
    is hand-curated from vendor pages — plan limits are marketing copy,
    not an API, so every entry carries its source URL and the date a
    human last verified it. `somm plans` warns when a referenced entry
    goes stale; re-verify against `source` and bump `last_verified`."""

    provider: str
    plan_id: str
    display: str = ""
    source: str = ""
    last_verified: str = ""  # ISO date
    notes: str = ""
    limits: list[PlanLimit] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.plan_id}"

    def age_days(self, now: datetime | None = None) -> int | None:
        if not self.last_verified:
            return None
        try:
            verified = datetime.fromisoformat(self.last_verified).replace(tzinfo=UTC)
        except ValueError:
            return None
        return max(0, ((now or datetime.now(UTC)) - verified).days)


@dataclass(slots=True)
class LimitStatus:
    """One limit's usage inside its current window.

    For metered plans the limit is a vendor quota (used/quota are
    notional units); for payg plans it's a self-imposed **budget** in
    real dollars — same pacing math, different meaning of exhaustion
    (vendor cuts you off vs. you chose a ceiling)."""

    provider: str
    plan_name: str
    limit: PlanLimit
    used: float
    window_start: datetime
    window_end: datetime
    soft_target_pct: float = 80.0
    mode: str = "metered"

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


def _parse_limit(provider: str, lim: dict) -> PlanLimit:
    unit = lim.get("unit", "requests")
    if unit not in VALID_UNITS:
        raise ValueError(f"[{provider}]: unit {unit!r} not in {VALID_UNITS}")
    limit = PlanLimit(
        window=str(lim.get("window", "month")),
        quota=float(lim["quota"]),
        unit=unit,
        anchor_day=int(lim.get("anchor_day", 1)),
        learned=bool(lim.get("learned", False)),
        source=str(lim.get("source", "")),
        last_observed=str(lim.get("last_observed", "")),
        n_events=int(lim.get("n_events", 0) or 0),
    )
    if limit.is_rolling():
        limit.window_seconds()  # validate format now, loudly
    return limit


def load_catalog(path: Path | None = None) -> dict[str, CatalogEntry]:
    """Load the bundled plan catalog → {"provider/plan-id": CatalogEntry}.

    Missing/broken catalog → {} (the catalog is a convenience layer;
    explicit limits in plans.toml never depend on it)."""
    try:
        if path is not None:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        else:
            from importlib import resources

            raw = (
                resources.files("somm_core") / "data" / "plan_catalog.toml"
            ).read_bytes()
            data = tomllib.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    out: dict[str, CatalogEntry] = {}
    for provider, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for plan_id, spec in entries.items():
            if not isinstance(spec, dict):
                continue
            try:
                limits = [_parse_limit(provider, lim) for lim in spec.get("limits", [])]
                lv = spec.get("last_verified", "")
                entry = CatalogEntry(
                    provider=provider,
                    plan_id=plan_id,
                    display=str(spec.get("display", plan_id)),
                    source=str(spec.get("source", "")),
                    last_verified=str(lv),
                    notes=str(spec.get("notes", "")),
                    limits=limits,
                )
            except Exception:
                continue
            out[entry.key] = entry
    return out


def load_plans(
    path: Path | None = None,
    catalog: dict[str, CatalogEntry] | None = None,
) -> dict[str, Plan]:
    """Parse plans.toml → {provider: Plan}. Missing file → {} (defaults
    still apply via plan_for). Malformed entries raise ValueError with the
    offending provider named — a silently ignored quota is worse than a
    loud config error.

    A plan may set ``catalog = "<plan-id>"`` to inherit its limits from
    the bundled plan catalog (resolved as ``<provider>/<plan-id>``)
    instead of spelling them out. Explicit ``[[limits]]`` always win over
    the catalog. Referencing a plan-id the catalog doesn't know raises —
    a typo here would silently disable pacing."""
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
        try:
            limits = [_parse_limit(provider, lim) for lim in spec.get("limits", [])]
        except ValueError as exc:
            raise ValueError(f"plans.toml {exc}") from exc

        catalog_ref = ""
        cat_id = spec.get("catalog", "")
        name = str(spec.get("plan", ""))
        if cat_id:
            if catalog is None:
                catalog = load_catalog()
            key = f"{provider}/{cat_id}"
            entry = catalog.get(key)
            if entry is None:
                known = sorted(k for k in catalog if k.startswith(f"{provider}/"))
                raise ValueError(
                    f"plans.toml [{provider}]: catalog = {cat_id!r} not found"
                    + (f"; known for {provider}: {', '.join(known)}" if known else "")
                )
            catalog_ref = key
            if not limits:
                limits = list(entry.limits)
            if not name:
                name = entry.display

        out[provider] = Plan(
            provider=provider,
            mode=mode,
            name=name,
            soft_target_pct=float(spec.get("soft_target_pct", 80.0)),
            enforce=bool(spec.get("enforce", False)),
            limits=limits,
            catalog_ref=catalog_ref,
            price_usd_month=float(spec.get("price", 0.0)),
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


def _provider_aliases(provider: str) -> tuple[str, ...]:
    """Match historical spellings ('claude_cli' rows predate 'claude-cli')."""
    return tuple({provider, provider.replace("-", "_"), provider.replace("_", "-")})


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
    start, end = limit.bounds(now)
    aliases = _provider_aliases(provider)
    marks = ",".join("?" for _ in aliases)
    total = 0.0
    for db in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                row = conn.execute(
                    f"SELECT {_UNIT_SQL[limit.unit]} FROM calls "
                    f"WHERE provider IN ({marks}) AND ts >= ? AND ts <= ?",
                    (*aliases, start.isoformat(), end.isoformat()),
                ).fetchone()
                total += float(row[0] or 0)
            finally:
                conn.close()
        except Exception:
            continue
    return total


@dataclass(slots=True)
class BurnRate:
    """Real-dollar spend velocity for one payg provider."""

    provider: str
    spend_1d: float
    spend_7d: float
    spend_30d: float

    @property
    def per_day(self) -> float:
        """Smoothed daily rate (7-day average)."""
        return self.spend_7d / 7.0

    @property
    def projected_month(self) -> float:
        return self.per_day * 30.0


def payg_burn_rates(
    db_paths: list[Path],
    plans: dict[str, Plan],
    now: datetime | None = None,
) -> list[BurnRate]:
    """Spend velocity per payg provider across the fleet.

    PAYG has no vendor window — the meaningful number is rate: $/day and
    where that lands by month-end. Providers with zero spend in 30 days
    are omitted."""
    now = now or datetime.now(UTC)
    out: list[BurnRate] = []
    for provider, plan in plans.items():
        if plan.mode != "payg":
            continue
        spends = []
        for days in (1, 7, 30):
            lim = PlanLimit(window=f"{days}d", quota=1, unit="usd_equiv")
            spends.append(usage_in_window(db_paths, provider, lim, now))
        if spends[2] <= 0:
            continue
        out.append(BurnRate(provider, *spends))
    out.sort(key=lambda b: -b.spend_30d)
    return out


def recent_ok_calls(
    db_paths: list[Path],
    provider: str,
    hours: float = 24.0,
    now: datetime | None = None,
) -> int:
    """Successful calls for a provider in the trailing window — used to
    tell 'quota exhausted' apart from 'declared quota is stale: usage
    exceeds it but the vendor is still serving us' (limits get raised
    and reset without notice)."""
    now = now or datetime.now(UTC)
    since = (now - timedelta(hours=hours)).isoformat()
    aliases = _provider_aliases(provider)
    marks = ",".join("?" for _ in aliases)
    total = 0
    for db in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM calls WHERE provider IN ({marks}) "
                    f"AND ts >= ? AND outcome = 'ok'",
                    (*aliases, since),
                ).fetchone()
                total += int(row[0] or 0)
            finally:
                conn.close()
        except Exception:
            continue
    return total


@dataclass(slots=True)
class ObservedCeiling:
    """A quota estimate inferred from your own telemetry.

    Vendors no longer publish complete numeric limits (as of mid-2026
    all major metered plans describe quotas qualitatively or in
    multipliers), but every quota-429 is ground truth: at that moment,
    trailing-window usage ≈ the binding ceiling. The estimate is the
    median across observed events — noisy, but empirical and
    self-updating, which beats stale marketing copy."""

    provider: str
    window: str
    unit: str
    estimate: float
    n_events: int
    last_event: str  # ISO timestamp of the most recent quota error


@dataclass(slots=True)
class LearnedLimitUpdate:
    provider: str
    window: str
    unit: str
    old_quota: float | None
    new_quota: float
    n_events: int
    last_event: str
    action: str  # "added" | "updated" | "unchanged"


_QUOTA_ERROR_WHERE = (
    "outcome != 'ok' AND ("
    "error_kind LIKE '%RateLimit%' "
    "OR lower(coalesce(error_detail,'')) LIKE '%rate_limit%' "
    "OR lower(coalesce(error_detail,'')) LIKE '%spend limit%' "
    "OR lower(coalesce(error_detail,'')) LIKE '%usage limit%' "
    "OR lower(coalesce(error_detail,'')) LIKE '%quota%'"
    ")"
)


def observed_ceilings(
    db_paths: list[Path],
    provider: str,
    windows: tuple[str, ...] = ("5h", "7d"),
    units: tuple[str, ...] = ("tokens_total", "requests"),
    lookback_days: int = 90,
    max_events: int = 24,
    now: datetime | None = None,
) -> list[ObservedCeiling]:
    """Estimate quota ceilings from quota-429 moments in the fleet.

    For each observed quota error, sum trailing-window usage across all
    DBs ending at that moment; the median over events estimates the
    limit. Events within one window-span of a previous event are
    collapsed (a burst of 429s is one ceiling observation, not many)."""
    now = now or datetime.now(UTC)
    aliases = _provider_aliases(provider)
    marks = ",".join("?" for _ in aliases)
    cutoff = (now - timedelta(days=lookback_days)).isoformat()

    events: list[datetime] = []
    for db in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                rows = conn.execute(
                    f"SELECT ts FROM calls WHERE provider IN ({marks}) "
                    f"AND ts >= ? AND {_QUOTA_ERROR_WHERE} "
                    f"ORDER BY ts DESC LIMIT ?",
                    (*aliases, cutoff, max_events),
                ).fetchall()
                for (ts,) in rows:
                    try:
                        dt = datetime.fromisoformat(ts)
                        events.append(dt if dt.tzinfo else dt.replace(tzinfo=UTC))
                    except ValueError:
                        continue
            finally:
                conn.close()
        except Exception:
            continue
    if not events:
        return []
    events.sort(reverse=True)

    out: list[ObservedCeiling] = []
    for window in windows:
        lim_probe = PlanLimit(window=window, quota=1)
        if not lim_probe.is_rolling():
            continue
        span = timedelta(seconds=lim_probe.window_seconds())
        # Collapse 429 bursts: keep events at least one span apart.
        distinct: list[datetime] = []
        for ev in events:
            if not distinct or (distinct[-1] - ev) >= span:
                distinct.append(ev)
            if len(distinct) >= max_events:
                break
        for unit in units:
            samples = [
                usage_in_window(db_paths, provider, PlanLimit(window=window, quota=1, unit=unit), now=ev)
                for ev in distinct
            ]
            samples = sorted(s for s in samples if s > 0)
            if not samples:
                continue
            median = samples[len(samples) // 2]
            out.append(
                ObservedCeiling(
                    provider=provider,
                    window=window,
                    unit=unit,
                    estimate=median,
                    n_events=len(samples),
                    last_event=events[0].isoformat(),
                )
            )
    return out


def providers_with_quota_errors(
    db_paths: list[Path],
    lookback_days: int = 90,
    now: datetime | None = None,
) -> list[str]:
    """Providers with recent quota/rate-limit-looking failures in the fleet."""

    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=lookback_days)).isoformat()
    out: set[str] = set()
    for db in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT provider FROM calls WHERE ts >= ? AND {_QUOTA_ERROR_WHERE}",
                    (cutoff,),
                ).fetchall()
                out.update(str(row[0]) for row in rows if row[0])
            finally:
                conn.close()
        except Exception:
            continue
    return sorted(out)


def learn_observed_limits(
    db_paths: list[Path],
    plans: dict[str, Plan],
    *,
    path: Path | None = None,
    dry_run: bool = False,
    min_events: int = 1,
    drift_threshold: float = 0.05,
) -> list[LearnedLimitUpdate]:
    """Fold observed quota ceilings into the local plans file."""

    path = path or plans_path()
    updates: list[LearnedLimitUpdate] = []
    candidate_providers = set(plans)
    candidate_providers.update(providers_with_quota_errors(db_paths))
    for provider in sorted(candidate_providers):
        plan = plans.get(provider) or plan_for(provider, plans)
        if plan.mode != "metered":
            continue
        if provider not in plans:
            plans[provider] = plan
        ceilings = observed_ceilings(db_paths, provider)
        for ceiling in ceilings:
            if ceiling.n_events < min_events:
                continue
            existing = _matching_limit(plan, ceiling.window, ceiling.unit)
            old_quota = existing.quota if existing is not None else None
            action = "unchanged"
            if existing is None:
                plan.limits.append(
                    PlanLimit(
                        window=ceiling.window,
                        quota=ceiling.estimate,
                        unit=ceiling.unit,
                        learned=True,
                        source="observed_429",
                        last_observed=ceiling.last_event,
                        n_events=ceiling.n_events,
                    )
                )
                action = "added"
            else:
                drift = (
                    abs(ceiling.estimate - existing.quota) / existing.quota
                    if existing.quota
                    else 1.0
                )
                if existing.learned or drift >= drift_threshold:
                    existing.quota = ceiling.estimate
                    existing.learned = True
                    existing.source = "observed_429"
                    existing.last_observed = ceiling.last_event
                    existing.n_events = ceiling.n_events
                    action = "updated"
            updates.append(
                LearnedLimitUpdate(
                    provider=provider,
                    window=ceiling.window,
                    unit=ceiling.unit,
                    old_quota=old_quota,
                    new_quota=ceiling.estimate,
                    n_events=ceiling.n_events,
                    last_event=ceiling.last_event,
                    action=action,
                )
            )
    if not dry_run and any(u.action in {"added", "updated"} for u in updates):
        write_plans(plans, path=path)
    return updates


def write_plans(plans: dict[str, Plan], path: Path | None = None) -> None:
    """Write a minimal machine-owned plans.toml preserving plan semantics."""

    path = path or plans_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_plans_toml(plans), encoding="utf-8")
    path.chmod(0o600)


def _matching_limit(plan: Plan, window: str, unit: str) -> PlanLimit | None:
    for limit in plan.limits:
        if limit.window == window and limit.unit == unit:
            return limit
    return None


def _format_plans_toml(plans: dict[str, Plan]) -> str:
    lines = [
        "# Machine-updated by somm. Comments from the previous file are not preserved.",
        "",
    ]
    for provider, plan in sorted(plans.items()):
        lines.append(f"[{_toml_key(provider)}]")
        lines.append(f'mode = "{_toml_str(plan.mode)}"')
        if plan.name:
            lines.append(f'plan = "{_toml_str(plan.name)}"')
        if plan.soft_target_pct != 80.0:
            lines.append(f"soft_target_pct = {_toml_float(plan.soft_target_pct)}")
        if plan.enforce:
            lines.append("enforce = true")
        if plan.catalog_ref:
            _provider, catalog = plan.catalog_ref.split("/", 1)
            lines.append(f'catalog = "{_toml_str(catalog)}"')
        if plan.price_usd_month:
            lines.append(f"price = {_toml_float(plan.price_usd_month)}")
        for limit in plan.limits:
            lines.append("")
            lines.append(f"[[{_toml_key(provider)}.limits]]")
            lines.append(f'window = "{_toml_str(limit.window)}"')
            lines.append(f"quota = {_toml_float(limit.quota)}")
            if limit.unit != "requests":
                lines.append(f'unit = "{_toml_str(limit.unit)}"')
            if limit.anchor_day != 1:
                lines.append(f"anchor_day = {limit.anchor_day}")
            if limit.learned:
                lines.append("learned = true")
            if limit.source:
                lines.append(f'source = "{_toml_str(limit.source)}"')
            if limit.last_observed:
                lines.append(f'last_observed = "{_toml_str(limit.last_observed)}"')
            if limit.n_events:
                lines.append(f"n_events = {int(limit.n_events)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return f'"{_toml_str(value)}"'


def _toml_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


def limit_statuses(
    db_paths: list[Path],
    plans: dict[str, Plan],
    providers: list[str] | None = None,
    now: datetime | None = None,
) -> list[LimitStatus]:
    """Current-window status for every limit of every plan that has any.

    Metered plans' limits are vendor quotas; payg plans' limits are
    self-imposed budgets (LiteLLM-style: a max spend over a duration).
    Free plans never have paceable limits."""
    now = now or datetime.now(UTC)
    out: list[LimitStatus] = []
    for provider, plan in plans.items():
        if providers is not None and provider not in providers:
            continue
        if plan.mode == "free" or not plan.limits:
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
                    mode=plan.mode,
                )
            )
    return out

"""Router + ProviderHealthTracker.

Routing rules:
  1. Providers tried in configured preference order.
  2. Each (provider, model) has its own cooldown entry in provider_health.
  3. On transient failure, cool the (provider, model) with per-error backoff.
  4. Router skips cooled entries. If ALL configured providers are cooled,
     Router fails fast unless the caller opted into wait-with-deadline.
  5. Fatal errors raise immediately — no fallback.

State is persisted in SQLite (`provider_health`) so cooldowns survive
process restarts — an overnight flaky free-tier doesn't get re-hammered
at dawn.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from somm_core.repository import Repository

from somm.capabilities import provider_can_serve
from somm.errors import (
    SommFatalError,
    SommNoCapableProvider,
    SommProvidersExhausted,
    SommTransientError,
)

if TYPE_CHECKING:
    from somm.providers.base import SommProvider, SommRequest, SommResponse


@dataclass(slots=True)
class HealthRecord:
    provider: str
    model: str
    last_ok_at: datetime | None
    cooldown_until: datetime | None
    consecutive_failures: int

    def is_cooling(self, now: datetime | None = None) -> bool:
        if self.cooldown_until is None:
            return False
        now = now or datetime.now(UTC)
        return self.cooldown_until > now


class ProviderHealthTracker:
    """SQLite-backed cooldown tracker. Per (provider, model) entries.

    `model=""` is a provider-wide entry — used when every known model for a
    provider has cooled, so the whole provider is skipped until cleared.
    """

    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    # ------------------------------------------------------------------

    def get(self, provider: str, model: str = "") -> HealthRecord:
        with self._repo._open() as conn:
            row = conn.execute(
                "SELECT last_ok_at, cooldown_until, consecutive_failures "
                "FROM provider_health WHERE provider = ? AND model = ?",
                (provider, model),
            ).fetchone()
        if not row:
            return HealthRecord(
                provider=provider,
                model=model,
                last_ok_at=None,
                cooldown_until=None,
                consecutive_failures=0,
            )
        return HealthRecord(
            provider=provider,
            model=model,
            last_ok_at=_parse_ts(row[0]),
            cooldown_until=_parse_ts(row[1]),
            consecutive_failures=row[2],
        )

    def mark_ok(self, provider: str, model: str = "") -> None:
        now = datetime.now(UTC).isoformat()
        with self._repo._open() as conn:
            conn.execute(
                """
                INSERT INTO provider_health
                    (provider, model, last_ok_at, cooldown_until, consecutive_failures)
                VALUES (?, ?, ?, NULL, 0)
                ON CONFLICT(provider, model) DO UPDATE SET
                    last_ok_at = excluded.last_ok_at,
                    cooldown_until = NULL,
                    consecutive_failures = 0
                """,
                (provider, model, now),
            )

    def mark_failure(self, provider: str, model: str = "", cooldown_s: float = 60) -> HealthRecord:
        """Mark a failure. Returns the updated health record."""
        cooldown_until = datetime.now(UTC) + timedelta(seconds=cooldown_s)
        with self._repo._open() as conn:
            conn.execute(
                """
                INSERT INTO provider_health
                    (provider, model, cooldown_until, consecutive_failures)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(provider, model) DO UPDATE SET
                    cooldown_until = excluded.cooldown_until,
                    consecutive_failures = consecutive_failures + 1
                """,
                (provider, model, cooldown_until.isoformat()),
            )
        return self.get(provider, model)

    def clear(self, provider: str, model: str = "") -> None:
        """Explicit cooldown clear (used by somm doctor)."""
        with self._repo._open() as conn:
            conn.execute(
                "UPDATE provider_health SET cooldown_until = NULL, consecutive_failures = 0 "
                "WHERE provider = ? AND model = ?",
                (provider, model),
            )

    def next_uncool_at(self, providers: list[str] | None = None) -> datetime | None:
        """Return the earliest cooldown_until across tracked entries, or None if none cooling."""
        now = datetime.now(UTC).isoformat()
        with self._repo._open() as conn:
            if providers:
                placeholders = ",".join("?" for _ in providers)
                row = conn.execute(
                    f"SELECT MIN(cooldown_until) FROM provider_health "
                    f"WHERE cooldown_until > ? AND provider IN ({placeholders})",
                    (now, *providers),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT MIN(cooldown_until) FROM provider_health WHERE cooldown_until > ?",
                    (now,),
                ).fetchone()
        return _parse_ts(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RouterResult:
    response: SommResponse
    provider: str


class Router:
    """Preference-ordered fallback across providers + circuit breaker.

    Providers are tried in order; a provider-wide cooldown (model="") skips
    the provider entirely. Circuit breaker: after `circuit_break_after`
    consecutive failures on the provider-wide entry, cool the whole provider
    for `circuit_break_cooldown_s` seconds.

    Per-model cooldowns (inside providers like openrouter) are the provider's
    concern — it internally rotates and only surfaces a SommTransientError
    back to Router when ALL its models are cooled.
    """

    def __init__(
        self,
        providers: list[SommProvider],
        tracker: ProviderHealthTracker,
        circuit_break_after: int = 5,
        circuit_break_cooldown_s: float = 600,
        # Deprecated: dispatch no longer auto-sleeps on exhaustion; pass
        # wait= to dispatch (or SOMM_WAIT_ON_EXHAUSTED) instead.
        exhausted_sleep_cap_s: float = 300,
        plan_governor=None,
    ) -> None:
        self.providers = providers
        self.tracker = tracker
        self.circuit_break_after = circuit_break_after
        self.circuit_break_cooldown_s = circuit_break_cooldown_s
        self.exhausted_sleep_cap_s = exhausted_sleep_cap_s
        # Optional somm.plan_governor.PlanGovernor — metered-plan pacing.
        self.plan_governor = plan_governor

    def _apply_plan_governor(self, active: list) -> list:
        """Reorder/filter the chain by metered-plan pacing.

        In-pace providers keep their configured order; over-pace
        (`defer`) providers move to the back so they only serve when
        everything in-pace has failed; exhausted+enforced (`block`)
        providers are dropped. If EVERY provider is blocked the caller
        gets an empty list and dispatch raises loudly — an enforced
        quota is a budget ceiling, not a suggestion."""
        if self.plan_governor is None:
            return active
        try:
            ok, defer = [], []
            for p in active:
                d = self.plan_governor.decision(p.name)
                if d == "block":
                    continue
                (defer if d == "defer" else ok).append(p)
            return ok + defer
        except Exception:
            # Pacing is advisory infrastructure — never take the call
            # path down with it.
            return active

    def wait_until_any_uncool(self, providers: list[SommProvider], wait: float) -> None:
        """Block until at least one provider-wide cooldown clears or deadline expires."""
        started = time.monotonic()
        deadline = started + wait
        while providers and all(self.tracker.get(p.name).is_cooling() for p in providers):
            next_ok = self.tracker.next_uncool_at([p.name for p in providers])
            if next_ok is None:
                raise SommProvidersExhausted("no providers configured or all failed fatally")
            sleep_s = (next_ok - datetime.now(UTC)).total_seconds()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waited = time.monotonic() - started
                raise SommProvidersExhausted(
                    _wait_deadline_exhausted_message(waited, sleep_s),
                    next_cool_in_s=sleep_s,
                )
            jitter = random.uniform(0.5, 2.0)
            sleep_for = min(max(0.0, sleep_s + jitter), remaining)
            if sleep_for <= 0:
                continue
            _sleep_bounded(sleep_for, deadline)

    def dispatch(
        self,
        request: SommRequest,
        wait: float | None = None,
        retry_max: int | None = None,
        retry_backoff_s: float | None = None,
    ) -> RouterResult:
        """Try each provider. Return the first successful response.

        Raises:
            SommNoCapableProvider: no provider in the chain has a model that
                can serve the request's required capabilities (e.g. `vision`).
                Raised before any network call.
            SommProvidersExhausted: all providers cooled past the caller's
                wait deadline, or fail-fast exhaustion when wait is omitted.
            SommFatalError: any provider raised a fatal error.
        """
        required = list(getattr(request, "capabilities_required", None) or [])
        capable_providers, skipped = self._capability_filter(required)
        if required and not capable_providers:
            raise SommNoCapableProvider(
                f"no provider in chain can serve capabilities={required}",
                required=required,
                skipped=skipped,
            )
        active = capable_providers if required else self.providers
        pre_plan = list(active)
        active = self._apply_plan_governor(active)
        if not active:
            raise SommProvidersExhausted(
                "all providers blocked by enforced metered-plan limits: "
                + ", ".join(p.name for p in pre_plan)
                + " — see `somm plans` (raise quota, wait for the window, "
                "or set enforce=false in ~/.somm/plans.toml)"
            )

        deadline = None
        wait_started = None
        if wait is not None and wait > 0:
            wait_started = time.monotonic()
            deadline = wait_started + wait

        retry_rounds = 0
        retry_policy = retry_max is not None or retry_backoff_s is not None
        retry_cooldown_s = (
            (0.0 if retry_backoff_s is None else retry_backoff_s)
            if retry_policy
            else None
        )
        while True:
            attempt = self._try_once(
                request,
                active,
                transient_cooldown_s=retry_cooldown_s,
            )
            if attempt is not None:
                return attempt

            # All providers cooled or all failed transiently this round.
            next_ok = self.tracker.next_uncool_at([p.name for p in active])
            if retry_policy:
                if retry_max is not None and retry_rounds >= retry_max:
                    raise SommProvidersExhausted("all providers exhausted after policy retries")
                retry_rounds += 1
                if deadline is not None and wait_started is not None:
                    sleep_s = (
                        (next_ok - datetime.now(UTC)).total_seconds()
                        if next_ok is not None
                        else 0.0
                    )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        waited = time.monotonic() - wait_started
                        raise SommProvidersExhausted(
                            _wait_deadline_exhausted_message(waited, sleep_s),
                            next_cool_in_s=sleep_s,
                        )
                    if retry_backoff_s and retry_backoff_s > 0:
                        _sleep_bounded(min(retry_backoff_s, remaining), deadline)
                elif retry_backoff_s and retry_backoff_s > 0:
                    time.sleep(retry_backoff_s)
                continue
            if next_ok is None:
                raise SommProvidersExhausted("no providers configured or all failed fatally")

            sleep_s = (next_ok - datetime.now(UTC)).total_seconds()
            if deadline is None or wait_started is None:
                raise SommProvidersExhausted(
                    f"all providers cooled; next available in {sleep_s:.0f}s",
                    next_cool_in_s=sleep_s,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waited = time.monotonic() - wait_started
                raise SommProvidersExhausted(
                    _wait_deadline_exhausted_message(waited, sleep_s),
                    next_cool_in_s=sleep_s,
                )

            jitter = random.uniform(0.5, 2.0)
            sleep_for = min(max(0.0, sleep_s + jitter), remaining)
            if sleep_for <= 0:
                continue
            _sleep_bounded(sleep_for, deadline)

    # ------------------------------------------------------------------

    def _capability_filter(
        self, required: list[str]
    ) -> tuple[list[SommProvider], list[tuple[str, str, str]]]:
        """Filter providers by model capability.

        Returns (capable_providers, skipped). `skipped` is a list of
        (provider, model, reason) triples — surfaced via SommNoCapableProvider
        so operators can see why a provider was passed over.
        """
        if not required:
            return list(self.providers), []

        repo = getattr(self.tracker, "_repo", None)
        if repo is None:
            # No repo access → defensive allow.
            return list(self.providers), []

        capable: list[SommProvider] = []
        skipped: list[tuple[str, str, str]] = []
        for provider in self.providers:
            model = getattr(provider, "default_model", "") or ""
            ok, reason = provider_can_serve(repo, provider.name, model, required)
            if ok:
                capable.append(provider)
            else:
                skipped.append((provider.name, model, reason))
        return capable, skipped

    def _try_once(
        self,
        request: SommRequest,
        providers: list[SommProvider] | None = None,
        transient_cooldown_s: float | None = None,
    ) -> RouterResult | None:
        for provider in (providers if providers is not None else self.providers):
            health = self.tracker.get(provider.name)
            if health.is_cooling():
                continue
            try:
                resp = provider.generate(request)
                if (
                    not resp.text.strip()
                    and not resp.tool_calls
                    and not request.allow_empty
                ):
                    # Provider returned an empty response. Don't mark it as
                    # a hard failure (the HTTP call succeeded), but do move
                    # on to the next provider — empty is almost never what
                    # the caller wanted. Short cooldown so we don't hammer
                    # a provider that's consistently returning blanks.
                    # Tool-only responses (empty text + tool_calls) are the
                    # expected shape for tool-use turns — NOT empty.
                    self.tracker.mark_failure(
                        provider.name, cooldown_s=15,
                    )
                    continue
                self.tracker.mark_ok(provider.name)
                return RouterResult(response=resp, provider=provider.name)
            except SommFatalError:
                raise
            except SommTransientError as e:
                cooldown_s = e.cooldown_s if transient_cooldown_s is None else transient_cooldown_s
                rec = self.tracker.mark_failure(provider.name, cooldown_s=cooldown_s)
                if rec.consecutive_failures >= self.circuit_break_after:
                    self.tracker.mark_failure(
                        provider.name, cooldown_s=self.circuit_break_cooldown_s
                    )
                continue
            except Exception as exc:
                cooldown_s, transient = _classify_unknown(exc)
                if not transient:
                    raise
                if transient_cooldown_s is not None:
                    cooldown_s = transient_cooldown_s
                self.tracker.mark_failure(provider.name, cooldown_s=cooldown_s)
                continue
        return None


# ---------------------------------------------------------------------------


def _sleep_bounded(duration_s: float, deadline: float) -> None:
    """Sleep up to duration_s in <=5s chunks without overshooting deadline."""
    remaining_duration = max(0.0, duration_s)
    while remaining_duration > 0:
        remaining_deadline = deadline - time.monotonic()
        if remaining_deadline <= 0:
            return
        chunk = min(remaining_duration, remaining_deadline, 5.0)
        time.sleep(chunk)
        remaining_duration -= chunk


def _wait_deadline_exhausted_message(waited_s: float, next_cool_in_s: float) -> str:
    return (
        "SOMM_PROVIDERS_EXHAUSTED\n\n"
        "Problem: Every provider in the routed candidate set is cooling down, "
        f"and the wait deadline expired after {waited_s:.1f}s.\n"
        f"Cause: The next tracked cooldown expires in ~{max(0.0, next_cool_in_s):.0f}s, "
        "which is later than this call's wait budget.\n"
        "Fix: Increase generate(wait=...), set SOMM_WAIT_ON_EXHAUSTED for batch "
        "workers, add a healthy fallback provider, or wait for the cooldown to clear.\n"
        "Docs: docs/errors/SOMM_PROVIDERS_EXHAUSTED.md"
    )


def _classify_unknown(exc: Exception) -> tuple[float, bool]:
    """Bucket non-Somm exceptions as transient (with cooldown) or fatal."""
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return (60.0, True)
    if "connection" in msg or "reset" in msg or "network" in msg:
        return (30.0, True)
    if "auth" in msg or "401" in msg or "403" in msg:
        return (0.0, False)
    if "429" in msg:
        return (120.0, True)
    if "server busy" in msg or "pending requests exceeded" in msg:
        # Local ollama queue-full (503 "server busy, maximum pending
        # requests exceeded"): the queue drains in seconds. A 30s cooldown
        # here instantly exhausts single-provider projects — keep it short
        # so the router's exhausted-wait retry actually succeeds.
        return (5.0, True)
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return (30.0, True)
    # Default: treat unknown as transient; better than losing a call.
    return (60.0, True)


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        # SQLite stores ISO8601; may or may not include timezone
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None

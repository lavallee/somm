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

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from somm_core.repository import Repository

from somm.capabilities import provider_can_serve
from somm.errors import (
    SommAuthError,
    SommFatalError,
    SommModelNotFound,
    SommNoCapableProvider,
    SommPrivacyViolation,
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


ESCALATION_STEP = 10
"""Consecutive failures per doubling of the cooldown."""
MAX_COOLDOWN_S = 86_400.0
"""The longest an escalated cooldown gets: one probe a day."""
AUTH_COOLDOWN_S = 300.0
"""How long a provider that failed authentication sits out of the chain.

A bad key will not fix itself in seconds, but a rotated one should be picked
up within minutes, and the escalation ladder lengthens this for a key that
stays bad (issue #71)."""
DEFAULT_POLICY_RETRY_MAX = 3
"""Retry rounds for a workload ``retry`` policy that sets no valid ``max`` and
no ``deadline_s``. Such a policy used to retry forever."""


def escalated_cooldown_s(base_s: float, consecutive_failures: int) -> float:
    """The cooldown after ``consecutive_failures`` failures in a row.

    The first ``ESCALATION_STEP`` failures get the base cooldown the error
    asked for; each further block of ``ESCALATION_STEP`` doubles it, capped
    at ``MAX_COOLDOWN_S``. Ten doublings is the most that is ever applied.
    """
    doublings = min(max(consecutive_failures - 1, 0) // ESCALATION_STEP, 10)
    return float(min(MAX_COOLDOWN_S, max(base_s, base_s * (2**doublings))))


def dead_entries(
    repo: Repository, *, min_failures: int = 100, quiet_days: int = 7
) -> list[HealthRecord]:
    """Health entries that look withdrawn rather than flaky.

    An entry with at least ``min_failures`` consecutive failures and no
    success in ``quiet_days`` is a roster model the provider has retired, or
    a key that no longer works. The escalated cooldown keeps the cost of
    retrying it to one probe a day; ``somm doctor`` lists them so the roster
    (``SOMM_OPENROUTER_ROSTER``) can be pruned.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=quiet_days)).isoformat()
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT provider, model, last_ok_at, cooldown_until, consecutive_failures
            FROM provider_health
            WHERE consecutive_failures >= ?
              AND (last_ok_at IS NULL OR last_ok_at < ?)
            ORDER BY consecutive_failures DESC
            """,
            (min_failures, cutoff),
        ).fetchall()
    return [
        HealthRecord(
            provider=row[0],
            model=row[1],
            last_ok_at=_parse_ts(row[2]),
            cooldown_until=_parse_ts(row[3]),
            consecutive_failures=int(row[4]),
        )
        for row in rows
    ]


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
        """Mark a failure and cool the entry. Returns the updated record.

        ``cooldown_s`` is the floor. Every ``ESCALATION_STEP`` consecutive
        failures double it, up to ``MAX_COOLDOWN_S``, so an entry that has
        failed a thousand times in a row is probed about once a day rather
        than every couple of minutes. In October 2026 one project's ledger
        held a roster model with 68,166 consecutive failures since April —
        retried, and billed as latency, on a two-minute cycle for six months.
        A single success (:meth:`mark_ok`) resets the count and the cooldown.
        """
        now = datetime.now(UTC)
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
                (provider, model, (now + timedelta(seconds=cooldown_s)).isoformat()),
            )
            row = conn.execute(
                "SELECT consecutive_failures FROM provider_health WHERE provider = ? AND model = ?",
                (provider, model),
            ).fetchone()
            failures = int(row[0]) if row else 1
            escalated = escalated_cooldown_s(cooldown_s, failures)
            if escalated > cooldown_s:
                conn.execute(
                    "UPDATE provider_health SET cooldown_until = ? WHERE provider = ? AND model = ?",
                    ((now + timedelta(seconds=escalated)).isoformat(), provider, model),
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
    # Providers the chain routed past because they failed authentication on
    # this call: (provider, error). The call succeeded, but a bad key is a
    # degradation the caller should hear about (issue #71).
    skipped: list[tuple[str, Exception]] = field(default_factory=list)


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
            SommAuthError: every provider that was tried failed
                authentication. A provider that fails authentication is
                cooled and routed past; the error is raised only when nothing
                else in the chain answered.
            SommFatalError: any provider raised another fatal error.
            SommPrivacyViolation: ``request.data_boundary`` excludes every
                provider in the chain. Raised before any network call, and
                never retried.
        """
        chain = self._boundary_filter(request)
        required = list(getattr(request, "capabilities_required", None) or [])
        capable_providers, skipped = self._capability_filter(required, chain)
        if required and not capable_providers:
            raise SommNoCapableProvider(
                f"no provider in chain can serve capabilities={required}",
                required=required,
                skipped=skipped,
            )
        active = capable_providers if required else chain
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
        if retry_policy and not _valid_retry_max(retry_max) and (
            retry_max is not None or deadline is None
        ):
            # A policy that names only ``backoff_s`` (or a malformed ``max``)
            # used to retry forever. A deadline bounds a max-less policy;
            # without one, cap the rounds.
            retry_max = DEFAULT_POLICY_RETRY_MAX
        retry_cooldown_s = (
            (0.0 if retry_backoff_s is None else retry_backoff_s)
            if retry_policy
            else None
        )
        while True:
            auth_failures: list[tuple[str, Exception]] = []
            not_found: list[tuple[str, Exception]] = []
            attempt = self._try_once(
                request,
                active,
                transient_cooldown_s=retry_cooldown_s,
                auth_failures=auth_failures,
                not_found=not_found,
            )
            if attempt is not None:
                return attempt
            hard_failures = len(auth_failures) + len(not_found)
            if hard_failures and (
                hard_failures == len(active) or (deadline is None and not retry_policy)
            ):
                # Nothing answered and at least one provider rejected its
                # key or the model. Waiting will not mend either, so say that
                # rather than "all providers cooled", which points at the
                # wrong fix.
                if auth_failures:
                    raise _auth_chain_error(auth_failures)
                raise not_found[0][1]

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

    def _boundary_filter(self, request: SommRequest) -> list[SommProvider]:
        """Drop providers outside ``request.data_boundary`` before the loop.

        A refusal is raised here, never retried, and never falls through to
        a looser route: the providers it drops are simply not candidates.
        """
        boundary = getattr(request, "data_boundary", None)
        if not boundary:
            return list(self.providers)
        from somm.boundary import ResolvedBoundary, admissible_providers, parse_tier, refuse

        ok, refused = admissible_providers(self.providers, request.model, boundary)
        if not ok:
            raise refuse(
                workload=request.metadata.get("somm_workload") if request.metadata else None,
                boundary=ResolvedBoundary(
                    parse_tier(boundary),
                    (request.metadata or {}).get("somm_boundary_source") or "request",
                ),
                refused=refused,
                context="the routing chain",
            )
        return ok

    def _capability_filter(
        self, required: list[str], providers: list[SommProvider] | None = None
    ) -> tuple[list[SommProvider], list[tuple[str, str, str]]]:
        """Filter providers by model capability.

        Returns (capable_providers, skipped). `skipped` is a list of
        (provider, model, reason) triples — surfaced via SommNoCapableProvider
        so operators can see why a provider was passed over.
        """
        pool = list(self.providers if providers is None else providers)
        if not required:
            return pool, []

        repo = getattr(self.tracker, "_repo", None)
        if repo is None:
            # No repo access → defensive allow.
            return pool, []

        capable: list[SommProvider] = []
        skipped: list[tuple[str, str, str]] = []
        for provider in pool:
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
        auth_failures: list[tuple[str, Exception]] | None = None,
        not_found: list[tuple[str, Exception]] | None = None,
    ) -> RouterResult | None:
        skipped = auth_failures if auth_failures is not None else []
        missing = not_found if not_found is not None else []
        for provider in (providers if providers is not None else self.providers):
            health = self.tracker.get(provider.name)
            if health.is_cooling():
                continue
            # The (provider, model) slot this attempt would use. A slot the
            # provider said it does not serve is cooled on its own, so the
            # same provider can still serve other models (a policy fallback
            # across two models of one provider).
            slot = request.model or getattr(provider, "default_model", "") or ""
            slot_health = self.tracker.get(provider.name, slot) if slot else None
            if slot_health is not None and slot_health.is_cooling():
                continue
            # Only the provider call itself decides whether this provider
            # failed. Health bookkeeping (a synchronous ledger write) runs
            # after the try and is best-effort: through 0.20.1 a "database is
            # locked" from mark_ok() landed in the except arms below, so a
            # paid, successful answer was thrown away, the healthy provider
            # was cooled, and the next provider was called and paid again.
            try:
                resp = provider.generate(request)
            except SommAuthError as e:
                # One provider's bad key is that provider's problem, not the
                # request's: cool it and route on. Before 0.19.2 this raised,
                # so a bad key on the head of the chain failed every unpinned
                # call even with a healthy provider behind it (issue #71).
                # Pinned calls never reach the router, so they stay sticky.
                _record_health(
                    self.tracker.mark_failure, provider.name, cooldown_s=AUTH_COOLDOWN_S
                )
                skipped.append((provider.name, e))
                logging.getLogger("somm.routing").warning(
                    "%s failed authentication; routing past it for %.0fs: %s",
                    provider.name,
                    AUTH_COOLDOWN_S,
                    str(e)[:200],
                )
                continue
            except SommModelNotFound as e:
                # A withdrawn or misnamed model: cool the slot on the
                # escalating ladder and route on. Fatal (and never cooled)
                # before, so it aborted the chain on every call.
                _record_health(
                    self.tracker.mark_failure,
                    provider.name,
                    slot or e.model,
                    cooldown_s=e.cooldown_s,
                )
                missing.append((provider.name, e))
                logging.getLogger("somm.routing").warning(
                    "%s does not serve model %r; cooling that slot and routing on: %s",
                    provider.name,
                    slot or e.model,
                    str(e)[:200],
                )
                continue
            except (SommFatalError, SommPrivacyViolation):
                # A boundary refusal from inside a provider (the OpenRouter
                # roster had no admissible model) must not be read as a
                # transient failure and routed past.
                raise
            except SommTransientError as e:
                cooldown_s = e.cooldown_s if transient_cooldown_s is None else transient_cooldown_s
                rec = _record_health(
                    self.tracker.mark_failure, provider.name, cooldown_s=cooldown_s
                )
                if rec is not None and rec.consecutive_failures >= self.circuit_break_after:
                    _record_health(
                        self.tracker.mark_failure,
                        provider.name,
                        cooldown_s=self.circuit_break_cooldown_s,
                    )
                continue
            except Exception as exc:
                cooldown_s, transient = _classify_unknown(exc)
                if not transient:
                    raise
                if transient_cooldown_s is not None:
                    cooldown_s = transient_cooldown_s
                _record_health(self.tracker.mark_failure, provider.name, cooldown_s=cooldown_s)
                continue
            if not resp.text.strip() and not resp.tool_calls and not request.allow_empty:
                # Provider returned an empty response. Don't mark it as
                # a hard failure (the HTTP call succeeded), but do move
                # on to the next provider — empty is almost never what
                # the caller wanted. Short cooldown so we don't hammer
                # a provider that's consistently returning blanks.
                # Tool-only responses (empty text + tool_calls) are the
                # expected shape for tool-use turns — NOT empty.
                _record_health(self.tracker.mark_failure, provider.name, cooldown_s=15)
                continue
            _record_health(self.tracker.mark_ok, provider.name)
            if slot_health is not None and slot_health.consecutive_failures:
                _record_health(self.tracker.mark_ok, provider.name, slot)
            return RouterResult(response=resp, provider=provider.name, skipped=list(skipped))
        return None


# ---------------------------------------------------------------------------


def _valid_retry_max(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _record_health(write, *args, **kwargs):
    """Run a provider-health write; a ledger error is logged, never raised.

    Health rows steer routing but are not the answer. A locked or broken
    ledger must not turn a provider's success into a failure, or a provider's
    failure into a crash that stops the chain. Returns the write's result, or
    None when it failed.
    """
    try:
        return write(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — bookkeeping only
        logging.getLogger("somm.routing").warning(
            "provider health write failed; continuing without it: %s: %s",
            type(exc).__name__,
            str(exc)[:200],
        )
        return None


def _auth_chain_error(failures: list[tuple[str, Exception]]) -> SommAuthError:
    """The first auth error, naming every provider that rejected its key."""
    _, first = failures[0]
    if len(failures) == 1:
        return first if isinstance(first, SommAuthError) else SommAuthError(str(first))
    names = ", ".join(name for name, _ in failures)
    err = SommAuthError(f"{first} | every provider tried failed authentication: {names}")
    err.__cause__ = first
    return err


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

"""Router + ProviderHealthTracker.

Routing rules (from PLAN.md Eng phase):
  1. Providers tried in configured preference order.
  2. Each (provider, model) has its own cooldown entry in provider_health.
  3. On transient failure, cool the (provider, model) with per-error backoff.
  4. Router skips cooled entries. If ALL configured providers are cooled,
     Router sleeps until the next cool expires (bounded) and retries once.
  5. Fatal errors raise immediately — no fallback.

State is persisted in SQLite (`provider_health`) so cooldowns survive
process restarts — an overnight flaky free-tier doesn't get re-hammered
at dawn (PLAN.md §routing-strategy).
"""

from __future__ import annotations

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
        exhausted_sleep_cap_s: float = 300,
    ) -> None:
        self.providers = providers
        self.tracker = tracker
        self.circuit_break_after = circuit_break_after
        self.circuit_break_cooldown_s = circuit_break_cooldown_s
        self.exhausted_sleep_cap_s = exhausted_sleep_cap_s

    def dispatch(self, request: SommRequest) -> RouterResult:
        """Try each provider. Return the first successful response.

        Raises:
            SommNoCapableProvider: no provider in the chain has a model that
                can serve the request's required capabilities (e.g. `vision`).
                Raised before any network call.
            SommProvidersExhausted: all providers cooled for too long to wait.
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

        first_attempt = self._try_once(request, active)
        if first_attempt is not None:
            return first_attempt

        # All providers cooled or all failed transiently this round.
        next_ok = self.tracker.next_uncool_at([p.name for p in active])
        if next_ok is None:
            raise SommProvidersExhausted("no providers configured or all failed fatally")
        sleep_s = (next_ok - datetime.now(UTC)).total_seconds()
        if sleep_s <= 0:
            # cooldown already passed — rare race; just retry
            pass
        elif sleep_s > self.exhausted_sleep_cap_s:
            raise SommProvidersExhausted(
                f"all providers cooled; next available in {sleep_s:.0f}s",
                next_cool_in_s=sleep_s,
            )
        else:
            time.sleep(sleep_s + 0.1)

        retry = self._try_once(request, active)
        if retry is not None:
            return retry
        raise SommProvidersExhausted("all providers still failing after wait")

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
        self, request: SommRequest, providers: list[SommProvider] | None = None
    ) -> RouterResult | None:
        for provider in (providers if providers is not None else self.providers):
            health = self.tracker.get(provider.name)
            if health.is_cooling():
                continue
            try:
                resp = provider.generate(request)
                # Empty responses are returned as-is (outcome=EMPTY is set by
                # SommLLM, not the router). Empty != provider failure — the
                # provider *worked*, the model just didn't produce output.
                self.tracker.mark_ok(provider.name)
                return RouterResult(response=resp, provider=provider.name)
            except SommFatalError:
                raise
            except SommTransientError as e:
                rec = self.tracker.mark_failure(provider.name, cooldown_s=e.cooldown_s)
                if rec.consecutive_failures >= self.circuit_break_after:
                    self.tracker.mark_failure(
                        provider.name, cooldown_s=self.circuit_break_cooldown_s
                    )
                continue
            except Exception as exc:
                cooldown_s, transient = _classify_unknown(exc)
                if not transient:
                    raise
                self.tracker.mark_failure(provider.name, cooldown_s=cooldown_s)
                continue
        return None


# ---------------------------------------------------------------------------


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

"""Routing layer tests — tracker CRUD, Router fallback, cooldowns, circuit breaker.

No network. Uses FakeProvider instances that raise on command.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm.errors import (
    SommAuthError,
    SommProvidersExhausted,
    SommRateLimited,
    SommTransientError,
)
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)
from somm.routing import ProviderHealthTracker, Router
from somm_core.config import Config
from somm_core.repository import Repository

# ---------------------------------------------------------------------------
# Helpers


def _tmp_repo(tmp_path: Path) -> Repository:
    cfg = Config()
    cfg.project = "rt"
    cfg.db_dir = tmp_path / ".somm"
    return Repository(cfg.db_path)


class ScriptedProvider:
    """Provider that plays from a script. Each call consumes one script entry.

    An entry is either a SommResponse (success) or an Exception instance.
    """

    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self._script = list(script)
        self.calls = 0

    def generate(self, request: SommRequest) -> SommResponse:
        self.calls += 1
        if not self._script:
            raise AssertionError(f"{self.name}: no more script entries")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:  # pragma: no cover
        yield

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def models(self) -> list[SommModel]:
        return []

    def estimate_tokens(self, text: str, model: str) -> int:
        return 1


def _ok(text: str = "hello", model: str = "p1") -> SommResponse:
    return SommResponse(text=text, model=model, tokens_in=1, tokens_out=1, latency_ms=1, raw=None)


# ---------------------------------------------------------------------------
# ProviderHealthTracker


def test_tracker_default_is_healthy(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    h = tr.get("p1")
    assert h.cooldown_until is None
    assert h.consecutive_failures == 0
    assert not h.is_cooling()


def test_mark_failure_sets_cooldown(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    h = tr.mark_failure("p1", cooldown_s=60)
    assert h.is_cooling()
    assert h.consecutive_failures == 1
    # Second failure increments consecutive_failures
    h2 = tr.mark_failure("p1", cooldown_s=60)
    assert h2.consecutive_failures == 2


def test_mark_ok_clears_cooldown(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    tr.mark_failure("p1", cooldown_s=60)
    tr.mark_ok("p1")
    h = tr.get("p1")
    assert not h.is_cooling()
    assert h.consecutive_failures == 0


def test_next_uncool_at_finds_soonest(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    tr.mark_failure("p1", cooldown_s=300)
    tr.mark_failure("p2", cooldown_s=60)
    nxt = tr.next_uncool_at(["p1", "p2"])
    assert nxt is not None
    delta = (nxt - datetime.now(UTC)).total_seconds()
    assert 50 <= delta <= 65  # p2's cooldown


def test_tracker_per_model_isolation(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    tr.mark_failure("openrouter", "model-a", cooldown_s=60)
    # model-b is unaffected
    assert not tr.get("openrouter", "model-b").is_cooling()
    # provider-wide entry (model="") also unaffected
    assert not tr.get("openrouter").is_cooling()


# ---------------------------------------------------------------------------
# Router — happy path + fallback


def test_router_returns_first_success(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p1 = ScriptedProvider("p1", [_ok(model="p1-m")])
    p2 = ScriptedProvider("p2", [_ok(model="p2-m")])
    r = Router([p1, p2], tr)

    result = r.dispatch(SommRequest(prompt="hi"))
    assert result.provider == "p1"
    assert result.response.model == "p1-m"
    assert p1.calls == 1
    assert p2.calls == 0


def test_router_falls_through_transient(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p1 = ScriptedProvider("p1", [SommTransientError("flaky", cooldown_s=5)])
    p2 = ScriptedProvider("p2", [_ok(model="p2-m")])
    r = Router([p1, p2], tr)

    result = r.dispatch(SommRequest(prompt="hi"))
    assert result.provider == "p2"
    assert p1.calls == 1
    assert tr.get("p1").is_cooling()
    assert not tr.get("p2").is_cooling()


def test_router_raises_fatal_immediately(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p1 = ScriptedProvider("p1", [SommAuthError("bad key")])
    p2 = ScriptedProvider("p2", [_ok()])
    r = Router([p1, p2], tr)

    with pytest.raises(SommAuthError):
        r.dispatch(SommRequest(prompt="hi"))
    # p2 never called
    assert p2.calls == 0


def test_router_skips_cooled_provider(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    # Pre-cool p1
    tr.mark_failure("p1", cooldown_s=300)
    p1 = ScriptedProvider("p1", [])  # no script — will error if called
    p2 = ScriptedProvider("p2", [_ok(model="p2-m")])
    r = Router([p1, p2], tr)

    result = r.dispatch(SommRequest(prompt="hi"))
    assert result.provider == "p2"
    assert p1.calls == 0


def test_router_exhausted_when_all_cooled_too_long(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    tr.mark_failure("p1", cooldown_s=3600)  # 1h cool
    tr.mark_failure("p2", cooldown_s=3600)
    p1 = ScriptedProvider("p1", [])
    p2 = ScriptedProvider("p2", [])
    r = Router([p1, p2], tr, exhausted_sleep_cap_s=5)

    with pytest.raises(SommProvidersExhausted):
        r.dispatch(SommRequest(prompt="hi"))


def test_router_rate_limited_uses_retry_after(tmp_path):
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p1 = ScriptedProvider("p1", [SommRateLimited("429", retry_after_s=200)])
    p2 = ScriptedProvider("p2", [_ok()])
    r = Router([p1, p2], tr)

    r.dispatch(SommRequest(prompt="hi"))
    h1 = tr.get("p1")
    assert h1.cooldown_until is not None
    remaining = (h1.cooldown_until - datetime.now(UTC)).total_seconds()
    # ~200s retry-after
    assert 190 <= remaining <= 210


def test_router_empty_response_returned_as_is(tmp_path):
    """Empty responses are NOT treated as provider failure. The provider
    worked; the model just had nothing to say. Caller gets outcome=EMPTY."""
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p1 = ScriptedProvider(
        "p1",
        [SommResponse(text="   ", model="m", tokens_in=0, tokens_out=0, latency_ms=1)],
    )
    p2 = ScriptedProvider("p2", [_ok()])
    r = Router([p1, p2], tr)

    result = r.dispatch(SommRequest(prompt="hi"))
    assert result.provider == "p1"  # p1 "succeeded" — empty is a valid response
    assert result.response.text.strip() == ""
    assert not tr.get("p1").is_cooling()  # no cooling — provider worked


def test_router_circuit_break_after_n_failures(tmp_path):
    """After N consecutive failures on a provider, cooldown escalates."""
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    # Pre-seed 4 failures with near-zero cooldown so they've expired when
    # router runs — we want the 5th failure (from the router) to trigger
    # circuit break, not be skipped as still-cooling.
    for _ in range(4):
        tr.mark_failure("p1", cooldown_s=0.001)
    # Sanity: p1 should no longer be cooling
    import time as _t

    _t.sleep(0.01)
    assert not tr.get("p1").is_cooling()

    # Now a 5th failure from the router triggers the long cooldown.
    p1 = ScriptedProvider("p1", [SommTransientError("still bad", cooldown_s=1)])
    p2 = ScriptedProvider("p2", [_ok()])
    r = Router([p1, p2], tr, circuit_break_after=5, circuit_break_cooldown_s=1800)

    r.dispatch(SommRequest(prompt="hi"))
    h = tr.get("p1")
    assert h.consecutive_failures >= 5
    remaining = (h.cooldown_until - datetime.now(UTC)).total_seconds()
    assert remaining >= 1700  # long cooldown engaged


def test_router_clears_cooldown_after_success(tmp_path):
    """Successful call clears the (provider) cooldown + failure counter."""
    repo = _tmp_repo(tmp_path)
    tr = ProviderHealthTracker(repo)
    p1 = ScriptedProvider("p1", [SommTransientError("flaky", cooldown_s=5), _ok()])
    p2 = ScriptedProvider("p2", [_ok(model="p2-m")])
    r = Router([p1, p2], tr)

    # First call — p1 fails, p2 handles
    r.dispatch(SommRequest(prompt="hi"))
    assert tr.get("p1").is_cooling()

    # Simulate cooldown expiry by manually clearing
    tr.clear("p1")
    assert not tr.get("p1").is_cooling()

    # Second call — p1 succeeds, counter clears
    result = r.dispatch(SommRequest(prompt="hi2"))
    assert result.provider == "p1"
    assert tr.get("p1").consecutive_failures == 0

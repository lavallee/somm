"""no_fallback=True suppresses pinned-call rescue through the router chain.

The default behavior on a pinned-but-failed call is to fall through to the
router chain so production workloads recover. Evaluation harnesses need the
opposite: silent reroute to a different model invalidates the experiment, so
the caller wants to see the upstream failure cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from somm.client import SommLLM
from somm.errors import SommTransientError
from somm.providers.base import ProviderHealth, SommResponse
from somm_core import Outcome
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "hi") -> None:
        self._text = text

    def generate(self, request):
        return SommResponse(
            text=self._text,
            model=request.model or "fake-model",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            raw=None,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


class FailingProvider:
    """A provider that always raises a transient error."""

    name = "broken"

    def __init__(self, error_msg: str = "upstream 502") -> None:
        self._error_msg = error_msg

    def generate(self, request):
        raise SommTransientError(self._error_msg)

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=False, detail=self._error_msg)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "nf"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def test_default_pinned_call_falls_through_when_provider_fails(tmp_path):
    """Baseline: pinned call that fails recovers via the router chain.

    This is current production behavior; the test pins it down so a regression
    on the no_fallback path doesn't accidentally also break this.
    """
    cfg = _tmp_config(tmp_path)
    broken = FailingProvider()
    rescue = FakeProvider(text="rescued")
    llm = SommLLM(config=cfg, providers=[broken, rescue])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
        )
        assert result.outcome == Outcome.OK
        assert result.text == "rescued"
        assert result.provider == "fake"  # the rescue, not the pin
    finally:
        llm.close()


def test_no_fallback_returns_upstream_error_with_pinned_attribution(tmp_path):
    """no_fallback=True: pinned call that fails surfaces the error directly,
    with the *pinned* (provider, model) attribution preserved so eval queries
    can group attempts by the model they intended to test."""
    cfg = _tmp_config(tmp_path)
    broken = FailingProvider("Provider returned error 502")
    rescue = FakeProvider(text="should-not-be-used")
    llm = SommLLM(config=cfg, providers=[broken, rescue])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
            no_fallback=True,
        )
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert result.text == ""
        assert result.provider == "broken"
        assert result.model == "pinned-model"
        assert result.error_kind == "SommTransientError"
        assert "Provider returned error 502" in (result.error_detail or "")

        # Telemetry row matches result attribution — admin queries can
        # group failed pinned attempts by the model the caller asked for.
        llm.close()
        call = llm.repo.get_call(result.call_id)
        assert call is not None
        assert call.provider == "broken"
        assert call.model == "pinned-model"
        assert call.outcome == Outcome.UPSTREAM_ERROR
    finally:
        llm.close()


def test_no_fallback_does_not_affect_router_only_calls(tmp_path):
    """When no provider is pinned, no_fallback has nothing to suppress —
    router-driven calls behave normally."""
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="ok")
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        # No provider= argument: the router picks. no_fallback is a no-op here
        # because there's no pinned target to honor.
        result = llm.generate("p", workload="ad_hoc_test", no_fallback=True)
        assert result.outcome == Outcome.OK
        assert result.text == "ok"
    finally:
        llm.close()


def test_no_fallback_succeeds_when_pinned_provider_works(tmp_path):
    """no_fallback should not affect the happy path. Pinned + works = ok."""
    cfg = _tmp_config(tmp_path)
    fake = FakeProvider(text="good")
    llm = SommLLM(config=cfg, providers=[fake])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="fake",
            model="my-model",
            no_fallback=True,
        )
        assert result.outcome == Outcome.OK
        assert result.text == "good"
        assert result.provider == "fake"
    finally:
        llm.close()

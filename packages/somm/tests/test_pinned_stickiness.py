"""A pin is sticky: naming a provider/model means "serve this", not "try this first".

Before v0.16 a pinned call that failed was silently rescued through the router
chain, so a caller who asked for one model could get an answer from another —
unexplainable in production and quietly invalidating for evaluation, comparison,
and replay runs. Rescue is now opt-in: `allow_fallback=True` per call, or
`SOMM_PINNED_FALLBACK=1` / `Config.pinned_fallback` process-wide.
"""

from __future__ import annotations

from pathlib import Path

from somm.client import SommLLM
from somm.errors import SommTransientError
from somm.providers.base import ProviderHealth, SommResponse
from somm_core import Outcome
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "hi") -> None:
        self._text = text
        self.seen_models: list[str | None] = []

    def generate(self, request):
        self.seen_models.append(request.model)
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
    default_model = "broken-default"

    def __init__(self, error_msg: str = "upstream 502") -> None:
        self._error_msg = error_msg
        self.seen_models: list[str | None] = []

    def generate(self, request):
        self.seen_models.append(request.model)
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


# ---------------------------------------------------------------------------
# The default


def test_pinned_call_sticks_to_its_model_when_the_provider_fails(tmp_path):
    """The headline default: a failed pin surfaces the failure with the
    *pinned* (provider, model) attribution — never a substituted answer."""
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
        )
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert result.text == ""
        assert result.provider == "broken"
        assert result.model == "pinned-model"
        assert result.error_kind == "SommTransientError"
        assert "Provider returned error 502" in (result.error_detail or "")
        # The rescue provider was never asked.
        assert rescue.seen_models == []

        # Telemetry row matches result attribution — admin queries can group
        # failed pinned attempts by the model the caller asked for.
        llm.close()
        call = llm.repo.get_call(result.call_id)
        assert call is not None
        assert call.provider == "broken"
        assert call.model == "pinned-model"
        assert call.outcome == Outcome.UPSTREAM_ERROR
    finally:
        llm.close()


def test_sticky_failure_detail_names_the_escape_hatch(tmp_path):
    """A dead end by design still has to say what the next step is, or
    `somm tail` shows an error an operator can't act on."""
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FailingProvider(), FakeProvider()])
    try:
        result = llm.generate("p", workload="ad_hoc_test", provider="broken")
        assert "allow_fallback=True" in (result.error_detail or "")
    finally:
        llm.close()


def test_pinned_provider_without_model_is_also_sticky(tmp_path):
    """`provider=` alone pins the provider's default model. Rescuing it is
    the same silent substitution, so it gets the same treatment."""
    cfg = _tmp_config(tmp_path)
    rescue = FakeProvider(text="should-not-be-used")
    llm = SommLLM(config=cfg, providers=[FailingProvider(), rescue])
    try:
        result = llm.generate("p", workload="ad_hoc_test", provider="broken")
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert result.provider == "broken"
        # The provider's default model is attributed, not a blank column —
        # a failure with no model is invisible to every per-model query.
        assert result.model == "broken-default"
        assert rescue.seen_models == []
    finally:
        llm.close()


def test_pinned_call_succeeds_normally(tmp_path):
    """Stickiness must not touch the happy path. Pinned + works = ok."""
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider(text="good")])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="fake",
            model="my-model",
        )
        assert result.outcome == Outcome.OK
        assert result.text == "good"
        assert result.provider == "fake"
        assert result.model == "my-model"
    finally:
        llm.close()


def test_router_only_calls_are_unaffected(tmp_path):
    """No pin, nothing to honor: the chain routes and rescues as always."""
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FailingProvider(), FakeProvider(text="ok")])
    try:
        result = llm.generate("p", workload="ad_hoc_test")
        assert result.outcome == Outcome.OK
        assert result.text == "ok"
        assert result.provider == "fake"
    finally:
        llm.close()


def test_explicit_model_is_never_swapped_on_the_router_path(tmp_path):
    """`model=` without `provider=` still routes across the chain, but the
    named model rides along unchanged — no provider gets to answer with its
    own default instead."""
    cfg = _tmp_config(tmp_path)
    broken = FailingProvider()
    rescue = FakeProvider(text="ok")
    llm = SommLLM(config=cfg, providers=[broken, rescue])
    try:
        result = llm.generate("p", workload="ad_hoc_test", model="asked-for")
        assert result.outcome == Outcome.OK
        assert result.model == "asked-for"
        assert broken.seen_models == ["asked-for"]
        assert rescue.seen_models == ["asked-for"]
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Opting back into rescue


def test_allow_fallback_true_restores_chain_rescue(tmp_path):
    """The escape hatch: callers who would rather have an answer from some
    model than none ask for it explicitly."""
    cfg = _tmp_config(tmp_path)
    rescue = FakeProvider(text="rescued")
    llm = SommLLM(config=cfg, providers=[FailingProvider(), rescue])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
            allow_fallback=True,
        )
        assert result.outcome == Outcome.OK
        assert result.text == "rescued"
        assert result.provider == "fake"  # the rescue, not the pin
        # The pinned model name is dropped before the chain runs — it means
        # nothing to a provider serving a different inventory.
        assert rescue.seen_models == [None]
    finally:
        llm.close()


def test_config_pinned_fallback_restores_the_old_default(tmp_path):
    """SOMM_PINNED_FALLBACK=1 for fleets that depended on the pre-0.16
    behavior and aren't ready to audit every call site."""
    cfg = _tmp_config(tmp_path)
    cfg.pinned_fallback = True
    llm = SommLLM(config=cfg, providers=[FailingProvider(), FakeProvider(text="rescued")])
    try:
        result = llm.generate(
            "p", workload="ad_hoc_test", provider="broken", model="pinned-model"
        )
        assert result.outcome == Outcome.OK
        assert result.text == "rescued"
    finally:
        llm.close()


def test_per_call_allow_fallback_false_beats_the_config_default(tmp_path):
    """A call that must not wander says so, even under a permissive config."""
    cfg = _tmp_config(tmp_path)
    cfg.pinned_fallback = True
    llm = SommLLM(config=cfg, providers=[FailingProvider(), FakeProvider(text="rescued")])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
            allow_fallback=False,
        )
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert result.model == "pinned-model"
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# The deprecated spelling


def test_no_fallback_true_still_forces_stickiness(tmp_path):
    """Existing eval harnesses pass no_fallback=True. It still means what it
    always meant, even under a config that permits rescue."""
    cfg = _tmp_config(tmp_path)
    cfg.pinned_fallback = True
    llm = SommLLM(config=cfg, providers=[FailingProvider(), FakeProvider(text="rescued")])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
            no_fallback=True,
        )
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert result.model == "pinned-model"
    finally:
        llm.close()


def test_no_fallback_false_does_not_re_enable_rescue(tmp_path):
    """`no_fallback=bool(args.provider)` was written to mean "don't rescue",
    never "please substitute". A False from that idiom must not resurrect the
    behavior we just removed — only allow_fallback= can do that."""
    cfg = _tmp_config(tmp_path)
    rescue = FakeProvider(text="rescued")
    llm = SommLLM(config=cfg, providers=[FailingProvider(), rescue])
    try:
        result = llm.generate(
            "p",
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
            no_fallback=False,
        )
        assert result.outcome == Outcome.UPSTREAM_ERROR
        assert rescue.seen_models == []
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# Wrappers


def test_structured_wrappers_stay_pinned_across_retries(tmp_path):
    """generate_structured retries in-place; a retry that lands on a
    different model would change what the schema is being tested against."""
    cfg = _tmp_config(tmp_path)
    rescue = FakeProvider(text='{"ok": true}')
    llm = SommLLM(config=cfg, providers=[FailingProvider(), rescue])
    try:
        raised = False
        try:
            llm.generate_structured(
                "p",
                schema={"type": "object"},
                workload="ad_hoc_test",
                provider="broken",
                model="pinned-model",
                retries=1,
            )
        except Exception:
            raised = True
        assert raised
        assert rescue.seen_models == []
    finally:
        llm.close()


def test_structured_wrappers_forward_allow_fallback(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FailingProvider(), FakeProvider(text='{"ok": true}')])
    try:
        obj, result = llm.generate_structured(
            "p",
            schema={"type": "object"},
            workload="ad_hoc_test",
            provider="broken",
            model="pinned-model",
            allow_fallback=True,
        )
        assert obj == {"ok": True}
        assert result.provider == "fake"
    finally:
        llm.close()

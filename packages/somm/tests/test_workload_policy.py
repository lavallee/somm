from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from somm.client import SommLLM
from somm.errors import SommProvidersExhausted, SommTransientError
from somm.providers.base import ProviderHealth, SommChunk, SommRequest, SommResponse
from somm_core.config import Config
from somm_core.repository import Repository


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "policy"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


class PolicyProvider:
    def __init__(
        self,
        name: str,
        script: list[object] | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.name = name
        self.default_model = f"{name}-default"
        self.timeout = timeout
        self.script = list(script or [])
        self.calls: list[dict] = []

    def generate(self, request: SommRequest) -> SommResponse:
        self.calls.append({"model": request.model, "timeout": self.timeout})
        item = self.script.pop(0) if self.script else "ok"
        if callable(item):
            item = item()
        if isinstance(item, Exception):
            raise item
        return SommResponse(
            text=str(item),
            model=request.model or self.default_model,
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            raw=None,
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:  # pragma: no cover
        yield

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def models(self) -> list:
        return []

    def estimate_tokens(self, text, model) -> int:
        return 1


def _transient() -> SommTransientError:
    return SommTransientError("try again", cooldown_s=0.0)


def _cooldown_transient() -> SommTransientError:
    return SommTransientError("try again later", cooldown_s=30.0)


def test_no_policy_uses_existing_provider_order_and_records_same_choice(tmp_path):
    a = PolicyProvider("a")
    b = PolicyProvider("b")
    with SommLLM(config=_tmp_config(tmp_path), providers=[a, b]) as llm:
        llm.register_workload(name="plain")

        result = llm.generate("hi", workload="plain")
        llm._writer.flush()

        assert result.provider == "a"
        assert result.model == "a-default"
        assert len(a.calls) == 1
        assert b.calls == []
        row = llm.repo.get_call(result.call_id)
        assert row is not None
        assert row.provider == "a"
        assert row.model == "a-default"


def test_policy_fallback_order_and_transient_rescue(tmp_path):
    preferred = PolicyProvider("b", [_transient()])
    rescue = PolicyProvider("a", ["rescued"])
    with SommLLM(
        config=_tmp_config(tmp_path),
        providers=[rescue, preferred],
        on_error=lambda _: None,
    ) as llm:
        llm.register_workload(name="chain")
        llm.set_workload_policy(
            "chain",
            {
                "fallback": [
                    {"provider": "b", "model": "b-model"},
                    {"provider": "a", "model": "a-model"},
                ]
            },
        )

        result = llm.generate("hi", workload="chain")

        assert result.provider == "a"
        assert result.model == "a-model"
        assert preferred.calls == [{"model": "b-model", "timeout": 60.0}]
        assert rescue.calls == [{"model": "a-model", "timeout": 60.0}]


def test_policy_gracefully_skips_unconfigured_provider(tmp_path):
    a = PolicyProvider("a")
    with SommLLM(config=_tmp_config(tmp_path), providers=[a]) as llm:
        llm.register_workload(name="degrade")
        llm.set_workload_policy(
            "degrade",
            {
                "fallback": [
                    {"provider": "missing", "model": "x"},
                    {"provider": "a", "model": "a-model"},
                ]
            },
        )

        result = llm.generate("hi", workload="degrade")

        assert result.provider == "a"
        assert result.model == "a-model"
        assert a.calls == [{"model": "a-model", "timeout": 60.0}]


def test_explicit_provider_overrides_policy_provider_choice(tmp_path):
    a = PolicyProvider("a")
    b = PolicyProvider("b")
    with SommLLM(config=_tmp_config(tmp_path), providers=[a, b]) as llm:
        llm.register_workload(name="override")
        llm.set_workload_policy(
            "override",
            {"fallback": [{"provider": "b", "model": "b-model"}]},
        )

        result = llm.generate("hi", workload="override", provider="a")

        assert result.provider == "a"
        assert result.model == "a-default"
        assert len(a.calls) == 1
        assert b.calls == []


def test_explicit_model_overrides_policy_entry_model(tmp_path):
    b = PolicyProvider("b")
    with SommLLM(config=_tmp_config(tmp_path), providers=[b]) as llm:
        llm.register_workload(name="model-override")
        llm.set_workload_policy(
            "model-override",
            {"fallback": [{"provider": "b", "model": "policy-model"}]},
        )

        result = llm.generate("hi", workload="model-override", model="caller-model")

        assert result.provider == "b"
        assert result.model == "caller-model"
        assert b.calls == [{"model": "caller-model", "timeout": 60.0}]


def test_policy_retry_recovers_without_deadline_or_real_sleep(tmp_path):
    a = PolicyProvider("a", [_cooldown_transient(), _cooldown_transient(), "ok"])
    with SommLLM(
        config=_tmp_config(tmp_path),
        providers=[a],
        on_error=lambda _: None,
    ) as llm:
        llm.register_workload(name="retry")
        llm.set_workload_policy(
            "retry",
            {
                "fallback": [{"provider": "a"}],
                "retry": {"max": 2, "backoff_s": 0.0},
            },
        )

        result = llm.generate("hi", workload="retry")

        assert result.provider == "a"
        assert result.text == "ok"
        assert len(a.calls) == 3


def test_policy_retry_max_exhaustion_without_deadline(tmp_path):
    a = PolicyProvider(
        "a",
        [_cooldown_transient(), _cooldown_transient(), _cooldown_transient()],
    )
    with SommLLM(
        config=_tmp_config(tmp_path),
        providers=[a],
        on_error=lambda _: None,
    ) as llm:
        llm.register_workload(name="retry-exhausted")
        llm.set_workload_policy(
            "retry-exhausted",
            {
                "fallback": [{"provider": "a"}],
                "retry": {"max": 2, "backoff_s": 0.0},
            },
        )

        with pytest.raises(SommProvidersExhausted):
            llm.generate("hi", workload="retry-exhausted")

        assert len(a.calls) == 3


def test_policy_retry_deadline_surfaces_exhaustion(tmp_path, monkeypatch):
    clock = {"now": 0.0}

    def fail_and_advance() -> SommTransientError:
        clock["now"] += 1.0
        return _transient()

    monkeypatch.setattr("somm.routing.time.monotonic", lambda: clock["now"])
    a = PolicyProvider("a", [fail_and_advance])
    with SommLLM(
        config=_tmp_config(tmp_path),
        providers=[a],
        on_error=lambda _: None,
    ) as llm:
        llm.register_workload(name="deadline")
        llm.set_workload_policy(
            "deadline",
            {
                "fallback": [{"provider": "a"}],
                "retry": {"max": 5, "backoff_s": 0.0, "deadline_s": 0.5},
            },
        )

        with pytest.raises(SommProvidersExhausted):
            llm.generate("hi", workload="deadline")


def test_set_workload_policy_validates_and_records_revision(tmp_path):
    repo = Repository(tmp_path / "somm.sqlite")
    wl = repo.register_workload(name="w", project="p")

    with pytest.raises(ValueError, match="fallback"):
        repo.set_workload_policy(wl.id, {"fallback": [{"model": "missing-provider"}]})

    policy = {
        "fallback": [{"provider": "a", "model": None}],
        "retry": {"max": 2, "backoff_s": 1, "deadline_s": 30},
        "timeout_s": 180,
    }
    repo.set_workload_policy(wl.id, policy, created_by="test")

    refreshed = repo.workload_by_name("w", "p")
    assert refreshed is not None
    assert refreshed.policy == {
        "fallback": [{"provider": "a", "model": None}],
        "retry": {"max": 2, "backoff_s": 1.0, "deadline_s": 30.0},
        "timeout_s": 180.0,
    }
    revisions = repo.workload_revisions(wl.id)
    assert [row["revision"] for row in revisions] == [1, 2]
    assert revisions[-1]["created_by"] == "test"
    assert revisions[-1]["config"]["policy"] == refreshed.policy


def test_policy_timeout_uses_copy_without_mutating_shared_provider(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingPolicyProvider(PolicyProvider):
        def generate(self, request: SommRequest) -> SommResponse:
            self.calls.append({"model": request.model, "timeout": self.timeout})
            if request.prompt == "policy":
                entered.set()
                assert release.wait(timeout=2.0)
            return SommResponse(
                text="ok",
                model=request.model or self.default_model,
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                raw=None,
            )

    a = BlockingPolicyProvider("a", timeout=10.0)
    with SommLLM(config=_tmp_config(tmp_path), providers=[a]) as llm:
        llm.register_workload(name="timeout")
        llm.set_workload_policy(
            "timeout",
            {"fallback": [{"provider": "a"}], "timeout_s": 180.0},
        )

        result_holder: dict[str, object] = {}

        def run_policy_call() -> None:
            try:
                result_holder["result"] = llm.generate("policy", workload="timeout")
            except Exception as exc:  # pragma: no cover - surfaced below
                result_holder["error"] = exc

        thread = threading.Thread(target=run_policy_call)
        thread.start()
        assert entered.wait(timeout=2.0)

        direct = a.generate(SommRequest(prompt="direct"))
        release.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        if "error" in result_holder:
            raise result_holder["error"]  # type: ignore[misc]

        result = result_holder["result"]
        assert result.provider == "a"
        assert direct.text == "ok"
        assert a.calls == [
            {"model": None, "timeout": 180.0},
            {"model": None, "timeout": 10.0},
        ]
        assert a.timeout == 10.0

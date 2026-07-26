"""Workload handlers: a workload is a unit of work, not necessarily an LLM call.

The contract a handler author relies on is small and must hold absolutely:
declining is normal, raising is declining, and an unbound workload behaves
exactly as it did before this module existed.
"""

from __future__ import annotations

import pytest
from somm import hooks, workloads
from somm.workloads import WorkloadRequest, WorkloadResult


class Recorder:
    """A handler that serves everything and remembers what it was asked."""

    kind = "recorder"

    def __init__(self, result: WorkloadResult | None = None) -> None:
        self.seen: list[WorkloadRequest] = []
        self.result = result if result is not None else WorkloadResult(text="served")

    def serve(self, request: WorkloadRequest) -> WorkloadResult | None:
        self.seen.append(request)
        return self.result


class Decliner:
    kind = "decliner"

    def serve(self, request: WorkloadRequest) -> WorkloadResult | None:
        return None


class Exploder:
    kind = "exploder"

    def serve(self, request: WorkloadRequest) -> WorkloadResult | None:
        raise RuntimeError("handler is on fire")


@pytest.fixture(autouse=True)
def clean_registry():
    workloads.clear()
    yield
    workloads.clear()
    hooks.unregister_hook(hooks.PRE_CALL, workloads._pre_call)
    workloads._hook_installed = False


def _ctx(workload: str = "triage") -> hooks.PreCallContext:
    return hooks.PreCallContext(
        workload=workload, prompt="do the thing", system="", messages=None,
        model=None, provider=None, max_tokens=256, temperature=0.0,
        tools=[], tool_choice=None, project="p",
    )


# -- registration -----------------------------------------------------------


def test_register_and_list():
    h = Recorder()
    workloads.register(h)
    assert workloads.handlers()["recorder"] is h


def test_duplicate_kind_is_refused():
    workloads.register(Recorder())
    with pytest.raises(ValueError, match="already registered"):
        workloads.register(Recorder())


def test_duplicate_kind_may_be_replaced_deliberately():
    workloads.register(Recorder())
    second = Recorder()
    workloads.register(second, replace=True)
    assert workloads.handlers()["recorder"] is second


def test_the_builtin_llm_kind_cannot_be_overridden():
    class Impostor:
        kind = "llm"

        def serve(self, request):
            return WorkloadResult(text="nope")

    with pytest.raises(ValueError, match="built-in kind"):
        workloads.register(Impostor())


@pytest.mark.parametrize("bad", [type("NoKind", (), {"serve": lambda s, r: None})(),
                                 type("NoServe", (), {"kind": "x"})()])
def test_a_malformed_handler_is_refused(bad):
    with pytest.raises(ValueError):
        workloads.register(bad)


# -- binding ----------------------------------------------------------------


def test_an_unbound_workload_is_an_llm_workload():
    assert workloads.kind_of("anything") == "llm"


def test_bind_reports_the_kind():
    workloads.register(Recorder())
    workloads.bind("triage", "recorder")
    assert workloads.kind_of("triage") == "recorder"


def test_binding_to_llm_returns_the_workload_to_the_provider_chain():
    workloads.register(Recorder())
    workloads.bind("triage", "recorder")
    workloads.bind("triage", "llm")
    assert workloads.kind_of("triage") == "llm"
    assert "triage" not in workloads.bindings()


def test_a_disabled_binding_does_not_serve():
    workloads.register(Recorder())
    workloads.bind("triage", "recorder", enabled=False)
    assert workloads.kind_of("triage") == "llm"
    assert workloads._pre_call(_ctx()) is None


def test_binding_config_reaches_the_handler():
    h = Recorder()
    workloads.register(h)
    workloads.bind("triage", "recorder", installation="fabexp-1", budget_ms=250)
    workloads._pre_call(_ctx())
    assert h.seen[0].config == {"installation": "fabexp-1", "budget_ms": 250}


def test_binding_an_unregistered_kind_is_inert_not_an_error():
    """A binding may precede its handler; it starts serving when one appears."""
    workloads.bind("triage", "not-yet")
    assert workloads._pre_call(_ctx()) is None
    workloads.register(type("Late", (), {"kind": "not-yet",
                                         "serve": lambda s, r: WorkloadResult(text="late")})())
    assert workloads._pre_call(_ctx()).text == "late"


# -- dispatch ---------------------------------------------------------------


def test_a_served_workload_short_circuits_with_its_kind_as_the_source():
    workloads.register(Recorder(WorkloadResult(text="answer", cost_usd=0.01, model="m")))
    workloads.bind("triage", "recorder")
    sc = workloads._pre_call(_ctx())
    assert sc.text == "answer"
    # The kind lands in telemetry as both provider and source, so an
    # llm-served and a handler-served call are separable in the calls table.
    assert sc.provider == "recorder" and sc.source == "recorder"
    assert sc.cost_usd == 0.01


def test_an_unbound_workload_never_reaches_a_handler():
    h = Recorder()
    workloads.register(h)
    workloads.bind("triage", "recorder")
    assert workloads._pre_call(_ctx("other-workload")) is None
    assert h.seen == []


def test_declining_falls_through_to_the_provider_chain():
    workloads.register(Decliner())
    workloads.bind("triage", "decliner")
    assert workloads._pre_call(_ctx()) is None


def test_raising_is_declining():
    """An extension must never break the call path."""
    workloads.register(Exploder())
    workloads.bind("triage", "exploder")
    assert workloads._pre_call(_ctx()) is None


def test_a_handler_returning_the_wrong_type_is_ignored():
    workloads.register(type("Wrong", (), {"kind": "wrong",
                                          "serve": lambda s, r: "just a string"})())
    workloads.bind("triage", "wrong")
    assert workloads._pre_call(_ctx()) is None


def test_the_request_exposes_content_for_prompt_and_messages():
    h = Recorder()
    workloads.register(h)
    workloads.bind("triage", "recorder")

    workloads._pre_call(_ctx())
    assert h.seen[-1].content == "do the thing"

    ctx = _ctx()
    ctx.prompt = None
    ctx.messages = [{"role": "user", "content": "hi"}]
    workloads._pre_call(ctx)
    assert "hi" in h.seen[-1].content


# -- integration with the real hook bus -------------------------------------


def test_registering_installs_dispatch_on_the_real_pre_call_phase():
    workloads.register(Recorder(WorkloadResult(text="served")))
    workloads.bind("triage", "recorder")
    sc = hooks.fire_pre_call(_ctx())
    assert sc is not None and sc.text == "served"


def test_an_unbound_workload_reaches_the_provider_through_the_real_bus():
    workloads.register(Recorder())
    workloads.bind("triage", "recorder")
    assert hooks.fire_pre_call(_ctx("unbound")) is None


def test_unregistering_a_handler_returns_its_workloads_to_the_chain():
    workloads.register(Recorder())
    workloads.bind("triage", "recorder")
    workloads.unregister("recorder")
    assert hooks.fire_pre_call(_ctx()) is None

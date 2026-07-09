from __future__ import annotations

import asyncio
import time
from threading import Event

import pytest
from somm import hooks


@pytest.fixture(autouse=True)
def _reset_hooks():
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    saved_hooks = {
        phase: list(hooks._hooks_by_phase[phase]) for phase in hooks.HOOK_PHASES
    }
    saved_index = hooks._next_insertion_index
    saved_entry_points_loaded = hooks._entry_points_loaded
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase].clear()
    hooks._next_insertion_index = 0
    hooks._entry_points_loaded = False
    yield
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase][:] = saved_hooks[phase]
    hooks._next_insertion_index = saved_index
    hooks._entry_points_loaded = saved_entry_points_loaded


def _ctx() -> hooks.PreCallContext:
    return hooks.PreCallContext(
        workload="unit",
        prompt="hello",
        system="system",
        messages=None,
        model="m",
        provider="p",
        max_tokens=10,
        temperature=0.1,
        tools=[],
        tool_choice=None,
        project="proj",
    )


def test_register_unregister_priority_ordering_and_invalid_phase():
    calls: list[str] = []

    def high(_event):
        calls.append("high")

    def low(_event):
        calls.append("low")

    hooks.register_hook(hooks.POST_CALL, high, priority=100)
    hooks.register_hook(hooks.POST_CALL, low, priority=10)

    hooks.fire_post_call({})

    assert calls == ["low", "high"]

    hooks.unregister_hook(hooks.POST_CALL, low)
    calls.clear()
    hooks.fire_post_call({})
    assert calls == ["high"]

    with pytest.raises(ValueError):
        hooks.register_hook("not_a_phase", high)


def test_pre_call_mutation_is_visible_on_context():
    def rewrite(ctx: hooks.PreCallContext):
        ctx.prompt = "redacted"
        ctx.metadata["rewritten"] = True
        with pytest.raises(AttributeError):
            ctx.project = "other"

    ctx = _ctx()
    hooks.register_hook(hooks.PRE_CALL, rewrite)

    result = hooks.fire_pre_call(ctx)

    assert result is None
    assert ctx.prompt == "redacted"
    assert ctx.metadata == {"rewritten": True}


def test_pre_call_short_circuit_first_result_wins_and_skips_later_hooks():
    calls: list[str] = []

    def miss(_ctx):
        calls.append("miss")

    def hit(_ctx):
        calls.append("hit")
        return hooks.ShortCircuit(text="cached", source="cache")

    def skipped(_ctx):
        calls.append("skipped")
        return hooks.ShortCircuit(text="later", source="later")

    hooks.register_hook(hooks.PRE_CALL, miss, priority=10)
    hooks.register_hook(hooks.PRE_CALL, hit, priority=20)
    hooks.register_hook(hooks.PRE_CALL, skipped, priority=30)

    result = hooks.fire_pre_call(_ctx())

    assert result == hooks.ShortCircuit(text="cached", source="cache")
    assert calls == ["miss", "hit"]


def test_pre_call_raising_hook_is_isolated_and_does_not_short_circuit():
    calls: list[str] = []

    def broken(ctx: hooks.PreCallContext):
        calls.append("broken")
        ctx.prompt = "mutated-before-raise"
        raise RuntimeError("hook bug")

    def later(ctx: hooks.PreCallContext):
        calls.append("later")
        assert ctx.prompt == "mutated-before-raise"

    ctx = _ctx()
    hooks.register_hook(hooks.PRE_CALL, broken, priority=10)
    hooks.register_hook(hooks.PRE_CALL, later, priority=20)

    assert hooks.fire_pre_call(ctx) is None
    assert calls == ["broken", "later"]
    assert ctx.prompt == "mutated-before-raise"


def test_async_pre_call_registration_raises_value_error():
    async def async_hook(_ctx):
        return None

    with pytest.raises(ValueError, match="pre_call hooks must be synchronous"):
        hooks.register_hook(hooks.PRE_CALL, async_hook)


def test_post_call_fires_all_hooks_with_isolation():
    calls: list[str] = []

    def broken(_event):
        calls.append("broken")
        raise RuntimeError("hook bug")

    def later(_event):
        calls.append("later")

    hooks.register_hook(hooks.POST_CALL, broken)
    hooks.register_hook(hooks.POST_CALL, later)

    hooks.fire_post_call({"call_id": "c"})

    assert calls == ["broken", "later"]


def test_call_observer_compatibility_fires_once_through_notify():
    calls: list[dict] = []

    hooks.add_call_observer(calls.append)
    hooks.add_call_observer(calls.append)
    hooks.notify_call_observers({"call_id": "c"})

    assert calls == [{"call_id": "c"}]

    hooks.remove_call_observer(calls.append)
    hooks.notify_call_observers({"call_id": "d"})
    assert calls == [{"call_id": "c"}]


def test_post_process_runs_on_background_executor_and_shutdown_drains():
    started = Event()
    finished = Event()

    def slow(_event):
        started.set()
        time.sleep(0.25)
        finished.set()

    hooks.register_hook(hooks.POST_PROCESS, slow)

    before = time.monotonic()
    hooks.fire_post_process({"call_id": "c"})
    elapsed = time.monotonic() - before

    assert elapsed < 0.15
    assert started.wait(1)
    assert finished.wait(1)

    hooks.shutdown_hooks(wait=True)
    assert hooks._post_process_executor is None


def test_post_process_isolates_errors_and_continues():
    finished = Event()

    def broken(_event):
        raise RuntimeError("hook bug")

    def later(_event):
        finished.set()

    hooks.register_hook(hooks.POST_PROCESS, broken, priority=10)
    hooks.register_hook(hooks.POST_PROCESS, later, priority=20)

    hooks.fire_post_process({"call_id": "c"})

    assert finished.wait(1)
    hooks.shutdown_hooks(wait=True)


def test_async_post_call_hook_executes_and_exceptions_are_isolated():
    calls: list[str] = []

    async def async_hook(_event):
        await asyncio.sleep(0)
        calls.append("async")

    async def broken(_event):
        await asyncio.sleep(0)
        raise RuntimeError("hook bug")

    def sync_hook(_event):
        calls.append("sync")

    hooks.register_hook(hooks.POST_CALL, async_hook, priority=10)
    hooks.register_hook(hooks.POST_CALL, broken, priority=20)
    hooks.register_hook(hooks.POST_CALL, sync_hook, priority=30)

    hooks.fire_post_call({"call_id": "c"})

    assert calls == ["async", "sync"]


def test_async_post_process_hook_executes_and_exceptions_are_isolated():
    finished = Event()

    async def async_hook(_event):
        await asyncio.sleep(0)
        finished.set()

    async def broken(_event):
        await asyncio.sleep(0)
        raise RuntimeError("hook bug")

    hooks.register_hook(hooks.POST_PROCESS, broken, priority=10)
    hooks.register_hook(hooks.POST_PROCESS, async_hook, priority=20)

    hooks.fire_post_process({"call_id": "c"})

    assert finished.wait(1)
    hooks.shutdown_hooks(wait=True)


def test_stamp_event_sets_schema_version_without_mutating_original():
    event = {"call_id": "c", "schema_version": 999}

    stamped = hooks.stamp_event(event)

    assert stamped["schema_version"] == hooks.HOOK_EVENT_SCHEMA_VERSION == 1
    assert event["schema_version"] == 999


def test_registered_hooks_reflects_registrations():
    def hook_a(_ctx):
        return None

    def hook_b(_event):
        return None

    hooks.register_hook(hooks.PRE_CALL, hook_a, priority=5)
    hooks.register_hook(hooks.POST_CALL, hook_b, priority=15)

    registered = hooks.registered_hooks()

    assert registered[hooks.PRE_CALL] == [
        (f"{hook_a.__module__}.{hook_a.__qualname__}", 5)
    ]
    assert registered[hooks.POST_CALL] == [
        (f"{hook_b.__module__}.{hook_b.__qualname__}", 15)
    ]
    assert registered[hooks.POST_PROCESS] == []

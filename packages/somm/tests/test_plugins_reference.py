from __future__ import annotations

import builtins
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from somm import hooks
from somm.client import SommLLM
from somm.plugins import cache, notifier, otel_exporter, redaction
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.generate_calls = 0
        self.last_request = None

    def generate(self, request):
        self.generate_calls += 1
        self.last_request = request
        return SommResponse(
            text=self.text,
            model=request.model or "fake-model",
            tokens_in=7,
            tokens_out=3,
            latency_ms=5,
            raw={"prompt": request.prompt, "messages": request.messages},
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


@pytest.fixture(autouse=True)
def _reset_hooks(monkeypatch):
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
    hooks._entry_points_loaded = True
    monkeypatch.setattr(hooks, "load_entry_points", lambda: None)
    cache.unregister()
    cache.clear()
    redaction.unregister()
    notifier.unregister()
    otel_exporter.unregister()
    yield
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    cache.unregister()
    cache.clear()
    redaction.unregister()
    notifier.unregister()
    otel_exporter.unregister()
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase][:] = saved_hooks[phase]
    hooks._next_insertion_index = saved_index
    hooks._entry_points_loaded = saved_entry_points_loaded


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "plugins"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _llm(tmp_path: Path, provider: FakeProvider) -> SommLLM:
    return SommLLM(config=_tmp_config(tmp_path), providers=[provider], on_error=lambda _: None)


def _install_fake_otel(monkeypatch):
    state = SimpleNamespace(providers=[], exporters=[], processors=[])

    class FakeStatusCode:
        ERROR = "error"

    class FakeStatus:
        def __init__(self, code):
            self.code = code

    class FakeSpan:
        def __init__(self, name):
            self.name = name
            self.attributes = {}
            self.status = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

    class FakeTracer:
        def __init__(self):
            self.spans = []

        def start_as_current_span(self, name):
            span = FakeSpan(name)
            self.spans.append(span)
            return span

    class FakeTracerProvider:
        def __init__(self, *, shutdown_on_exit=True):
            self.shutdown_on_exit = shutdown_on_exit
            self.tracer = FakeTracer()
            self.processors = []
            self.force_flush_calls = 0
            self.shutdown_calls = 0
            state.providers.append(self)

        def add_span_processor(self, processor):
            self.processors.append(processor)

        def get_tracer(self, _name):
            return self.tracer

        def force_flush(self):
            self.force_flush_calls += 1
            return True

        def shutdown(self):
            self.shutdown_calls += 1

    class FakeExporter:
        def __init__(self, *, endpoint):
            self.endpoint = endpoint
            self.shutdown_calls = 0
            state.exporters.append(self)

        def shutdown(self):
            self.shutdown_calls += 1

    class FakeBatchSpanProcessor:
        def __init__(self, exporter):
            self.exporter = exporter
            self.shutdown_calls = 0
            state.processors.append(self)

        def shutdown(self):
            self.shutdown_calls += 1

    fallback_tracer = FakeTracer()
    trace_api = SimpleNamespace(get_tracer=lambda _name: fallback_tracer)
    monkeypatch.setattr(
        otel_exporter,
        "_load_trace_api",
        lambda: (trace_api, FakeStatus, FakeStatusCode),
    )
    monkeypatch.setattr(
        otel_exporter,
        "_load_otlp_components",
        lambda: (FakeTracerProvider, FakeExporter, FakeBatchSpanProcessor),
    )
    state.provider_cls = FakeTracerProvider
    state.fallback_tracer = fallback_tracer
    return state


def test_cache_wrap_populates_and_proxy_hit_skips_provider(tmp_path):
    cache.register(ttl_s=300, maxsize=512)
    fake = FakeProvider(text="cached-body")
    llm = _llm(tmp_path, fake)
    try:
        cached = cache.wrap(llm)
        first = cached.generate("hello", workload="cache", model="m")
        second = cached.generate("hello", workload="cache", model="m")
    finally:
        llm.close()

    assert first.text == "cached-body"
    assert second.text == "cached-body"
    assert second.provider == "cache"
    assert fake.generate_calls == 1


def test_cache_populated_entry_short_circuits_real_llm_path(tmp_path):
    cache.register(ttl_s=300, maxsize=512)
    fake = FakeProvider(text="cached-body")
    llm = _llm(tmp_path, fake)
    try:
        cache.wrap(llm).generate("hello", workload="cache")
        fake.text = "provider-body"
        direct = llm.generate("hello", workload="cache")
    finally:
        llm.close()

    assert direct.text == "cached-body"
    assert fake.generate_calls == 1


def test_cache_ttl_expiry_calls_provider_again(tmp_path):
    cache.register(ttl_s=0.01, maxsize=512)
    fake = FakeProvider(text="one")
    llm = _llm(tmp_path, fake)
    try:
        cached = cache.wrap(llm)
        assert cached.generate("hello", workload="cache").text == "one"
        time.sleep(0.03)
        fake.text = "two"
        assert cached.generate("hello", workload="cache").text == "two"
    finally:
        llm.close()

    assert fake.generate_calls == 2


def test_cache_maxsize_lru_eviction(tmp_path):
    cache.register(ttl_s=300, maxsize=1)
    fake = FakeProvider(text="one")
    llm = _llm(tmp_path, fake)
    try:
        cached = cache.wrap(llm)
        cached.generate("a", workload="cache")
        fake.text = "two"
        cached.generate("b", workload="cache")
        fake.text = "three"
        assert cached.generate("a", workload="cache").text == "three"
    finally:
        llm.close()

    assert fake.generate_calls == 3


def test_cache_workload_filter(tmp_path):
    cache.register(ttl_s=300, maxsize=512, workloads={"allowed"})
    fake = FakeProvider(text="one")
    llm = _llm(tmp_path, fake)
    try:
        cached = cache.wrap(llm)
        cached.generate("hello", workload="blocked")
        fake.text = "two"
        assert cached.generate("hello", workload="blocked").text == "two"
        cached.generate("hello", workload="allowed")
        fake.text = "three"
        assert cached.generate("hello", workload="allowed").text == "two"
    finally:
        llm.close()

    assert fake.generate_calls == 3


def test_redaction_prompt_secret_and_email_before_provider(tmp_path):
    redaction.register()
    fake = FakeProvider()
    llm = _llm(tmp_path, fake)
    try:
        llm.generate(
            "key sk-testsecret123 and marc@example.com",
            system="notify ops@example.com",
            workload="redact",
        )
    finally:
        llm.close()

    assert fake.last_request.prompt == "key [redacted] and [redacted]"
    assert fake.last_request.system == "notify [redacted]"


def test_redaction_messages_path(tmp_path):
    redaction.register()
    fake = FakeProvider()
    llm = _llm(tmp_path, fake)
    try:
        llm.generate(
            "ignored",
            workload="redact",
            messages=[
                {
                    "role": "user",
                    "content": "email me@example.com card 4242 4242 4242 4242",
                }
            ],
        )
    finally:
        llm.close()

    assert fake.last_request.messages[0]["content"] == "email [redacted] card [redacted]"


def test_redaction_noop_on_clean_text(tmp_path):
    redaction.register()
    fake = FakeProvider()
    llm = _llm(tmp_path, fake)
    try:
        llm.generate("clean prompt", workload="redact")
    finally:
        llm.close()

    assert fake.last_request.prompt == "clean prompt"


def test_notifier_error_event_posts(monkeypatch):
    calls: list[dict] = []

    class Response:
        def raise_for_status(self):
            return None

    def post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(notifier.httpx, "post", post)
    notifier.register("https://example.test/hook", timeout_s=1.5)

    hooks.fire_post_process(
        {
            "call_id": "c1",
            "workload": "w",
            "provider": "p",
            "model": "m",
            "outcome": "upstream_error",
            "cost_usd": 0.0,
        }
    )
    hooks.shutdown_hooks(wait=True)

    assert calls == [
        {
            "url": "https://example.test/hook",
            "json": {"text": "somm upstream_error workload=w provider=p model=m call_id=c1"},
            "timeout": 1.5,
        }
    ]


def test_notifier_ok_event_does_not_post(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(notifier.httpx, "post", lambda *a, **k: calls.append({}))
    notifier.register("https://example.test/hook")

    hooks.fire_post_process({"outcome": "ok", "cost_usd": 0.0})
    hooks.shutdown_hooks(wait=True)

    assert calls == []


def test_notifier_network_failure_is_swallowed(monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(notifier.httpx, "post", boom)
    notifier.register("https://example.test/hook")

    hooks.fire_post_process({"outcome": "upstream_error"})
    hooks.shutdown_hooks(wait=True)

    assert "somm notifier webhook failed" in caplog.text


def test_otel_exporter_entry_point_and_extra_are_packaged():
    package_config = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert package_config["project"]["entry-points"]["somm.plugins"] == {
        "otel_exporter": "somm.plugins.otel_exporter:register_from_env"
    }
    assert "opentelemetry-exporter-otlp-proto-http>=1.20" in (
        package_config["project"]["optional-dependencies"]["otel"]
    )


def test_otel_register_from_env_unset_is_dependency_and_thread_free(monkeypatch):
    monkeypatch.delenv("SOMM_OTEL_ENDPOINT", raising=False)
    before_hooks = hooks.registered_hooks()
    before_executor = hooks._post_process_executor

    def unexpected_import():
        raise AssertionError("unset endpoint must not load OpenTelemetry")

    monkeypatch.setattr(otel_exporter, "_load_trace_api", unexpected_import)
    monkeypatch.setattr(otel_exporter, "_load_otlp_components", unexpected_import)

    otel_exporter.register_from_env()

    assert hooks.registered_hooks() == before_hooks
    assert hooks._post_process_executor is before_executor
    assert otel_exporter._owned_tracer_provider is None


def test_otel_register_from_env_constructs_provider_and_emits_genai_span(monkeypatch):
    endpoint = "https://collector.example.test/custom/traces"
    monkeypatch.setenv("SOMM_OTEL_ENDPOINT", endpoint)
    state = _install_fake_otel(monkeypatch)

    otel_exporter.register_from_env()
    hooks.fire_post_process(
        {
            "call_id": "c-env",
            "project": "proj",
            "workload": "configured",
            "provider": "openai",
            "model": "gpt-test",
            "outcome": "ok",
            "tokens_in": 13,
            "tokens_out": 5,
            "cost_usd": 0.025,
        }
    )
    hooks.shutdown_hooks(wait=True)

    assert len(state.providers) == 1
    assert state.providers[0].shutdown_on_exit is False
    assert state.providers[0].processors == [state.processors[0]]
    assert state.processors[0].exporter is state.exporters[0]
    assert state.exporters[0].endpoint == endpoint
    assert len(state.providers[0].tracer.spans) == 1
    span = state.providers[0].tracer.spans[0]
    assert span.name == "llm configured"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-test"
    assert span.attributes["gen_ai.response.model"] == "gpt-test"
    assert span.attributes["gen_ai.usage.input_tokens"] == 13
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert span.attributes["somm.call_id"] == "c-env"


def test_otel_register_from_env_is_idempotent_and_cleans_up_owned_provider(monkeypatch):
    monkeypatch.setenv("SOMM_OTEL_ENDPOINT", "https://collector.example.test/v1/traces")
    state = _install_fake_otel(monkeypatch)

    otel_exporter.register_from_env()
    otel_exporter.register_from_env()

    active = [
        name
        for name, _priority in hooks.registered_hooks()[hooks.POST_PROCESS]
        if name.endswith("otel_exporter._post_process")
    ]
    assert len(state.providers) == 1
    assert len(state.exporters) == 1
    assert len(state.processors) == 1
    assert len(active) == 1

    provider = state.providers[0]
    otel_exporter.unregister()
    otel_exporter.unregister()

    assert provider.force_flush_calls == 1
    assert provider.shutdown_calls == 1
    assert hooks.registered_hooks()[hooks.POST_PROCESS] == []


def test_otel_manual_registration_cycles_do_not_close_caller_provider(monkeypatch):
    state = _install_fake_otel(monkeypatch)
    provider = state.provider_cls()

    otel_exporter.register(tracer_provider=provider)
    otel_exporter.register(tracer_provider=provider)
    assert len(hooks.registered_hooks()[hooks.POST_PROCESS]) == 1

    otel_exporter.unregister()
    otel_exporter.unregister()
    assert provider.force_flush_calls == 0
    assert provider.shutdown_calls == 0

    otel_exporter.register(tracer_provider=provider)
    assert len(hooks.registered_hooks()[hooks.POST_PROCESS]) == 1
    otel_exporter.unregister()
    assert provider.force_flush_calls == 0
    assert provider.shutdown_calls == 0


def test_otel_register_from_env_reports_missing_optional_dependencies(monkeypatch):
    monkeypatch.setenv("SOMM_OTEL_ENDPOINT", "https://collector.example.test/v1/traces")
    real_import = builtins.__import__

    def without_opentelemetry(name, *args, **kwargs):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("simulated missing OpenTelemetry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_opentelemetry)

    with pytest.raises(ImportError, match=r"pip install somm\[otel\]"):
        otel_exporter.register_from_env()

    assert hooks.registered_hooks()[hooks.POST_PROCESS] == []
    assert otel_exporter._owned_tracer_provider is None


def test_otel_exporter_emits_span():
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ImportError:
        pytest.skip("OpenTelemetry in-memory exporter unavailable")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_exporter.register(tracer_provider=provider)

    hooks.fire_post_process(
        {
            "call_id": "c1",
            "project": "proj",
            "workload": "w",
            "provider": "openai",
            "model": "gpt-test",
            "outcome": "ok",
            "tokens_in": 11,
            "tokens_out": 7,
            "cost_usd": 0.012,
        }
    )
    hooks.shutdown_hooks(wait=True)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "llm w"
    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-test"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["somm.call_id"] == "c1"
    assert attrs["somm.outcome"] == "ok"
    assert attrs["somm.cost_usd"] == 0.012


def test_cache_key_isolates_by_provider_tool_choice_and_caps():
    base = dict(
        workload="w", model=None, system="", prompt="hello",
        messages=None, temperature=0.2, max_tokens=256, tools=[],
    )
    k_openai = cache._cache_key(provider="openai", **base)
    k_anthropic = cache._cache_key(provider="anthropic", **base)
    assert k_openai != k_anthropic  # a pinned provider must not share a key

    k_tc_none = cache._cache_key(provider="openai", tool_choice="none", **base)
    k_tc_auto = cache._cache_key(provider="openai", tool_choice="auto", **base)
    assert k_tc_none != k_tc_auto


def test_redaction_does_not_mutate_caller_messages(tmp_path):
    redaction.register()
    fake = FakeProvider()
    llm = _llm(tmp_path, fake)
    original = [{"role": "user", "content": "secret sk-testsecret123 here"}]
    try:
        llm.generate("ignored", workload="redact", messages=original)
    finally:
        llm.close()

    # The provider saw redacted content...
    assert "[redacted]" in fake.last_request.messages[0]["content"]
    # ...but the caller's own object is untouched.
    assert original[0]["content"] == "secret sk-testsecret123 here"

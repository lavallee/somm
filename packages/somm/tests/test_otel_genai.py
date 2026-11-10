"""GenAI semantic-convention shape of the OpenTelemetry export.

Runs against the real SDK with an in-memory exporter; skipped when the
optional ``otel`` extra is not installed.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm import hooks
from somm.client import SommLLM
from somm.harnesses import HarnessOutcome, HarnessResult, record_attempt
from somm.plugins import otel_exporter
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

# Every gen_ai.* attribute the exporter may emit, each read from the registry
# and span definitions at the commit pinned in otel_exporter. Adding a name is
# a deliberate change: update this set, the docs page, and the pin together.
ALLOWED_GEN_AI_ATTRIBUTES = {
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.request.max_tokens",
    "gen_ai.request.temperature",
    "gen_ai.response.model",
    "gen_ai.response.id",
    "gen_ai.conversation.id",
    "gen_ai.agent.name",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_write.input_tokens",
    "gen_ai.usage.reasoning.output_tokens",
}


class SlowProvider:
    name = "gemini"

    def __init__(self, delay: float = 0.15, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail

    def generate(self, request):
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return SommResponse(
            text="hi",
            model=request.model or "m",
            tokens_in=11,
            tokens_out=4,
            latency_ms=int(self.delay * 1000),
            raw={"id": "resp-1"},
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_exporter.unregister()
    otel_exporter.register(tracer_provider=provider)
    yield exporter, provider
    hooks.shutdown_hooks(wait=True)
    otel_exporter.unregister()


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "otel"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def _event(**over):
    base = {
        "call_id": "c1",
        "project": "p",
        "workload": "w",
        "provider": "openai",
        "model": "gpt-test",
        "outcome": "ok",
        "tokens_in": 11,
        "tokens_out": 7,
        "cost_usd": 0.012,
    }
    base.update(over)
    return base


def test_llm_span_follows_genai_shape(spans):
    exporter, _ = spans
    hooks.fire_post_process(
        _event(
            provider_request_id="resp-9",
            correlation_id="conv-1",
            cache_tokens_in=3,
            cache_tokens_out=2,
            reasoning_tokens=5,
            max_tokens=256,
            temperature=0.2,
        )
    )
    hooks.shutdown_hooks(wait=True)

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-test"
    a = span.attributes
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "openai"
    assert "gen_ai.system" not in a
    assert a["gen_ai.request.model"] == a["gen_ai.response.model"] == "gpt-test"
    assert a["gen_ai.response.id"] == "resp-9"
    assert a["gen_ai.conversation.id"] == "conv-1"
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.cache_read.input_tokens"] == 3
    assert a["gen_ai.usage.cache_write.input_tokens"] == 2
    assert a["gen_ai.usage.reasoning.output_tokens"] == 5
    assert a["somm.cost_usd"] == 0.012 and a["somm.workload"] == "w"
    assert "error.type" not in a
    assert span.kind.name == "CLIENT"


def test_failure_sets_error_type_and_status(spans):
    exporter, _ = spans
    hooks.fire_post_process(_event(outcome="upstream_error", error_kind="HTTPStatusError"))
    hooks.fire_post_process(_event(outcome="timeout"))
    hooks.shutdown_hooks(wait=True)

    first, second = exporter.get_finished_spans()
    assert first.attributes["error.type"] == "HTTPStatusError"
    assert second.attributes["error.type"] == "timeout"
    assert first.status.status_code.name == "ERROR"


def test_emitted_attributes_stay_inside_the_pinned_allow_list(spans):
    exporter, _ = spans
    hooks.fire_post_process(
        _event(
            provider_request_id="r",
            correlation_id="c",
            cache_tokens_in=1,
            cache_tokens_out=1,
            reasoning_tokens=1,
            max_tokens=1,
            temperature=0.0,
            outcome="upstream_error",
            error_kind="X",
        )
    )
    hooks.shutdown_hooks(wait=True)
    (span,) = exporter.get_finished_spans()
    emitted = {k for k in span.attributes if k.startswith("gen_ai.")}
    assert emitted <= ALLOWED_GEN_AI_ATTRIBUTES, emitted - ALLOWED_GEN_AI_ATTRIBUTES
    # No prompt or response content is ever exported.
    assert not {k for k in emitted if "messages" in k or "instructions" in k}
    assert len(otel_exporter.GENAI_SEMCONV_COMMIT) == 40


def test_span_timing_and_parent_match_the_real_call(spans, tmp_path):
    exporter, provider = spans
    tracer = provider.get_tracer("test")
    llm = SommLLM(config=_cfg(tmp_path), providers=[SlowProvider(0.15)], on_error=lambda _: None)
    try:
        with tracer.start_as_current_span("caller") as caller:
            t0 = time.time_ns()
            llm.generate("hello", workload="timing", model="m")
            t1 = time.time_ns()
        hooks.shutdown_hooks(wait=True)
    finally:
        llm.close()

    by_name = {s.name: s for s in exporter.get_finished_spans()}
    call = by_name["chat m"]
    assert call.attributes["gen_ai.provider.name"] == "gcp.gemini"
    # Nests under the caller's active span, same trace.
    assert call.parent is not None
    assert call.parent.span_id == caller.get_span_context().span_id
    assert call.context.trace_id == caller.get_span_context().trace_id
    # Starts after the call began, ends before it returned, lasts about as long.
    assert t0 <= call.start_time and call.end_time <= t1
    assert 0.1e9 <= call.end_time - call.start_time <= 0.5e9


def test_harness_attempt_becomes_invoke_agent_span(spans, tmp_path):
    exporter, provider = spans
    tracer = provider.get_tracer("test")
    started = datetime.now(UTC)
    result = HarnessResult(
        harness="claude-cli",
        outcome=HarnessOutcome.RATE_LIMIT,
        session_id="sess-1",
        model="claude-x",
        duration_ms=2500,
        usage={"input_tokens": 100, "output_tokens": 40},
    )
    with tracer.start_as_current_span("runner") as runner:
        call = record_attempt(result, config=_cfg(tmp_path), started_at=started)

    span = next(s for s in exporter.get_finished_spans() if s.name.startswith("invoke_agent"))
    assert span.name == "invoke_agent claude-cli"
    a = span.attributes
    assert a["gen_ai.operation.name"] == "invoke_agent"
    assert a["gen_ai.agent.name"] == "claude-cli"
    assert a["gen_ai.conversation.id"] == "sess-1"
    assert a["gen_ai.request.model"] == "claude-x"
    assert a["error.type"] == "rate_limit"
    assert a["somm.call_id"] == call.id
    assert span.parent.span_id == runner.get_span_context().span_id
    assert span.end_time - span.start_time == 2500 * 1_000_000
    emitted = {k for k in a if k.startswith("gen_ai.")}
    assert emitted <= ALLOWED_GEN_AI_ATTRIBUTES

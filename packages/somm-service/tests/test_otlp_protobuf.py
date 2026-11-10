"""OTLP/HTTP protobuf ingest: somm's own exporter posting to somm's service."""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.proto")
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans  # noqa: E402
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # noqa: E402
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from somm import hooks  # noqa: E402
from somm.harnesses import HarnessOutcome, HarnessResult, record_attempt  # noqa: E402
from somm.plugins import otel_exporter  # noqa: E402
from somm_core.config import Config  # noqa: E402
from somm_service.app import create_app  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

PB = {"content-type": "application/x-protobuf"}


def _cfg(root, project):
    cfg = Config()
    cfg.project = project
    cfg.db_dir = root / project
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


@pytest.fixture
def svc(tmp_path):
    app = create_app(_cfg(tmp_path, "svc-test"))
    c = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
    return c, app, {"authorization": f"Bearer {app.state.service_token.value}", **PB}


@pytest.fixture
def exporter():
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    otel_exporter.unregister()
    otel_exporter.register(tracer_provider=provider)
    yield mem
    hooks.shutdown_hooks(wait=True)
    otel_exporter.unregister()


def test_exporter_spans_round_trip_as_protobuf(svc, exporter, tmp_path):
    c, app, headers = svc
    hooks.fire_post_process(
        {
            "call_id": "chat-1",
            "project": "p",
            "workload": "summarise",
            "provider": "gemini",
            "model": "gem-x",
            "outcome": "ok",
            "tokens_in": 11,
            "tokens_out": 7,
            "cost_usd": 0.012,
        }
    )
    hooks.shutdown_hooks(wait=True)
    result = HarnessResult(
        harness="claude-cli",
        outcome=HarnessOutcome.RATE_LIMIT,
        session_id="sess-1",
        model="claude-x",
        duration_ms=2500,
        usage={"input_tokens": 100, "output_tokens": 40},
    )
    harness_call = record_attempt(
        result, config=_cfg(tmp_path, "source"), started_at=datetime.now(UTC)
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    body = encode_spans(spans).SerializeToString()

    # The bearer token is still required.
    assert c.post("/v1/traces", content=body, headers=PB).status_code == 403
    res = c.post("/v1/traces", content=body, headers=headers)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/x-protobuf"
    assert not ExportTraceServiceResponse.FromString(res.content).HasField("partial_success")

    chat = app.state.repo.get_call("chat-1")
    assert (chat.provider, chat.model, chat.tokens_in, chat.tokens_out) == (
        "gcp.gemini",
        "gem-x",
        11,
        7,
    )
    assert chat.cost_usd == pytest.approx(0.012)
    harness = app.state.repo.get_call(harness_call.id)
    assert harness.latency_ms == 2500
    assert harness.session_id == "sess-1"
    assert harness.outcome.value == "upstream_error"
    rows = c.get("/api/calls", headers=headers).json()["calls"]
    assert {r["workload"] for r in rows} == {"summarise", "invoke_agent claude-cli"}

    # Re-sending the same spans does not double-count them.
    assert c.post("/api/otlp/v1/traces", content=body, headers=headers).status_code == 200
    assert len(c.get("/api/calls", headers=headers).json()["calls"]) == 2


def test_protobuf_garbage_and_oversize(svc):
    c, app, headers = svc
    assert c.post("/v1/traces", content=b"\xff\xff\xff", headers=headers).status_code == 400
    app.state.config.service_otlp_max_body_bytes = 8
    assert c.post("/v1/traces", content=b"x" * 64, headers=headers).status_code == 413


def test_protobuf_without_package_is_415_and_json_still_works(svc, monkeypatch):
    c, _, headers = svc
    monkeypatch.setitem(sys.modules, "opentelemetry.proto", None)
    res = c.post("/v1/traces", content=b"", headers=headers)
    assert res.status_code == 415
    assert "somm-service[otlp]" in res.json()["error"]
    as_json = {**headers, "content-type": "application/json"}
    assert c.post("/v1/traces", json={}, headers=as_json).status_code == 200

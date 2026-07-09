"""Tests for somm-service web + API."""

from __future__ import annotations

import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from somm_core import SCHEMA_VERSION, VERSION
from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository
from somm_service.app import create_app, load_service_token
from starlette.testclient import TestClient


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "svc-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


@pytest.fixture
def client(tmp_path):
    cfg = _tmp_config(tmp_path)
    app = create_app(cfg)
    return TestClient(app), cfg, app


def test_home_renders_empty_state(client):
    c, cfg, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "somm" in r.text
    assert cfg.project in r.text
    assert "NO DATA YET" in r.text or "HEALTHY" in r.text
    assert "Sessions and traces" in r.text
    assert "Recent calls" in r.text


def test_home_has_html_security_headers(client):
    c, _, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert r.headers["content-security-policy"] == "default-src 'none'; style-src 'unsafe-inline'"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_health_endpoint(client):
    c, cfg, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["project"] == cfg.project
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_api_version(client):
    c, cfg, _ = client
    r = c.get("/api/version")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == VERSION
    assert data["schema_version"] == SCHEMA_VERSION


def test_api_stats_empty(client):
    c, cfg, _ = client
    r = c.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["project"] == cfg.project
    assert data["rows"] == []


def test_status_calls_and_sessions_api(client):
    c, cfg, _ = client
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="trace_work", project=cfg.project)
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    for call_id, model, parent in (
        (parent_id, "parent-model", None),
        (child_id, "child-model", parent_id),
    ):
        repo.write_call(
            Call(
                id=call_id,
                ts=datetime.now(UTC),
                project=cfg.project,
                workload_id=wl.id,
                prompt_id=None,
                provider="fake",
                model=model,
                tokens_in=2,
                tokens_out=3,
                latency_ms=10,
                cost_usd=0.0,
                outcome=Outcome.OK,
                error_kind=None,
                prompt_hash=f"p-{model}",
                response_hash=f"r-{model}",
                session_id="session-1",
                parent_call_id=parent,
            )
        )

    status = c.get("/api/status").json()
    assert status["total_calls"] == 2
    assert status["health"] == "healthy"

    calls = c.get("/api/calls", params={"q": "child-model"}).json()
    assert calls["count"] == 1
    assert calls["calls"][0]["parent_call_id"] == parent_id

    sessions = c.get("/api/sessions").json()
    assert sessions["count"] == 1
    assert sessions["sessions"][0]["session_id"] == "session-1"
    assert sessions["sessions"][0]["n_calls"] == 2


def test_otlp_trace_ingest_records_call(client):
    c, cfg, app = client
    now_ns = int(datetime.now(UTC).timestamp() * 1_000_000_000)
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "svc"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace-otlp-1",
                                "spanId": "span-otlp-1",
                                "name": "chat",
                                "startTimeUnixNano": str(now_ns),
                                "endTimeUnixNano": str(now_ns + 25_000_000),
                                "attributes": [
                                    {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                                    {
                                        "key": "gen_ai.request.model",
                                        "value": {"stringValue": "gpt-4o-mini"},
                                    },
                                    {
                                        "key": "gen_ai.usage.input_tokens",
                                        "value": {"intValue": "7"},
                                    },
                                    {
                                        "key": "gen_ai.usage.output_tokens",
                                        "value": {"intValue": "3"},
                                    },
                                    {"key": "somm.workload", "value": {"stringValue": "otlp_chat"}},
                                    {"key": "somm.session_id", "value": {"stringValue": "sess-otlp"}},
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }

    unauth = c.post("/api/otlp/v1/traces", json=payload)
    assert unauth.status_code == 403

    token = app.state.service_token.value
    res = c.post(
        "/api/otlp/v1/traces",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert res.json()["ingested"] == 1

    calls = c.get("/api/calls", params={"q": "trace-otlp-1"}).json()["calls"]
    assert len(calls) == 1
    assert calls[0]["workload"] == "otlp_chat"
    assert calls[0]["provider"] == "openai"
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["tokens_in"] == 7
    assert calls[0]["tokens_out"] == 3
    assert calls[0]["latency_ms"] == 25
    assert calls[0]["session_id"] == "sess-otlp"


def test_home_with_calls_shows_healthy(client, tmp_path):
    """End-to-end: library writes → service reads same SQLite → web shows HEALTHY."""
    c, cfg, _ = client

    # Write a call via the library so the service sees it via the same SQLite.
    from somm.client import SommLLM
    from somm.providers.base import ProviderHealth, SommResponse

    class FakeProvider:
        name = "fake"

        def generate(self, request):
            return SommResponse(
                text="ok",
                model="fake-m",
                tokens_in=2,
                tokens_out=1,
                latency_ms=5,
                raw={},
            )

        def stream(self, request):  # pragma: no cover
            yield

        def health(self):
            return ProviderHealth(available=True)

        def models(self):
            return []

        def estimate_tokens(self, text, model):
            return 1

    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    llm.generate("hi", workload="svc_end_to_end")
    llm.close()

    r = c.get("/")
    assert r.status_code == 200
    assert "HEALTHY" in r.text
    assert "svc_end_to_end" in r.text


def test_xss_in_workload_name_is_escaped(client, tmp_path):
    """Load-bearing: Jinja autoescape equivalent for the hand-rolled template."""
    c, cfg, _ = client

    # Register a workload with an XSS payload as the name.
    from somm_core.repository import Repository

    repo = Repository(cfg.db_path)
    repo.register_workload(name="<script>alert(1)</script>", project=cfg.project)

    # Insert a dummy call so it shows in the table.
    import uuid
    from datetime import UTC, datetime

    from somm_core.models import Call, Outcome

    call = Call(
        id=str(uuid.uuid4()),
        ts=datetime.now(UTC),
        project=cfg.project,
        workload_id=repo.workload_by_name("<script>alert(1)</script>", cfg.project).id,
        prompt_id=None,
        provider="fake",
        model="fake-m",
        tokens_in=1,
        tokens_out=1,
        latency_ms=1,
        cost_usd=0.0,
        outcome=Outcome.OK,
        error_kind=None,
        prompt_hash="a",
        response_hash="b",
    )
    repo.write_call(call)

    r = c.get("/")
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;" in r.text


def test_service_token_file_created_0600(tmp_path, monkeypatch):
    monkeypatch.delenv("SOMM_SERVICE_TOKEN", raising=False)
    cfg = _tmp_config(tmp_path)

    service_token = load_service_token(cfg)

    token_path = cfg.db_dir / "service_token"
    assert service_token.created is True
    assert service_token.value
    assert token_path.read_text(encoding="utf-8").strip() == service_token.value
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_service_token_env_override_takes_precedence(tmp_path, monkeypatch):
    cfg = _tmp_config(tmp_path)
    token_path = cfg.db_dir / "service_token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("file-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setenv("SOMM_SERVICE_TOKEN", "env-token")

    service_token = load_service_token(cfg)

    assert service_token.value == "env-token"
    assert service_token.source == "env"
    assert token_path.read_text(encoding="utf-8").strip() == "file-token"


def test_service_token_never_empty_on_zero_byte_file(tmp_path, monkeypatch):
    """A pre-existing empty token file must not yield an empty token — a
    bare 'Bearer ' would otherwise authenticate."""
    monkeypatch.delenv("SOMM_SERVICE_TOKEN", raising=False)
    cfg = _tmp_config(tmp_path)
    token_path = cfg.db_dir / "service_token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("", encoding="utf-8")  # zero-byte: a half-written race

    service_token = load_service_token(cfg)

    assert service_token.value  # non-empty
    assert token_path.read_text(encoding="utf-8").strip() == service_token.value


@pytest.mark.parametrize(
    "host,ok",
    [
        ("localhost", True),
        ("localhost:7878", True),
        ("127.0.0.1", True),
        ("127.0.0.1:7878", True),
        ("127.5.5.5", True),
        ("[::1]", True),
        ("[::1]:7878", True),
        ("127.0.0.1.attacker.com", False),  # the rebinding-name bypass
        ("localhost.attacker.com", False),
        ("[::1].attacker.com", False),  # malformed-bracket bypass
        ("[::1]junk", False),
        ("[127.0.0.1].attacker.com", False),
        ("[::1", False),  # unclosed bracket
        ("attacker.example", False),
        ("10.0.0.5", False),
        ("", False),
    ],
)
def test_host_is_loopback_rejects_rebinding_names(host, ok):
    from somm_service.app import _host_is_loopback

    assert _host_is_loopback(host) is ok

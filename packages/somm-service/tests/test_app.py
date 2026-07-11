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

_LOCAL_HEADERS = {"x-somm-local": "1", "sec-fetch-site": "same-origin"}


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
    return TestClient(app, base_url="http://localhost"), cfg, app


def _auth_headers(app, **extra: str) -> dict[str, str]:
    return {"authorization": f"Bearer {app.state.service_token.value}", **extra}


def test_home_renders_empty_state(client):
    c, cfg, app = client
    assert c.get("/").status_code == 403
    r = c.get("/", headers=_LOCAL_HEADERS)
    assert c.get("/", headers=_auth_headers(app)).status_code == 200
    assert r.status_code == 200
    assert "somm" in r.text
    assert cfg.project in r.text
    assert "NO DATA YET" in r.text or "HEALTHY" in r.text
    assert "Sessions and traces" in r.text
    assert "Recent calls" in r.text


def test_home_has_html_security_headers(client):
    c, _, _ = client
    r = c.get("/", headers=_LOCAL_HEADERS)
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
    assert "db_path" not in data
    assert data["db_exists"] is True
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_api_version(client):
    c, cfg, app = client
    assert c.get("/api/version").status_code == 403
    r = c.get("/api/version", headers=_auth_headers(app))
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == VERSION
    assert data["schema_version"] == SCHEMA_VERSION


def test_api_stats_empty(client):
    c, cfg, app = client
    assert c.get("/api/stats").status_code == 403

    r = c.get("/api/stats", headers=_auth_headers(app))
    assert r.status_code == 200
    local = c.get("/api/stats", headers=_LOCAL_HEADERS)
    assert local.status_code == 200
    data = r.json()
    assert data["project"] == cfg.project
    assert data["rows"] == []


def test_api_spend_today_reports_current_project_rows(client):
    c, cfg, app = client
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(
        name="daily_spend",
        project=cfg.project,
        budget_cap_usd_daily=1.5,
    )
    repo.write_call(
        Call(
            id=str(uuid.uuid4()),
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="fake",
            model="m",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50,
            cost_usd=0.25,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )

    assert c.get("/api/spend/today").status_code == 403
    r = c.get("/api/spend/today", headers=_auth_headers(app))

    assert r.status_code == 200
    assert r.json() == {
        "project": cfg.project,
        "rows": [{"workload": "daily_spend", "spent_usd": 0.25, "cap_usd": 1.5}],
    }


def test_api_stats_includes_serving_profile(client):
    c, cfg, app = client
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(
        name="svc_profile",
        project=cfg.project,
        max_p95_latency_ms=75,
        max_p95_ttft_ms=15,
        max_tpot_ms=10.0,
    )
    repo.write_call(
        Call(
            id=str(uuid.uuid4()),
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="fake",
            model="m",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
            ttft_ms=10,
            cache_tokens_in=4,
            cache_tokens_out=1,
        )
    )

    r = c.get("/api/stats", headers=_auth_headers(app))
    assert r.status_code == 200
    row = r.json()["rows"][0]
    assert row["p95_latency_ms"] == 50
    assert row["p99_ttft_ms"] == 10
    assert row["tpot_ms"] == 10
    assert row["input_tokens_per_second"] == 200
    assert row["requests_per_second"] == 20
    assert row["cache_read_ratio"] == 0.4
    assert row["goodput_slo_latency_ms"] == 75
    assert row["goodput_slo_ttft_ms"] == 15
    assert row["goodput_slo_tpot_ms"] == 10.0
    assert row["goodput_under_slo"] == 1.0
    assert row["goodput_requests_per_second"] == 20
    assert row["goodput_output_tokens_per_second"] == 100

    home = c.get("/", headers=_LOCAL_HEADERS)
    assert "p95 ms" in home.text
    assert "ttft p95" in home.text
    assert "cache" in home.text


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

    status = c.get("/api/status", headers=_LOCAL_HEADERS).json()
    assert status["total_calls"] == 2
    assert status["health"] == "healthy"
    assert status["load"]["calls_per_minute"] == 2
    assert status["load"]["active_workloads"] == 1
    assert status["load"]["active_models"] == 2
    assert status["load"]["output_tokens_per_minute"] == 6

    calls = c.get("/api/calls", params={"q": "child-model"}, headers=_LOCAL_HEADERS).json()
    assert calls["count"] == 1
    assert calls["calls"][0]["parent_call_id"] == parent_id

    sessions = c.get("/api/sessions", headers=_LOCAL_HEADERS).json()
    assert sessions["count"] == 1
    assert sessions["sessions"][0]["session_id"] == "session-1"
    assert sessions["sessions"][0]["n_calls"] == 2


def test_non_loopback_host_with_local_header_cannot_read_api(client):
    _, _, app = client
    remote = TestClient(app, base_url="http://example.com")

    r = remote.get("/api/calls", headers=_LOCAL_HEADERS)

    assert r.status_code == 403


def _otlp_payload(*, trace_id: str = "trace-otlp-1", span_id: str = "span-otlp-1") -> dict:
    now_ns = int(datetime.now(UTC).timestamp() * 1_000_000_000)
    return {
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
                                "traceId": trace_id,
                                "spanId": span_id,
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


def test_otlp_trace_ingest_records_call(client):
    c, cfg, app = client
    payload = _otlp_payload()

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

    calls = c.get(
        "/api/calls",
        params={"q": "trace-otlp-1"},
        headers=_LOCAL_HEADERS,
    ).json()["calls"]
    assert len(calls) == 1
    assert calls[0]["workload"] == "otlp_chat"
    assert calls[0]["provider"] == "openai"
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["tokens_in"] == 7
    assert calls[0]["tokens_out"] == 3
    assert calls[0]["latency_ms"] == 25
    assert calls[0]["session_id"] == "sess-otlp"


def test_otlp_rejects_oversized_content_length_without_writes(tmp_path):
    cfg = _tmp_config(tmp_path)
    cfg.service_otlp_max_body_bytes = 8
    app = create_app(cfg)
    c = TestClient(app, base_url="http://localhost")

    r = c.post(
        "/api/otlp/v1/traces",
        content=b'{"spans":[]}',
        headers={
            "authorization": f"Bearer {app.state.service_token.value}",
            "content-type": "application/json",
            "content-length": "999",
        },
    )

    assert r.status_code == 413
    with app.state.repo._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


def test_otlp_rejects_over_span_cap_without_partial_writes(tmp_path):
    cfg = _tmp_config(tmp_path)
    cfg.service_otlp_max_spans = 1
    app = create_app(cfg)
    c = TestClient(app, base_url="http://localhost")
    payload = {"spans": []}
    for idx in range(2):
        payload["spans"].append(_otlp_payload(trace_id=f"trace-{idx}")["resourceSpans"][0]["scopeSpans"][0]["spans"][0])

    r = c.post(
        "/api/otlp/v1/traces",
        json=payload,
        headers=_auth_headers(app),
    )

    assert r.status_code == 413
    assert r.json()["max_spans"] == 1
    with app.state.repo._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0


def test_otlp_malformed_spans_skip_and_attrs_are_bounded(tmp_path):
    cfg = _tmp_config(tmp_path)
    cfg.service_otlp_max_attr_chars = 5
    app = create_app(cfg)
    c = TestClient(app, base_url="http://localhost")
    span = _otlp_payload(trace_id="trace-bounds")["resourceSpans"][0]["scopeSpans"][0][
        "spans"
    ][0]
    span["attributes"].extend(
        [
            {"key": "somm.model", "value": {"stringValue": "very-long-model"}},
            {"key": "somm.workload", "value": {"stringValue": "very-long-workload"}},
        ]
    )

    r = c.post(
        "/api/otlp/v1/traces",
        json={"spans": [span, "not-a-span"]},
        headers=_auth_headers(app),
    )

    assert r.status_code == 200
    assert r.json()["ingested"] == 1
    calls = c.get("/api/calls", headers=_LOCAL_HEADERS).json()["calls"]
    assert len(calls) == 1
    assert calls[0]["model"] == "very-"
    assert calls[0]["workload"] == "very-"


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

    r = c.get("/", headers=_LOCAL_HEADERS)
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

    r = c.get("/", headers=_LOCAL_HEADERS)
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


def test_created_service_token_log_does_not_print_secret(tmp_path, capsys):
    from somm_service.app import _log_service_token

    cfg = _tmp_config(tmp_path)
    service_token = load_service_token(cfg)

    _log_service_token(service_token, host="127.0.0.1", port=7878)

    out = capsys.readouterr().out
    assert service_token.value not in out
    assert str(service_token.path) in out
    assert "$(cat " in out


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

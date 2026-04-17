"""Tests for somm-service web + API."""

from __future__ import annotations

from pathlib import Path

import pytest
from somm_core.config import Config
from somm_service.app import create_app
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


def test_health_endpoint(client):
    c, cfg, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["project"] == cfg.project


def test_api_version(client):
    c, cfg, _ = client
    r = c.get("/api/version")
    assert r.status_code == 200
    data = r.json()
    assert data["version"].startswith("0.")
    assert data["schema_version"] >= 1


def test_api_stats_empty(client):
    c, cfg, _ = client
    r = c.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["project"] == cfg.project
    assert data["rows"] == []


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

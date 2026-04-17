"""Web admin recommendations rendering + dismiss/apply endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from somm_core.config import Config
from somm_service.app import create_app
from starlette.testclient import TestClient


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "recs-test"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


@pytest.fixture
def client_with_rec(tmp_path):
    cfg = _tmp_config(tmp_path)
    app = create_app(cfg)
    c = TestClient(app)

    # Seed a workload + recommendation
    repo = app.state.repo
    wl = repo.register_workload(name="demo_w", project=cfg.project)
    evidence = {
        "workload": "demo_w",
        "current": {
            "provider": "ollama",
            "model": "slow",
            "score": 0.4,
            "cost_usd": 0.0,
            "latency_ms": 500,
        },
        "candidate": {
            "provider": "ollama",
            "model": "fast",
            "score": 0.85,
            "cost_usd": 0.0,
            "latency_ms": 100,
        },
        "score_delta": 0.45,
        "n_evals": 8,
    }
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO recommendations "
            "(workload_id, action, evidence_json, expected_impact, confidence) "
            "VALUES (?, 'switch_model', ?, '+45% quality, -80% latency', 0.85)",
            (wl.id, json.dumps(evidence)),
        )
    return c, cfg, app


def test_home_renders_recommendation(client_with_rec):
    c, _, _ = client_with_rec
    r = c.get("/")
    assert r.status_code == 200
    assert "demo_w" in r.text
    assert "switch_model" in r.text
    assert "confidence" in r.text
    assert "+45%" in r.text
    assert "slow" in r.text
    assert "fast" in r.text


def test_api_recommendations_json(client_with_rec):
    c, cfg, _ = client_with_rec
    r = c.get("/api/recommendations")
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) == 1
    rec = data["recommendations"][0]
    assert rec["workload"] == "demo_w"
    assert rec["action"] == "switch_model"
    assert rec["evidence"]["candidate"]["model"] == "fast"


def test_dismiss_rec(client_with_rec):
    c, cfg, app = client_with_rec
    # Grab id
    data = c.get("/api/recommendations").json()
    rec_id = data["recommendations"][0]["id"]

    r = c.post(f"/api/recommendations/{rec_id}/dismiss")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # No longer open
    after = c.get("/api/recommendations").json()
    assert after["recommendations"] == []


def test_apply_rec(client_with_rec):
    c, cfg, app = client_with_rec
    data = c.get("/api/recommendations").json()
    rec_id = data["recommendations"][0]["id"]

    r = c.post(f"/api/recommendations/{rec_id}/apply")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Verify applied_at set
    repo = app.state.repo
    with repo._open() as conn:
        row = conn.execute(
            "SELECT applied_at FROM recommendations WHERE id = ?",
            (rec_id,),
        ).fetchone()
    assert row[0] is not None


def test_xss_in_recommendation_evidence_is_escaped(tmp_path):
    """Workload names + evidence fields render safely even with <script> payloads."""
    cfg = _tmp_config(tmp_path)
    app = create_app(cfg)
    c = TestClient(app)
    repo = app.state.repo
    from somm_core.models import PrivacyClass

    wl = repo.register_workload(
        name="<script>alert(1)</script>",
        project=cfg.project,
        privacy_class=PrivacyClass.INTERNAL,
    )
    evidence = {
        "workload": "<script>alert(1)</script>",
        "current": {"provider": "<img src=x>", "model": "</table><script>x"},
        "candidate": {"provider": "ok", "model": "ok"},
    }
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO recommendations "
            "(workload_id, action, evidence_json, expected_impact, confidence) "
            "VALUES (?, 'switch_model', ?, 'ok', 0.5)",
            (wl.id, json.dumps(evidence)),
        )

    r = c.get("/")
    assert "<script>alert(1)</script>" not in r.text
    assert "<img src=x>" not in r.text
    assert "&lt;script&gt;" in r.text

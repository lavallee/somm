"""Web admin recommendations rendering + dismiss/apply endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from somm_core.config import Config
from somm_service.app import create_app
from starlette.testclient import TestClient

_LOCAL_HEADERS = {"x-somm-local": "1", "sec-fetch-site": "same-origin"}


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "recs-test"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


@pytest.fixture
def client_with_rec(tmp_path):
    cfg = _tmp_config(tmp_path)
    app = create_app(cfg)
    # Loopback base_url so the X-Somm-Local dashboard path is honored — the
    # header path is gated on a loopback Host to defeat DNS rebinding.
    c = TestClient(app, base_url="http://localhost")

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

    blocked = c.post(f"/api/recommendations/{rec_id}/dismiss")
    assert blocked.status_code == 403
    assert "X-Somm-Local: 1" in blocked.json()["error"]

    r = c.post(f"/api/recommendations/{rec_id}/dismiss", headers=_LOCAL_HEADERS)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # No longer open
    after = c.get("/api/recommendations").json()
    assert after["recommendations"] == []


def test_apply_rec(client_with_rec):
    c, cfg, app = client_with_rec
    data = c.get("/api/recommendations").json()
    rec_id = data["recommendations"][0]["id"]

    blocked = c.post(f"/api/recommendations/{rec_id}/apply")
    assert blocked.status_code == 403
    assert "X-Somm-Local: 1" in blocked.json()["error"]

    r = c.post(f"/api/recommendations/{rec_id}/apply", headers=_LOCAL_HEADERS)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # DNS-rebinding: same local headers but a non-loopback Host must NOT
    # ride the header path (a rebound page's Host is attacker.example).
    rebind = TestClient(app, base_url="http://attacker.example")
    blocked_rebind = rebind.post(
        f"/api/recommendations/{rec_id}/apply", headers=_LOCAL_HEADERS
    )
    assert blocked_rebind.status_code == 403

    # Verify applied_at set and that apply wrote the workload policy revision
    # plus a decision row; this is not a mere "mark read" endpoint.
    repo = app.state.repo
    with repo._open() as conn:
        row = conn.execute(
            "SELECT applied_at FROM recommendations WHERE id = ?",
            (rec_id,),
        ).fetchone()
    assert row[0] is not None
    refreshed = repo.workload_by_name("demo_w", cfg.project)
    assert refreshed.policy["fallback"][0] == {"provider": "ollama", "model": "fast"}
    decisions = repo.search_decisions(workload="demo_w")
    assert len(decisions) == 1
    assert decisions[0].chosen_provider == "ollama"
    assert decisions[0].chosen_model == "fast"


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

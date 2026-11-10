"""Data boundary at the proxy: /v1/messages and /v1/chat/completions refuse a
target outside the workload's boundary with 403 SOMM_PRIVACY_VIOLATION,
record a refusal row (written by the client), and never call a provider."""

from __future__ import annotations

from pathlib import Path

from somm_core.config import Config
from somm_core.models import PrivacyClass
from somm_core.repository import Repository
from somm_service.app import create_app
from starlette.testclient import TestClient


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "proxy-boundary"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.ollama_url = "http://127.0.0.1:11434"
    return cfg


def _headers(app, **extra):
    return {"authorization": f"Bearer {app.state.service_token.value}", **extra}


_ROWS = "SELECT provider, model, outcome, error_kind, data_tier FROM calls ORDER BY ts"


def test_private_workload_messages_refused_without_calling_a_provider(
    tmp_path, fake_provider, ledger
):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    fake = fake_provider("anthropic")
    app = create_app(cfg, providers=[fake])
    client = TestClient(app)
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
    r = client.post("/v1/messages", json=body, headers=_headers(app, **{"x-somm-workload": "secret"}))
    streamed = client.post("/v1/messages", json={**body, "stream": True},
                           headers=_headers(app, **{"x-somm-workload": "secret"}))
    assert r.status_code == 403 and streamed.status_code == 403
    assert r.json()["error"]["type"] == "permission_error"
    assert r.json()["error"]["message"].startswith("SOMM_PRIVACY_VIOLATION")
    assert r.headers["x-should-retry"] == "false"
    assert fake.requests == []
    assert ledger(app, _ROWS) == [("", "claude-x", "unknown", "SommPrivacyViolation", None)] * 2


def test_chat_completions_refused_with_code_and_no_retry(tmp_path, fake_provider):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    fake = fake_provider("openai")
    app = create_app(cfg, providers=[fake])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "gpt-x", "messages": [{"role": "user", "content": "x"}]},
        headers=_headers(app, **{"x-somm-workload": "secret"}),
    )
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["code"] == "SOMM_PRIVACY_VIOLATION" and err["retryable"] is False
    assert err["data_boundary"] == "local" and err["refused"][0]["provider"] == "openai"
    assert r.headers["x-should-retry"] == "false"
    assert fake.requests == []


def test_local_target_is_admitted_and_tier_recorded(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    app = create_app(cfg, providers=[fake_provider("ollama", base_url=cfg.ollama_url)])
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "ollama/qwen3:8b", "messages": [{"role": "user", "content": "x"}]},
        headers=_headers(app, **{"x-somm-workload": "secret"}),
    )
    assert r.status_code == 200
    assert ledger(app, _ROWS)[-1][-1] == "local"


def test_default_boundary_leaves_the_proxy_unchanged(tmp_path, fake_provider, ledger):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="open", project=cfg.project)
    fake = fake_provider("anthropic")
    app = create_app(cfg, providers=[fake])
    msg = {"model": "anthropic/claude-x", "messages": [{"role": "user", "content": "x"}]}
    ok = TestClient(app).post("/v1/chat/completions", json=msg,
                              headers=_headers(app, **{"x-somm-workload": "open"}))
    assert ok.status_code == 200
    assert len(fake.requests) == 1
    assert ledger(app, _ROWS)[0][-1] == "subscribed"

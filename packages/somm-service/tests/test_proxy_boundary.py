"""Data boundary at the proxy: /v1/messages and /v1/chat/completions refuse a
target outside the workload's boundary with 403 SOMM_PRIVACY_VIOLATION,
record a refusal row, and never invoke litellm."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


def _anthropic_resp():
    return SimpleNamespace(
        id="msg_1", model="claude-x",
        choices=[SimpleNamespace(message={"role": "assistant", "content": "hi"},
                                 finish_reason="stop")],
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _rows(repo):
    with repo._open() as conn:
        return conn.execute(
            "SELECT provider, model, outcome, error_kind, data_tier FROM calls ORDER BY ts"
        ).fetchall()


def test_private_workload_messages_refused_without_calling_litellm(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    app = create_app(cfg)
    client = TestClient(app)
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
    with patch("somm_service.proxy.litellm.completion") as completion:
        r = client.post("/v1/messages", json=body,
                        headers=_headers(app, **{"x-somm-workload": "secret"}))
        streamed = client.post("/v1/messages", json={**body, "stream": True},
                               headers=_headers(app, **{"x-somm-workload": "secret"}))
    assert r.status_code == 403 and streamed.status_code == 403
    assert r.json()["error"]["type"] == "permission_error"
    assert r.json()["error"]["message"].startswith("SOMM_PRIVACY_VIOLATION")
    completion.assert_not_called()
    assert _rows(repo) == [("", "claude-x", "unknown", "SommPrivacyViolation", None)] * 2


def test_chat_completions_refused_with_code_and_no_retry(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    app = create_app(cfg)
    client = TestClient(app)
    with patch("somm_service.proxy.litellm.completion") as completion:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-x", "messages": [{"role": "user", "content": "x"}]},
            headers=_headers(app, **{"x-somm-workload": "secret"}),
        )
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["code"] == "SOMM_PRIVACY_VIOLATION" and err["retryable"] is False
    assert err["data_boundary"] == "local" and err["refused"][0]["provider"] == "openai"
    assert r.headers["x-should-retry"] == "false"
    completion.assert_not_called()


def test_local_target_is_admitted_and_tier_recorded(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="secret", project=cfg.project, privacy_class=PrivacyClass.PRIVATE)
    app = create_app(cfg)
    client = TestClient(app)
    with patch("somm_service.proxy.litellm.completion", return_value=_anthropic_resp()):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "ollama/qwen3:8b", "messages": [{"role": "user", "content": "x"}]},
            headers=_headers(app, **{"x-somm-workload": "secret"}),
        )
    assert r.status_code == 200
    assert _rows(repo)[-1][-1] == "local"


def test_default_boundary_leaves_the_proxy_unchanged(tmp_path):
    cfg = _cfg(tmp_path)
    repo = Repository(cfg.db_path)
    repo.register_workload(name="open", project=cfg.project)
    app = create_app(cfg)
    client = TestClient(app)
    msg = {"model": "anthropic/claude-x", "messages": [{"role": "user", "content": "x"}]}
    with patch("somm_service.proxy.litellm.completion", return_value=_anthropic_resp()) as c:
        ok = client.post("/v1/chat/completions", json=msg,
                         headers=_headers(app, **{"x-somm-workload": "open"}))
    assert ok.status_code == 200
    assert c.call_count == 1
    assert _rows(repo)[0][-1] == "subscribed"

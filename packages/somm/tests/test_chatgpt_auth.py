import os
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from somm.chatgpt_auth import PERMISSION, ChatGPTAuth, ChatGPTAuthError, private_json, read_json


def test_authorization_has_pkce_state_nonce_and_stable_host(tmp_path):
    auth = ChatGPTAuth(tmp_path / "auth.json")
    first, pending = auth.begin("http://127.0.0.1:1456/auth/callback")
    second, other = auth.begin("http://127.0.0.1:1456/auth/callback")
    query = parse_qs(urlsplit(first).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["agent_name_hint"] == ["Somm"]
    assert query["code_challenge_method"] == ["S256"]
    assert pending["state"] != other["state"] and pending["nonce"] != other["nonce"]
    assert pending["host_id"] == other["host_id"]
    assert second != first
    assert not auth.path.exists()
    with pytest.raises(ChatGPTAuthError):
        auth.begin("https://example.com/auth/callback")


def test_invalid_callback_does_not_exchange_or_overwrite(tmp_path, monkeypatch):
    auth = ChatGPTAuth(tmp_path / "auth.json")
    private_json(auth.path, {"previous": True})
    _, pending = auth.begin("http://127.0.0.1:1456/auth/callback")
    monkeypatch.setattr(auth, "_token_request", lambda _: pytest.fail("must not exchange"))
    with pytest.raises(ChatGPTAuthError, match="state"):
        auth.finish(pending, {"state": "wrong", "code": "secret"})
    assert read_json(auth.path) == {"previous": True}


def record():
    return {"client_id": "issued", "subject": "synthetic", "ext_agent_host_id": "urn:uuid:test",
            "access_token": "old", "refresh_token": "old-refresh", "scopes": [PERMISSION],
            "expires_at": time.time() - 1}


def test_rotating_refresh_is_atomic_private_and_preserves_host(tmp_path, monkeypatch):
    auth = ChatGPTAuth(tmp_path / "auth.json")
    private_json(auth.path, record())
    private_json(auth.host_file, {"ext_agent_host_id": "urn:uuid:vm"})
    def refresh(data):
        assert data["refresh_token"] == "old-refresh"
        assert data["ext_agent_host_id"] == "urn:uuid:vm"
        return {"access_token": "new", "refresh_token": "new-refresh", "expires_in": 3600}
    monkeypatch.setattr(auth, "_token_request", refresh)
    assert auth.access_token() == "new"
    assert read_json(auth.path)["refresh_token"] == "new-refresh"
    assert os.stat(auth.path).st_mode & 0o777 == 0o600
    assert auth.access_token() == "new"


def test_failed_refresh_preserves_existing_record(tmp_path, monkeypatch):
    auth = ChatGPTAuth(tmp_path / "auth.json")
    private_json(auth.path, record())
    original = auth.path.read_bytes()
    monkeypatch.setattr(auth, "_token_request", lambda _: {"access_token": "new", "expires_in": 3600})
    with pytest.raises(ChatGPTAuthError, match="rotating"):
        auth.access_token()
    assert auth.path.read_bytes() == original


def test_identity_mismatch_cannot_activate_account(tmp_path, monkeypatch):
    auth = ChatGPTAuth(tmp_path / "auth.json")
    private_json(auth.path, record())
    _, pending = auth.begin("http://127.0.0.1:1456/auth/callback")
    monkeypatch.setattr(auth, "_token_request", lambda _: {"id_token": "fake"})
    monkeypatch.setattr("somm.chatgpt_auth.validate_identity", lambda *a, **kw: {"sub": "someone-else"})
    with pytest.raises(ChatGPTAuthError, match="identity"):
        auth.finish(pending, {"state": pending["state"], "client_id": "issued", "code": "fake"})
    assert read_json(auth.path)["access_token"] == "old"


def test_returning_authorization_url_does_not_expose_retained_id_token(tmp_path):
    auth = ChatGPTAuth(tmp_path / 'auth.json')
    private_json(auth.path, {**record(), 'id_token':'sensitive-id-token'})
    url, _ = auth.begin('http://127.0.0.1:1456/auth/callback')
    assert 'sensitive-id-token' not in url and 'id_token_hint' not in url


def test_refresh_schedule_is_respected_for_iso_timestamp(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    auth = ChatGPTAuth(tmp_path / 'auth.json')
    value = ChatGPTAuth._record({'access_token':'new', 'refresh_token':'rotated', 'expires_in':3600,
        'earliest_refresh_at':datetime.fromtimestamp(time.time()+600, UTC).isoformat()}, record())
    value['expires_at'] = time.time()-1
    private_json(auth.path, value)
    monkeypatch.setattr(auth, '_token_request', lambda _: pytest.fail('must not refresh early'))
    with pytest.raises(ChatGPTAuthError, match='not yet allowed'):
        auth.access_token()

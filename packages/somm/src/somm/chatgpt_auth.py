"""Somm-owned Sign in with ChatGPT credentials for the public Responses API.

Never reads another application's credential file. Tokens are private, atomic,
and refreshed under a process lock so rotating refresh tokens cannot race.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import secrets
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

import httpx

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PERMISSION = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PERMISSION}"


class ChatGPTAuthError(RuntimeError):
    pass


def private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".somm-auth-", dir=path.parent)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as exc:
        raise ChatGPTAuthError("Cannot read Somm's ChatGPT credential record") from exc


def validate_identity(token: str, client_id: str, *, nonce: str | None = None) -> dict:
    import jwt

    discovery = httpx.get(ISSUER + "/.well-known/openid-configuration", timeout=20)
    discovery.raise_for_status()
    jwks = discovery.json().get("jwks_uri", "")
    if not jwks.startswith(ISSUER + "/"):
        raise ChatGPTAuthError("Unexpected OpenAI signing-key URL")
    try:
        key = jwt.PyJWKClient(jwks).get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256", "ES256"], audience=client_id,
                            issuer=ISSUER, options={"require": ["exp", "sub", "iss", "aud"]})
        if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise ChatGPTAuthError("ChatGPT sign-in nonce did not match")
        return claims
    except jwt.PyJWTError as exc:
        raise ChatGPTAuthError("ChatGPT identity validation failed") from exc


class ChatGPTAuth:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.environ.get("SOMM_CHATGPT_AUTH_FILE")
                         or Path.home() / ".config/somm/chatgpt.json")
        self.registration = self.path.with_suffix(".registration.json")
        self.host_file = self.path.parent / "chatgpt-host.json"

    def _lock(self):
        from filelock import FileLock

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return FileLock(str(self.path) + ".lock", timeout=30, mode=0o600)

    def _host_id(self) -> str:
        # Call only while holding this connection's lock. A copied credential
        # record must not overwrite the destination host's installation ID.
        host = read_json(self.host_file)
        if not host.get("ext_agent_host_id"):
            host = {"ext_agent_host_id": "urn:uuid:" + str(uuid4())}
            private_json(self.host_file, host)
        return host["ext_agent_host_id"]

    def begin(self, redirect_uri: str) -> tuple[str, dict]:
        redirect = urlsplit(redirect_uri)
        if (redirect.scheme != "http" or redirect.hostname != "127.0.0.1"
                or redirect.path != "/auth/callback" or redirect.query or redirect.fragment
                or redirect.username or redirect.password):
            raise ChatGPTAuthError("ChatGPT sign-in requires an HTTP loopback /auth/callback")
        with self._lock():
            saved = read_json(self.path)
            host = {"ext_agent_host_id": self._host_id()}
            registration = read_json(self.registration)
            client_id = saved.get("client_id") or registration.get("client_id") or "dynamic_agent_client"
            pending = {"client_id": client_id, "state": secrets.token_urlsafe(32),
                       "nonce": secrets.token_urlsafe(32), "verifier": secrets.token_urlsafe(64),
                       "redirect_uri": redirect_uri, "host_id": host["ext_agent_host_id"],
                       "subject": saved.get("subject")}
            challenge = base64.urlsafe_b64encode(hashlib.sha256(pending["verifier"].encode()).digest()).decode().rstrip("=")
            query = {"client_id": client_id, "ext_agent_host_id": pending["host_id"],
                     "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES,
                     "resource": RESOURCE, "state": pending["state"], "nonce": pending["nonce"],
                     "code_challenge_method": "S256", "code_challenge": challenge}
            if client_id == "dynamic_agent_client":
                query["agent_name_hint"] = "Somm"
            return ISSUER + "/api/accounts/authorize?" + urlencode(query), pending

    def finish(self, pending: dict, callback: dict) -> dict:
        if not secrets.compare_digest(str(callback.get("state", "")), pending["state"]):
            raise ChatGPTAuthError("ChatGPT sign-in state did not match")
        if callback.get("error"):
            raise ChatGPTAuthError("ChatGPT sign-in was declined or failed")
        client_id = callback.get("client_id") or pending["client_id"]
        if client_id == "dynamic_agent_client" or not callback.get("code"):
            raise ChatGPTAuthError("ChatGPT registration did not return an issued client ID and code")
        if pending["client_id"] != "dynamic_agent_client" and client_id != pending["client_id"]:
            raise ChatGPTAuthError("ChatGPT sign-in returned a different client registration")
        with self._lock():
            # Keep the issued registration if the single-use code expires, without
            # marking the new account active or destroying existing credentials.
            private_json(self.registration, {"client_id": client_id})
            tokens = self._token_request({"grant_type": "authorization_code", "client_id": client_id,
                                          "code": callback["code"], "code_verifier": pending["verifier"],
                                          "redirect_uri": pending["redirect_uri"], "resource": RESOURCE})
            claims = validate_identity(tokens.get("id_token", ""), client_id, nonce=pending["nonce"])
            if pending.get("subject") and claims["sub"] != pending["subject"]:
                raise ChatGPTAuthError("Sign-in identity differs from the selected Somm account")
            record = self._record(tokens, {"client_id": client_id, "subject": claims["sub"],
                                           "email": claims.get("email"), "issuer": ISSUER,
                                           "ext_agent_host_id": pending["host_id"]})
            private_json(self.path, record)
            return {"connected": PERMISSION in record["scopes"], "expires_at": record["expires_at"]}

    @staticmethod
    def _token_request(data: dict) -> dict:
        try:
            response = httpx.post(ISSUER + "/api/accounts/oauth/token", data=data, timeout=30)
        except httpx.HTTPError as exc:
            raise ChatGPTAuthError("ChatGPT token exchange could not connect") from exc
        if not response.is_success:
            raise ChatGPTAuthError("ChatGPT token exchange failed; sign in to Somm again")
        value = response.json()
        if not isinstance(value, dict):
            raise ChatGPTAuthError("Invalid ChatGPT token response")
        return value

    @staticmethod
    def _record(tokens: dict, previous: dict) -> dict:
        scopes = str(tokens.get("scope", " ".join(previous.get("scopes", [])))).split()
        expiry = tokens.get("expires_in")
        if (not isinstance(expiry, (int, float)) or isinstance(expiry, bool)
                or not 0 < expiry <= 86400 or not isinstance(tokens.get("access_token"), str)
                or not tokens["access_token"] or tokens.get("token_type", "Bearer").lower() != "bearer"):
            raise ChatGPTAuthError("Invalid ChatGPT token expiry or missing access token")
        refresh = tokens.get("refresh_token")
        if not isinstance(refresh, str) or not refresh:
            raise ChatGPTAuthError("ChatGPT did not return a rotating refresh token")
        earliest = tokens.get("earliest_refresh_at")
        if isinstance(earliest, str):
            try:
                parsed = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("missing timezone")
                earliest = parsed.timestamp()
            except ValueError as exc:
                raise ChatGPTAuthError("Invalid ChatGPT refresh schedule") from exc
        if earliest is not None and (isinstance(earliest, bool) or not isinstance(earliest, (int, float)) or not math.isfinite(earliest)):
            raise ChatGPTAuthError("Invalid ChatGPT refresh schedule")
        return {**previous, "access_token": tokens["access_token"], "refresh_token": refresh,
                "id_token": tokens.get("id_token") or previous.get("id_token"),
                "scopes": scopes, "expires_at": time.time() + expiry,
                "earliest_refresh_at": earliest, "saved_at": time.time()}

    def access_token(self) -> str:
        with self._lock():
            record = read_json(self.path)
            if PERMISSION not in record.get("scopes", []) or not record.get("access_token"):
                raise ChatGPTAuthError("Run somm chatgpt login and enable ChatGPT plan usage")
            now = time.time()
            if record.get("expires_at", 0) > now + 60:
                return record["access_token"]
            earliest = record.get("earliest_refresh_at")
            if isinstance(earliest, (int, float)) and earliest > now:
                if record.get("expires_at", 0) > now:
                    return record["access_token"]
                raise ChatGPTAuthError("ChatGPT token is expired and refresh is not yet allowed")
            if any(not record.get(k) for k in ("client_id", "refresh_token", "subject", "ext_agent_host_id")):
                raise ChatGPTAuthError("Incomplete ChatGPT connection; run somm chatgpt login")
            tokens = self._token_request({"grant_type": "refresh_token", "client_id": record["client_id"],
                                          "refresh_token": record["refresh_token"], "resource": RESOURCE,
                                          "ext_agent_host_id": self._host_id()})
            if tokens.get("id_token"):
                claims = validate_identity(tokens["id_token"], record["client_id"])
                if claims["sub"] != record["subject"]:
                    raise ChatGPTAuthError("ChatGPT refresh changed the selected account")
            updated = self._record(tokens, record)
            private_json(self.path, updated)
            if PERMISSION not in updated["scopes"]:
                raise ChatGPTAuthError("ChatGPT plan permission was revoked")
            return updated["access_token"]

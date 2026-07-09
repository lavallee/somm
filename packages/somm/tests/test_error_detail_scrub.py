"""Tests for persisted error_detail secret redaction."""

from __future__ import annotations

from somm.client import _format_error_detail


class _Response:
    def __init__(self, text: str, status_code: int = 401):
        self.text = text
        self.status_code = status_code


class _HTTPError(Exception):
    def __init__(self, text: str):
        super().__init__("upstream rejected request")
        self.response = _Response(text)


def _detail_for(body: str) -> str:
    return _format_error_detail(_HTTPError(body), "fake", "fake-model")


def _body_part(detail: str) -> str:
    return detail.split("body=", 1)[1].split(" | provider=", 1)[0]


def test_error_detail_redacts_known_secret_shapes():
    secrets = [
        "sk-12345678",
        "sk-ant-1234567890abcdef",
        "sk-proj-1234567890abcdef",
        "AKIA1234567890ABCDEF",
        "ghp_1234567890abcdefghij",
        "github_pat_1234567890_abcdefghijklmnop",
        "xoxb-1234567890-abcd",
        "AIza1234567890abcdef1234567890ABCD",
        "api_key=1234567890abcdef",
        "authorization: 1234567890abcdef",
        "Bearer 1234567890abcdef",
        "token '1234567890abcdef'",
    ]

    for secret in secrets:
        detail = _detail_for(f'{{"error":"bad auth","credential":"{secret}"}}')
        assert secret not in detail
        assert "[redacted]" in detail


def test_error_detail_leaves_normal_text_unchanged():
    body = (
        '{"error":"model not found","url":"https://api.example.test/v1/models",'
        '"message":"try another model"}'
    )

    detail = _detail_for(body)

    assert _body_part(detail) == body
    assert "[redacted]" not in detail


def test_error_detail_scrubs_before_body_truncation():
    secret = "sk-" + ("A" * 220)
    body = ("x" * 195) + secret

    detail = _detail_for(body)
    body_detail = _body_part(detail)

    assert len(body_detail) == 200
    assert "sk-" not in body_detail
    assert body_detail.endswith("[reda")

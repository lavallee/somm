"""Unit tests for somm_core.config.load() env-var override behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from somm_core.config import load

_SOMM_ENV_VARS = [
    "SOMM_PROJECT",
    "SOMM_MODE",
    "SOMM_OLLAMA_URL",
    "SOMM_OLLAMA_MODEL",
    "SOMM_OLLAMA_THINK",
    "SOMM_OLLAMA_KEEP_ALIVE",
    "SOMM_OPENROUTER_ROSTER",
    "OPENROUTER_API_KEY",
    "SOMM_CROSS_PROJECT",
    "SOMM_BUDGET_FAIL_CLOSED",
    "SOMM_BUDGET_DEFAULT_CAP_USD_DAILY",
    "SOMM_INPROCESS_WORKERS",
    "SOMM_GLOBAL_PATH",
    "SOMM_PROVIDER_ORDER",
    "ANTHROPIC_API_KEY",
    "SOMM_ANTHROPIC_MODEL",
    "OPENAI_API_KEY",
    "SOMM_OPENAI_MODEL",
    "SOMM_OPENAI_BASE_URL",
    "MINIMAX_API_KEY",
    "SOMM_MINIMAX_MODEL",
    "DEEPSEEK_API_KEY",
    "SOMM_DEEPSEEK_MODEL",
    "GEMINI_API_KEY",
    "SOMM_GEMINI_MODEL",
    "PERPLEXITY_API_KEY",
    "SOMM_PERPLEXITY_MODEL",
    "SOMM_HTTP_TIMEOUT",
]


def _clear_somm_env(monkeypatch) -> None:
    for var in _SOMM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_load_defaults_no_env(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    cfg = load(cwd=tmp_path)
    assert cfg.project == "default"
    assert cfg.budget_fail_closed is False
    assert cfg.provider_order is None


def test_load_budget_fail_closed_env(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("SOMM_BUDGET_FAIL_CLOSED", "1")
    cfg = load(cwd=tmp_path)
    assert cfg.budget_fail_closed is True


def test_load_provider_order_env(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("SOMM_PROVIDER_ORDER", "openrouter,ollama")
    cfg = load(cwd=tmp_path)
    assert cfg.provider_order == ["openrouter", "ollama"]


def test_load_budget_default_cap_env(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("SOMM_BUDGET_DEFAULT_CAP_USD_DAILY", "25.0")
    cfg = load(cwd=tmp_path)
    assert cfg.budget_default_cap_usd_daily == 25.0

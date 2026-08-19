"""Unit tests for somm_core.config.load() env-var override behavior."""

from __future__ import annotations

from pathlib import Path

from somm_core.config import load

_SOMM_ENV_VARS = [
    "SOMM_PROJECT",
    "SOMM_MODE",
    "SOMM_OLLAMA_URL",
    "SOMM_OLLAMA_MODEL",
    "SOMM_DB_DIR",
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
    "SOMM_REGISTRY_PATH",
    "SOMM_REGISTRY_ALLOW_TMP",
    "SOMM_PINNED_FALLBACK",
]


def _clear_somm_env(monkeypatch) -> None:
    for var in _SOMM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_load_defaults_no_env(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    cfg = load(cwd=tmp_path)
    assert cfg.project == "default"
    assert cfg.budget_fail_closed is False
    assert cfg.pinned_fallback is False  # pins are sticky unless opted out
    assert cfg.provider_order is None
    assert cfg.minimax_model == "MiniMax-M3"
    assert cfg.ollama_model == "qwen3:8b"


def test_load_budget_fail_closed_env(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("SOMM_BUDGET_FAIL_CLOSED", "1")
    cfg = load(cwd=tmp_path)
    assert cfg.budget_fail_closed is True


def test_load_pinned_fallback_env(tmp_path: Path, monkeypatch) -> None:
    """The opt-out for fleets that relied on pre-0.16 pinned-call rescue."""
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("SOMM_PINNED_FALLBACK", "1")
    assert load(cwd=tmp_path).pinned_fallback is True
    monkeypatch.setenv("SOMM_PINNED_FALLBACK", "0")
    assert load(cwd=tmp_path).pinned_fallback is False


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


def test_load_db_dir_env_wins(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    explicit = tmp_path / "explicit-somm"
    local = tmp_path / ".somm"
    local.mkdir()
    monkeypatch.setenv("SOMM_DB_DIR", str(explicit))
    cfg = load(cwd=tmp_path)
    assert cfg.db_dir == explicit.resolve()


def test_load_db_dir_pyproject_wins(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "pyproject.toml").write_text(
        "[tool.somm]\nproject = 'py-proj'\ndb_dir = 'configured-somm'\n",
        encoding="utf-8",
    )
    (tmp_path / ".somm").mkdir()
    cfg = load(cwd=tmp_path)
    assert cfg.project == "py-proj"
    assert cfg.db_dir == (tmp_path / "configured-somm").resolve()


def test_load_existing_cwd_somm_wins_over_registry(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SOMM_PROJECT", "reuse")
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("SOMM_REGISTRY_ALLOW_TMP", "1")
    from somm_core.registry import register_project
    from somm_core.repository import Repository

    registered_db = tmp_path / "registered" / "calls.sqlite"
    Repository(registered_db)
    register_project("reuse", registered_db)
    local = tmp_path / ".somm"
    local.mkdir()

    cfg = load(cwd=tmp_path)
    assert cfg.db_dir == local.resolve()


def test_load_existing_project_root_somm_from_subdir(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "pyproject.toml").write_text("[tool.somm]\nproject = 'root-proj'\n")
    root_somm = tmp_path / ".somm"
    root_somm.mkdir()
    subdir = tmp_path / "src" / "pkg"
    subdir.mkdir(parents=True)

    cfg = load(cwd=subdir)
    assert cfg.project == "root-proj"
    assert cfg.db_dir == root_somm.resolve()


def test_load_reuses_registered_project_db(tmp_path: Path, monkeypatch, capsys) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SOMM_PROJECT", "registered-proj")
    monkeypatch.setenv("SOMM_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("SOMM_REGISTRY_ALLOW_TMP", "1")
    from somm_core.registry import register_project
    from somm_core.repository import Repository

    registered_db = tmp_path / "registered" / "calls.sqlite"
    Repository(registered_db)
    register_project("registered-proj", registered_db)
    cwd = tmp_path / "other-cwd"
    cwd.mkdir()

    cfg = load(cwd=cwd)
    err = capsys.readouterr().err
    assert cfg.db_dir == registered_db.parent.resolve()
    assert "using registered DB at" in err
    assert "set SOMM_DB_DIR to override" in err


def test_load_falls_back_to_local_somm(tmp_path: Path, monkeypatch) -> None:
    _clear_somm_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg = load(cwd=tmp_path)
    assert cfg.db_dir == (tmp_path / ".somm").resolve()

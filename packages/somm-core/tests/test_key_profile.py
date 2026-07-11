"""Named key profiles: <NAME>_API_KEY_<PROFILE> resolution in config.load().

The contract: with no profile set, behavior is byte-identical to before
(plain <NAME>_API_KEY). With a profile, the suffixed variable wins and the
plain one remains the fallback, so a profile can be introduced before every
provider has a provisioned key for it.
"""

from __future__ import annotations

import somm_core.config as config


def _clean_env(monkeypatch):
    for var in list(__import__("os").environ):
        if var.endswith("_API_KEY") or var.startswith("SOMM_"):
            monkeypatch.delenv(var, raising=False)


def test_no_profile_reads_plain_key(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "plain")
    cfg = config.load(cwd=tmp_path)
    assert cfg.key_profile is None
    assert cfg.deepseek_api_key == "plain"


def test_profile_prefers_suffixed_key(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SOMM_KEY_PROFILE", "research")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "plain")
    monkeypatch.setenv("DEEPSEEK_API_KEY_RESEARCH", "suffixed")
    cfg = config.load(cwd=tmp_path)
    assert cfg.key_profile == "research"
    assert cfg.deepseek_api_key == "suffixed"


def test_profile_falls_back_to_plain_key(tmp_path, monkeypatch):
    """A profile without a provisioned key for some provider must not
    break that provider — gradual migration depends on this."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("SOMM_KEY_PROFILE", "research")
    monkeypatch.setenv("OPENAI_API_KEY", "plain-openai")
    cfg = config.load(cwd=tmp_path)
    assert cfg.openai_api_key == "plain-openai"


def test_profile_applies_to_openrouter(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SOMM_KEY_PROFILE", "staging")
    monkeypatch.setenv("OPENROUTER_API_KEY", "plain")
    monkeypatch.setenv("OPENROUTER_API_KEY_STAGING", "staged")
    cfg = config.load(cwd=tmp_path)
    assert cfg.openrouter_api_key == "staged"


def test_profile_from_pyproject(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.somm]\nkey_profile = "research"\n'
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY_RESEARCH", "from-pyproject-profile")
    cfg = config.load(cwd=tmp_path)
    assert cfg.key_profile == "research"
    assert cfg.deepseek_api_key == "from-pyproject-profile"


def test_env_profile_overrides_pyproject(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.somm]\nkey_profile = "research"\n'
    )
    monkeypatch.setenv("SOMM_KEY_PROFILE", "staging")
    monkeypatch.setenv("DEEPSEEK_API_KEY_RESEARCH", "r")
    monkeypatch.setenv("DEEPSEEK_API_KEY_STAGING", "s")
    cfg = config.load(cwd=tmp_path)
    assert cfg.deepseek_api_key == "s"


def test_profile_name_sanitized(tmp_path, monkeypatch):
    """Hyphens and case in profile names map onto env-var-safe suffixes."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("SOMM_KEY_PROFILE", "my-team.prod")
    monkeypatch.setenv("MINIMAX_API_KEY_MY_TEAM_PROD", "matrixed")
    cfg = config.load(cwd=tmp_path)
    assert cfg.minimax_api_key == "matrixed"


def test_non_key_somm_vars_unaffected(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SOMM_KEY_PROFILE", "research")
    monkeypatch.setenv("SOMM_DEEPSEEK_MODEL", "deepseek-chat")
    cfg = config.load(cwd=tmp_path)
    assert cfg.deepseek_model == "deepseek-chat"

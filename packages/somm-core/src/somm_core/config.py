"""Config loading: env > pyproject.toml > runtime override > defaults.

Minimal v0.1 surface. Expands as features demand.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Config:
    project: str = "default"
    mode: str = "observe"  # "observe" | "strict"
    db_dir: Path = field(default_factory=lambda: Path("./.somm"))
    spool_dir: Path = field(default_factory=lambda: Path("./.somm/spool"))
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"
    openrouter_api_key: str | None = None
    openrouter_roster: list[str] | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    minimax_api_key: str | None = None
    minimax_model: str = "MiniMax-M2"
    busy_timeout_ms: int = 5000

    @property
    def db_path(self) -> Path:
        return self.db_dir / "calls.sqlite"


def load(project: str | None = None, cwd: Path | None = None) -> Config:
    """Load config from env + ./pyproject.toml [tool.somm] + defaults.

    Order: defaults < pyproject [tool.somm] < env (SOMM_*) < explicit args.
    """
    cwd = cwd or Path.cwd()
    cfg = Config()

    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        somm_cfg = data.get("tool", {}).get("somm", {})
        for key in ("project", "mode", "ollama_url", "ollama_model"):
            if key in somm_cfg:
                setattr(cfg, key, somm_cfg[key])

    env_map = {
        "SOMM_PROJECT": "project",
        "SOMM_MODE": "mode",
        "SOMM_OLLAMA_URL": "ollama_url",
        "SOMM_OLLAMA_MODEL": "ollama_model",
    }
    for env_var, attr in env_map.items():
        if env_var in os.environ:
            setattr(cfg, attr, os.environ[env_var])
    if "OPENROUTER_API_KEY" in os.environ:
        cfg.openrouter_api_key = os.environ["OPENROUTER_API_KEY"]
    if "SOMM_OPENROUTER_ROSTER" in os.environ:
        cfg.openrouter_roster = [
            m.strip() for m in os.environ["SOMM_OPENROUTER_ROSTER"].split(",") if m.strip()
        ]
    for env_var, attr in (
        ("ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("SOMM_ANTHROPIC_MODEL", "anthropic_model"),
        ("OPENAI_API_KEY", "openai_api_key"),
        ("SOMM_OPENAI_MODEL", "openai_model"),
        ("SOMM_OPENAI_BASE_URL", "openai_base_url"),
        ("MINIMAX_API_KEY", "minimax_api_key"),
        ("SOMM_MINIMAX_MODEL", "minimax_model"),
    ):
        if env_var in os.environ:
            setattr(cfg, attr, os.environ[env_var])

    if project is not None:
        cfg.project = project

    # Resolve paths relative to cwd; caller may also pass explicit paths.
    cfg.db_dir = (
        Path(cfg.db_dir).resolve() if cfg.db_dir.is_absolute() else (cwd / cfg.db_dir).resolve()
    )
    cfg.spool_dir = cfg.db_dir / "spool"

    return cfg

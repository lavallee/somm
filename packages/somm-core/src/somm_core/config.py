"""Config loading: env > pyproject.toml > runtime override > defaults.

Minimal v0.1 surface. Expands as features demand.

SOMM_WAIT_ON_EXHAUSTED can set a process default wait deadline, in seconds,
for calls whose routed provider set is entirely cooling down. Per-call
``wait`` arguments on ``generate()``, ``stream()``, and
``extract_structured()`` take precedence; ``wait=None`` and ``wait=0`` mean
fail fast.

SOMM_PINNED_FALLBACK=1 restores the pre-0.16 default in which a pinned
``provider=``/``model=`` call that failed was silently rescued by the router
chain — possibly on a different model. The default is off: a pin is honored
or the call fails with the pinned attribution intact. Per-call
``allow_fallback=`` takes precedence.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from somm_core.registry import registered_db_path


@dataclass(slots=True)
class Config:
    project: str = "default"
    mode: str = "observe"  # "observe" | "strict"
    # Named key profile (AWS_PROFILE-style): when set, provider keys
    # resolve from <NAME>_API_KEY_<PROFILE> before <NAME>_API_KEY, so one
    # environment can hold several provisioned keys per provider (per
    # product, team, or environment) and provider dashboards attribute
    # spend per profile. Unset → exactly the old single-key behavior.
    key_profile: str | None = None
    db_dir: Path = field(default_factory=lambda: Path("./.somm"))
    spool_dir: Path = field(default_factory=lambda: Path("./.somm/spool"))
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_think: bool = False  # Set "think": true on ollama requests (reasoning models)
    ollama_keep_alive: str = "30m"  # Model residency window; "0" to opt out
    openrouter_api_key: str | None = None
    openrouter_roster: list[str] | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    minimax_api_key: str | None = None
    minimax_model: str = "MiniMax-M3"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-pro"
    perplexity_api_key: str | None = None
    perplexity_model: str = "sonar"  # search-grounded; only used when pinned
    http_timeout: float = 180.0  # seconds; used by openai-compat providers
    provider_order: list[str] | None = None  # e.g. ["minimax", "openrouter", "ollama"]
    busy_timeout_ms: int = 5000
    # Local-only replication of calls/workloads into ~/.somm/global.sqlite so
    # cross-project features (sommelier, fleet rollups) have data. The raw
    # dataclass default is False (programmatic Config() stays inert, e.g. in
    # tests); load() enables it by default since v0.8 — nothing leaves the
    # machine; SOMM_CROSS_PROJECT=0 disables.
    cross_project_enabled: bool = False
    cross_project_path: Path | None = None  # defaults to ~/.somm/global.sqlite
    budget_fail_closed: bool = False  # hard pre-request gate: block calls once a workload's daily cap is reached
    budget_default_cap_usd_daily: float | None = None  # daily cap for workloads with no explicit budget_cap_usd_daily (when fail_closed)
    inprocess_workers: bool = False  # run the somm-service scheduler inside the library process (SOMM_INPROCESS_WORKERS=1)
    wait_on_exhausted: float | None = None  # default generate/stream wait deadline when all routed providers are cooling
    # Pinned calls are sticky by default (since v0.16): naming a provider
    # and/or model means "serve it with this or fail", never "try this first
    # and silently substitute something else". Set SOMM_PINNED_FALLBACK=1 to
    # restore the pre-0.16 rescue-through-the-router-chain behavior process-wide.
    pinned_fallback: bool = False
    service_public_read: bool = False  # allow unauthenticated dashboard/read API only when explicitly opted in
    service_proxy_max_body_bytes: int = 1_048_576
    service_otlp_max_body_bytes: int = 1_048_576
    service_otlp_max_spans: int = 1000
    service_otlp_max_attr_chars: int = 2048
    mcp_compare_max_models: int = 5
    mcp_compare_allow_expensive_max_models: int = 10
    mcp_compare_max_tokens: int = 1024
    mcp_compare_allow_expensive_max_tokens: int = 4096

    @property
    def db_path(self) -> Path:
        return self.db_dir / "calls.sqlite"

    @property
    def global_db_path(self) -> Path:
        if self.cross_project_path is not None:
            return Path(self.cross_project_path)
        return Path.home() / ".somm" / "global.sqlite"


def load(project: str | None = None, cwd: Path | None = None) -> Config:
    """Load config from env + ./pyproject.toml [tool.somm] + defaults.

    Order: defaults < pyproject [tool.somm] < env (SOMM_*) < explicit args.
    """
    cwd = (cwd or Path.cwd()).resolve()
    project_root = _discover_project_root(cwd)
    cfg = Config()
    explicit_db_dir = False
    db_dir_base = cwd

    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        somm_cfg = data.get("tool", {}).get("somm", {})
        for key in ("project", "mode", "ollama_url", "ollama_model", "key_profile"):
            if key in somm_cfg:
                setattr(cfg, key, somm_cfg[key])
        if "db_dir" in somm_cfg:
            cfg.db_dir = Path(somm_cfg["db_dir"])
            explicit_db_dir = True
            db_dir_base = project_root

    env_map = {
        "SOMM_PROJECT": "project",
        "SOMM_MODE": "mode",
        "SOMM_OLLAMA_URL": "ollama_url",
        "SOMM_OLLAMA_MODEL": "ollama_model",
        "SOMM_KEY_PROFILE": "key_profile",
    }
    for env_var, attr in env_map.items():
        if env_var in os.environ:
            setattr(cfg, attr, os.environ[env_var])
    if "SOMM_DB_DIR" in os.environ:
        cfg.db_dir = Path(os.environ["SOMM_DB_DIR"])
        explicit_db_dir = True
        db_dir_base = cwd
    def _key_env(base: str) -> str | None:
        """Resolve a provider key honouring the named key profile:
        <base>_<PROFILE> wins, plain <base> is the fallback — so setting
        a profile never breaks an environment that only has plain keys."""
        if cfg.key_profile:
            suffix = re.sub(r"[^A-Za-z0-9]+", "_", cfg.key_profile).upper().strip("_")
            if suffix:
                val = os.environ.get(f"{base}_{suffix}")
                if val:
                    return val
        return os.environ.get(base)

    _openrouter_key = _key_env("OPENROUTER_API_KEY")
    if _openrouter_key:
        cfg.openrouter_api_key = _openrouter_key
    if "SOMM_OPENROUTER_ROSTER" in os.environ:
        cfg.openrouter_roster = [
            m.strip() for m in os.environ["SOMM_OPENROUTER_ROSTER"].split(",") if m.strip()
        ]
    if "SOMM_OLLAMA_THINK" in os.environ:
        val = os.environ["SOMM_OLLAMA_THINK"].strip().lower()
        cfg.ollama_think = val in ("1", "true", "yes", "on")
    if "SOMM_OLLAMA_KEEP_ALIVE" in os.environ:
        cfg.ollama_keep_alive = os.environ["SOMM_OLLAMA_KEEP_ALIVE"].strip()
    # Default-on for env-resolved (real user) configs; see the field comment.
    cfg.cross_project_enabled = True
    if "SOMM_CROSS_PROJECT" in os.environ:
        val = os.environ["SOMM_CROSS_PROJECT"].strip().lower()
        cfg.cross_project_enabled = val in ("1", "true", "yes", "on")
    if "SOMM_BUDGET_FAIL_CLOSED" in os.environ:
        val = os.environ["SOMM_BUDGET_FAIL_CLOSED"].strip().lower()
        cfg.budget_fail_closed = val in ("1", "true", "yes", "on")
    if "SOMM_INPROCESS_WORKERS" in os.environ:
        val = os.environ["SOMM_INPROCESS_WORKERS"].strip().lower()
        cfg.inprocess_workers = val in ("1", "true", "yes", "on")
    if "SOMM_BUDGET_DEFAULT_CAP_USD_DAILY" in os.environ:
        with contextlib.suppress(ValueError):
            cfg.budget_default_cap_usd_daily = float(os.environ["SOMM_BUDGET_DEFAULT_CAP_USD_DAILY"])
    if "SOMM_PINNED_FALLBACK" in os.environ:
        val = os.environ["SOMM_PINNED_FALLBACK"].strip().lower()
        cfg.pinned_fallback = val in ("1", "true", "yes", "on")
    if "SOMM_WAIT_ON_EXHAUSTED" in os.environ:
        with contextlib.suppress(ValueError):
            cfg.wait_on_exhausted = float(os.environ["SOMM_WAIT_ON_EXHAUSTED"])
    if "SOMM_SERVICE_PUBLIC_READ" in os.environ:
        val = os.environ["SOMM_SERVICE_PUBLIC_READ"].strip().lower()
        cfg.service_public_read = val in ("1", "true", "yes", "on")
    for env_var, attr in (
        ("SOMM_SERVICE_PROXY_MAX_BODY_BYTES", "service_proxy_max_body_bytes"),
        ("SOMM_SERVICE_OTLP_MAX_BODY_BYTES", "service_otlp_max_body_bytes"),
        ("SOMM_SERVICE_OTLP_MAX_SPANS", "service_otlp_max_spans"),
        ("SOMM_SERVICE_OTLP_MAX_ATTR_CHARS", "service_otlp_max_attr_chars"),
        ("SOMM_MCP_COMPARE_MAX_MODELS", "mcp_compare_max_models"),
        ("SOMM_MCP_COMPARE_ALLOW_EXPENSIVE_MAX_MODELS", "mcp_compare_allow_expensive_max_models"),
        ("SOMM_MCP_COMPARE_MAX_TOKENS", "mcp_compare_max_tokens"),
        ("SOMM_MCP_COMPARE_ALLOW_EXPENSIVE_MAX_TOKENS", "mcp_compare_allow_expensive_max_tokens"),
    ):
        if env_var in os.environ:
            with contextlib.suppress(ValueError):
                setattr(cfg, attr, max(1, int(os.environ[env_var])))
    if "SOMM_GLOBAL_PATH" in os.environ:
        cfg.cross_project_path = Path(os.environ["SOMM_GLOBAL_PATH"])
    if "SOMM_PROVIDER_ORDER" in os.environ:
        cfg.provider_order = [
            p.strip() for p in os.environ["SOMM_PROVIDER_ORDER"].split(",") if p.strip()
        ]
    for env_var, attr in (
        ("ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("SOMM_ANTHROPIC_MODEL", "anthropic_model"),
        ("OPENAI_API_KEY", "openai_api_key"),
        ("SOMM_OPENAI_MODEL", "openai_model"),
        ("SOMM_OPENAI_BASE_URL", "openai_base_url"),
        ("MINIMAX_API_KEY", "minimax_api_key"),
        ("SOMM_MINIMAX_MODEL", "minimax_model"),
        ("DEEPSEEK_API_KEY", "deepseek_api_key"),
        ("SOMM_DEEPSEEK_MODEL", "deepseek_model"),
        ("GEMINI_API_KEY", "gemini_api_key"),
        ("SOMM_GEMINI_MODEL", "gemini_model"),
        ("PERPLEXITY_API_KEY", "perplexity_api_key"),
        ("SOMM_PERPLEXITY_MODEL", "perplexity_model"),
    ):
        if env_var.endswith("_API_KEY"):
            val = _key_env(env_var)
            if val:
                setattr(cfg, attr, val)
        elif env_var in os.environ:
            setattr(cfg, attr, os.environ[env_var])
    if "SOMM_HTTP_TIMEOUT" in os.environ:
        with contextlib.suppress(ValueError):
            cfg.http_timeout = float(os.environ["SOMM_HTTP_TIMEOUT"])

    if project is not None:
        cfg.project = project

    # Resolve paths relative to cwd; caller may also pass explicit paths.
    if explicit_db_dir:
        cfg.db_dir = (
            Path(cfg.db_dir).resolve()
            if cfg.db_dir.is_absolute()
            else (db_dir_base / cfg.db_dir).resolve()
        )
    elif (cwd / ".somm").exists():
        cfg.db_dir = (cwd / ".somm").resolve()
    elif project_root != cwd and (project_root / ".somm").exists():
        cfg.db_dir = (project_root / ".somm").resolve()
    elif (registered := registered_db_path(cfg.project)) is not None:
        cfg.db_dir = registered.parent.resolve()
        print(
            f"[somm] using registered DB at {registered.resolve()} ; set SOMM_DB_DIR to override",
            file=sys.stderr,
        )
    else:
        cfg.db_dir = (cwd / ".somm").resolve()
    cfg.spool_dir = cfg.db_dir / "spool"

    return cfg


def _discover_project_root(cwd: Path) -> Path:
    for path in (cwd, *cwd.parents):
        if (path / "pyproject.toml").exists():
            return path
    return cwd

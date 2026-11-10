"""Connect coding agents to somm: register the MCP server, link the skill.

MCP registration is delegated to each harness's own CLI (`claude mcp add`,
`codex mcp add`); when that CLI is not on PATH the exact snippet is printed
instead. The skill is a symlink into the installed ``somm_skill`` package, so
upgrading the package updates the skill; `status` flags a link that points
elsewhere or at nothing.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

NAME = "somm"
HARNESSES = ("claude", "codex")


def found_harnesses() -> list[str]:
    return [h for h in HARNESSES if shutil.which(h)]


def mcp_command() -> list[str]:
    """Absolute interpreter + module, so it resolves from any harness's PATH."""
    if importlib.util.find_spec("somm_mcp") is None:
        raise RuntimeError("somm-mcp is not installed in this environment (pip install somm-mcp)")
    return [sys.executable, "-m", "somm_mcp.cli"]


def _run(argv: list[str]) -> tuple[int, str]:
    p = subprocess.run(argv, capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


# ---- MCP ------------------------------------------------------------------


def mcp_status(harness: str, scope: str = "user") -> bool:
    """Is a `somm` server configured? Reads the harness's config file."""
    home = Path.home()
    try:
        if harness == "claude":
            path = home / ".claude.json" if scope == "user" else Path.cwd() / ".mcp.json"
            return NAME in json.loads(path.read_text()).get("mcpServers", {})
        if harness == "codex":
            if scope != "user":
                return False
            path = home / ".codex" / "config.toml"
            return NAME in tomllib.loads(path.read_text()).get("mcp_servers", {})
    except (OSError, ValueError):
        return False
    raise ValueError(f"unsupported harness: {harness}")


def _mcp_argv(harness: str, action: str, scope: str, cmd: list[str]) -> list[str]:
    if harness == "claude":
        base = ["claude", "mcp", action, "--scope", scope, NAME]
    else:
        base = ["codex", "mcp", action, NAME]
    return base + (["--", *cmd] if action == "add" else [])


def _snippet(harness: str, scope: str, cmd: list[str]) -> str:
    if harness == "claude" and scope == "project":
        entry = {"mcpServers": {NAME: {"command": cmd[0], "args": cmd[1:]}}}
        return json.dumps(entry, indent=2) + "\n(add to .mcp.json in the project root)"
    if harness == "claude":
        return " ".join(_mcp_argv("claude", "add", scope, cmd))
    return (
        f"[mcp_servers.{NAME}]\ncommand = {json.dumps(cmd[0])}\nargs = {json.dumps(cmd[1:])}\n"
        "(add to ~/.codex/config.toml)"
    )


def mcp_apply(harness: str, action: str, scope: str, *, dry_run: bool) -> str:
    """Install (action="add") or uninstall ("remove"); returns a result message."""
    if harness not in HARNESSES:
        raise ValueError(f"unsupported harness: {harness} (supported: {', '.join(HARNESSES)})")
    if harness == "codex" and scope != "user":
        return "codex: has no project scope; use --scope user"
    cmd = mcp_command() if action == "add" else []
    argv = _mcp_argv(harness, action, scope, cmd)
    if dry_run:
        return f"{harness}: would run: {' '.join(argv)}"
    if not shutil.which(harness):
        if action == "add":
            return (
                f"{harness}: CLI not on PATH; paste this instead:\n"
                f"{_snippet(harness, scope, cmd)}"
            )
        return f"{harness}: CLI not on PATH; remove the '{NAME}' entry by hand"
    was = mcp_status(harness, scope)
    if action == "remove" and not was:
        return f"{harness}: not configured"
    if action == "add" and was:  # replace, so a stale interpreter path is refreshed
        _run(_mcp_argv(harness, "remove", scope, cmd))
    rc, out = _run(argv)
    verb = "installed" if action == "add" else "removed"
    return f"{harness}: {verb}" if rc == 0 else f"{harness}: FAILED ({out})"


# ---- skill ----------------------------------------------------------------


def skill_source() -> Path:
    spec = importlib.util.find_spec("somm_skill")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "somm-skill is not installed in this environment (pip install somm-skill)"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def skill_target(harness: str) -> Path:
    home = Path.home()
    if harness == "claude":
        return home / ".claude" / "skills" / NAME
    if harness == "codex":
        return home / ".agents" / "skills" / NAME
    raise ValueError(f"unsupported harness: {harness} (supported: {', '.join(HARNESSES)})")


def skill_status(harness: str) -> str:
    """missing | current | stale (broken link) | foreign (real directory).

    A link into another somm environment is current too: the skill is usually
    installed once, from the `somm` tool, and `somm doctor` is run from many
    project environments that each carry their own copy of the package.
    """
    t = skill_target(harness)
    if t.is_symlink():
        try:
            resolved = t.resolve()
            if resolved == skill_source().resolve() or (resolved / "SKILL.md").is_file():
                return "current"
        except (RuntimeError, OSError):
            pass
        return "stale"
    return "foreign" if t.exists() else "missing"


def skill_apply(harness: str, action: str, *, dry_run: bool) -> str:
    t = skill_target(harness)
    src = skill_source() if action == "add" else None
    if dry_run:
        what = f"link {t} -> {src}" if src else f"remove {t}"
        return f"{harness}: would {what}"
    state = skill_status(harness)
    if state == "foreign":
        return f"{harness}: {t} is a real directory, not a somm link; leaving it alone"
    if src is None:
        if state == "missing":
            return f"{harness}: not installed"
        t.unlink()
        return f"{harness}: removed"
    if state == "current":
        return f"{harness}: already installed ({t})"
    if t.is_symlink():
        t.unlink()
    t.parent.mkdir(parents=True, exist_ok=True)
    t.symlink_to(src, target_is_directory=True)
    return f"{harness}: installed ({t} -> {src})"


# ---- reporting ------------------------------------------------------------


def status_lines(harnesses: list[str]) -> list[str]:
    """MCP and skill state, one line per harness."""
    lines = []
    for h in harnesses:
        mcp = "configured" if mcp_status(h, "user") else "NOT configured (run `somm mcp install`)"
        if h == "claude" and mcp_status(h, "project"):
            mcp += " (+ project .mcp.json)"
        skill = skill_status(h)
        hint = {
            "missing": " (run `somm skill install`)",
            "stale": " (link is broken; run `somm skill install`)",
            "foreign": " (real directory, not managed by somm)",
        }.get(skill, "")
        lines.append(f"{h}: mcp {mcp}; skill {skill}{hint}")
    return lines

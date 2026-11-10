"""`somm mcp` / `somm skill` against a temp HOME with fake harness CLIs."""

from __future__ import annotations

import json
import stat
import sys

import pytest
from somm import connect
from somm.cli import main

# A fake harness CLI: logs its argv, and on `mcp add`/`remove` edits the
# config file the way the real one would, so `status` has something to read.
FAKE = """#!{py}
import json, os, sys, pathlib
home = pathlib.Path(os.environ["HOME"])
name = pathlib.Path(sys.argv[0]).name
with open(home / "calls.log", "a") as f:
    f.write(name + " " + " ".join(sys.argv[1:]) + "\\n")
action = sys.argv[2]
if name == "claude":
    p = home / ".claude.json"
    d = json.loads(p.read_text()) if p.exists() else {{}}
    if action == "add":
        i = sys.argv.index("--")
        d.setdefault("mcpServers", {{}})["somm"] = {{"command": sys.argv[i + 1]}}
    else:
        d.get("mcpServers", {{}}).pop("somm", None)
    p.write_text(json.dumps(d))
else:
    p = home / ".codex" / "config.toml"
    p.parent.mkdir(exist_ok=True)
    if action == "add":
        p.write_text('[mcp_servers.somm]\\ncommand = "x"\\n')
    else:
        p.write_text("")
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    bin_ = tmp_path / "bin"
    home.mkdir()
    bin_.mkdir()
    for n in ("claude", "codex"):
        f = bin_ / n
        f.write_text(FAKE.format(py=sys.executable))
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bin_))
    monkeypatch.chdir(tmp_path)
    return home


def calls(home):
    p = home / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


def test_mcp_print_changes_nothing(env, capsys):
    assert main(["mcp", "install", "--print"]) == 0
    out = capsys.readouterr().out
    assert "claude: would run: claude mcp add --scope user somm -- " in out
    assert sys.executable in out
    assert "codex: would run: codex mcp add somm -- " in out
    assert calls(env) == []


def test_mcp_install_status_uninstall_both(env, capsys):
    assert main(["mcp", "install"]) == 0
    assert "claude: installed" in capsys.readouterr().out
    log = calls(env)
    assert log[0].startswith("claude mcp add --scope user somm -- " + sys.executable)
    assert log[1].startswith("codex mcp add somm -- " + sys.executable)
    assert json.loads((env / ".claude.json").read_text())["mcpServers"]["somm"]
    assert connect.mcp_status("claude") and connect.mcp_status("codex")

    main(["mcp", "status"])
    out = capsys.readouterr().out
    assert "claude: mcp configured" in out and "codex: mcp configured" in out

    assert main(["mcp", "uninstall"]) == 0
    assert not connect.mcp_status("claude") and not connect.mcp_status("codex")
    main(["mcp", "uninstall", "--harness", "claude"])
    assert "claude: not configured" in capsys.readouterr().out


def test_mcp_reinstall_replaces_entry(env):
    main(["mcp", "install", "--harness", "claude"])
    main(["mcp", "install", "--harness", "claude"])
    assert any(c.startswith("claude mcp remove --scope user somm") for c in calls(env))


def test_mcp_missing_cli_prints_snippet(env, monkeypatch, capsys):
    (env.parent / "bin" / "codex").unlink()
    assert main(["mcp", "install", "--harness", "codex"]) == 0
    out = capsys.readouterr().out
    assert "CLI not on PATH" in out and "[mcp_servers.somm]" in out
    assert calls(env) == []


def test_mcp_default_targets_only_harnesses_on_path(env, capsys):
    (env.parent / "bin" / "codex").unlink()
    main(["mcp", "install"])
    out = capsys.readouterr().out
    assert "claude: installed" in out and "codex" not in out


def test_mcp_project_scope_claude_only(env, capsys):
    main(["mcp", "install", "--scope", "project", "--print"])
    out = capsys.readouterr().out
    assert "claude mcp add --scope project somm" in out
    assert "codex: has no project scope" in out


def test_skill_link_status_and_upgrade(env, capsys):
    target = env / ".claude" / "skills" / "somm"
    assert main(["skill", "install", "--harness", "claude", "--print"]) == 0
    assert not target.exists() and "would link" in capsys.readouterr().out

    assert main(["skill", "install"]) == 0
    assert target.is_symlink() and (target / "SKILL.md").is_file()
    assert (env / ".agents" / "skills" / "somm" / "SKILL.md").is_file()
    assert connect.skill_status("claude") == "current"

    # A link pointing somewhere else (an older install) is reported stale and repaired.
    target.unlink()
    old = env / "old"
    old.mkdir()
    target.symlink_to(old)
    assert connect.skill_status("claude") == "stale"
    main(["skill", "status", "--harness", "claude"])
    assert "skill stale" in capsys.readouterr().out
    main(["skill", "install", "--harness", "claude"])
    assert connect.skill_status("claude") == "current"

    main(["skill", "uninstall"])
    assert not target.exists() and not target.is_symlink()


def test_skill_never_clobbers_real_directory(env, capsys):
    target = env / ".claude" / "skills" / "somm"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("mine")
    main(["skill", "install", "--harness", "claude"])
    main(["skill", "uninstall", "--harness", "claude"])
    assert (target / "SKILL.md").read_text() == "mine"
    assert "leaving it alone" in capsys.readouterr().out


def test_doctor_reports_agent_connections(env, capsys):
    main(["doctor", "--project", "t_connect"])
    out = capsys.readouterr().out
    assert "agent claude: mcp NOT configured" in out
    assert "skill missing" in out

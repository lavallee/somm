# somm-skill

Onboarding guidance for coding agents working in projects that use `somm`.

`SKILL.md` is the canonical content — a Claude Code skill. Agent-specific
variants (Codex, Cursor, Windsurf, …) live under `templates/` and are derived
from the same core guidance.

## Installation

For Claude Code:

```bash
mkdir -p ~/.claude/skills/somm
cp packages/somm-skill/SKILL.md ~/.claude/skills/somm/SKILL.md
```

Or, once `somm mcp install --client=claude-code` lands in D2:

```bash
somm mcp install --client=claude-code   # installs MCP + skill in one step
```

For other agents:

- **Cursor:** copy `templates/cursor.md` into `.cursor/rules/` in your project.
- **Windsurf:** copy `templates/windsurf.md` into `.windsurf/rules/`.
- **Codex:** copy `templates/codex.md` into your project's Codex config.

(Agent-specific templates ship in D2+; v0.1 has only the canonical `SKILL.md`.)

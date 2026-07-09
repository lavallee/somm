# somm-skill

Onboarding guidance for coding agents working in projects that use
`somm`.

The package ships two markdown resources:

- `SKILL.md` — general LLM-call guidance for agents editing Python code.
- `SOMMELIER.md` — model-selection guidance for agents using `somm-mcp`.

## Installation

Install the package with the rest of the workspace:

```bash
pip install somm-skill
```

For Claude Code, copy the canonical skill into the local skill folder:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
import shutil

target = Path.home() / ".claude" / "skills" / "somm"
target.mkdir(parents=True, exist_ok=True)
for name in ("SKILL.md", "SOMMELIER.md"):
    shutil.copyfile(files("somm_skill") / name, target / name)
PY
```

For other agents, use the same two markdown files as the source of
truth and adapt only the surrounding packaging format.

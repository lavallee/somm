"""Fleet registry — which project databases live on this machine.

Metered-plan quotas (see somm_core.plans) are shared by every project
using the same provider account, so pacing must aggregate usage across
ALL local project databases, not just the current one. The registry is
how they find each other: every `somm.llm()` init upserts its project's
db_path into ``~/.somm/registry.json``.

Strictly local metadata (project name, db path, last-seen timestamp) —
nothing leaves the machine. Entries whose database has vanished are
pruned on read.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def registry_path() -> Path:
    env = os.environ.get("SOMM_REGISTRY_PATH")
    return Path(env) if env else Path.home() / ".somm" / "registry.json"


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("projects"), dict):
            return data
    except Exception:
        pass
    return {"projects": {}}


def register_project(project: str, db_path: Path) -> None:
    """Upsert this project's DB location. Never raises — registry
    maintenance must not break `somm.llm()`."""
    try:
        path = registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _load(path)
        data["projects"][project] = {
            "db_path": str(Path(db_path).resolve()),
            "last_seen": datetime.now(UTC).isoformat(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(path)
    except Exception:
        pass


def fleet_db_paths(include: Path | None = None) -> list[Path]:
    """Every registered project DB that still exists, deduplicated.
    `include` (usually the current project's DB) is always present."""
    seen: dict[str, Path] = {}
    if include is not None:
        p = Path(include).resolve()
        if p.exists():
            seen[str(p)] = p
    data = _load(registry_path())
    for entry in data["projects"].values():
        raw = entry.get("db_path")
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            seen[str(p)] = p
    return list(seen.values())

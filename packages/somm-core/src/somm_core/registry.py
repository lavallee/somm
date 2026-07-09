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
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_registered_projects: set[tuple[str, str]] = set()
_registered_projects_lock = threading.Lock()


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


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name per (pid, thread): a fixed .json.tmp lets two
    # concurrent writers clobber each other's temp before replace().
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


@contextmanager
def _registry_lock(path: Path):
    """Cross-process advisory lock over the registry file's read-modify-write.

    The in-process threading lock only orders threads; two separate somm
    processes initializing at once would each _load(), mutate their own
    project, and _write() — last writer wins, silently dropping the other's
    entry. An flock on a sidecar lockfile serializes the whole RMW across
    processes. Best-effort: if flock is unavailable (e.g. Windows) we fall
    back to no cross-process lock rather than failing registration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        os.close(fd)


def _is_pytest_tmp_path(path: Path) -> bool:
    raw = str(path)
    return "/pytest-" in raw or "/tmp/pytest" in raw


def _tmp_registration_allowed() -> bool:
    return os.environ.get("SOMM_REGISTRY_ALLOW_TMP") == "1"


def prune_registry() -> int:
    """Drop registry entries whose db_path no longer exists. Never raises."""
    try:
        path = registry_path()
        with _registry_lock(path):
            data = _load(path)
            projects = data.get("projects", {})
            kept = {
                project: entry
                for project, entry in projects.items()
                if entry.get("db_path") and Path(entry["db_path"]).exists()
            }
            pruned = len(projects) - len(kept)
            if pruned:
                data["projects"] = kept
                _write(path, data)
        return pruned
    except Exception:
        return 0


def registered_db_path(project: str) -> Path | None:
    """Return an existing registered DB path for project, if one is known."""
    try:
        entry = _load(registry_path()).get("projects", {}).get(project)
        if not entry:
            return None
        raw = entry.get("db_path")
        if not raw:
            return None
        path = Path(raw)
        return path if path.exists() else None
    except Exception:
        return None


def register_project(project: str, db_path: Path) -> None:
    """Upsert this project's DB location. Never raises — registry
    maintenance must not break `somm.llm()`."""
    try:
        db_key = str(Path(db_path).resolve())
        if _is_pytest_tmp_path(Path(db_key)) and not _tmp_registration_allowed():
            return
        key = (project, db_key)
        with _registered_projects_lock:
            if key in _registered_projects:
                return
            path = registry_path()
            # One cross-process lock over prune + upsert as a single RMW, so
            # a concurrent process can't read a stale registry and write back
            # a version missing our (or its) entry.
            with _registry_lock(path):
                data = _load(path)
                projects = data.setdefault("projects", {})
                # Inline prune under the same lock (drop vanished DBs).
                for gone in [
                    p
                    for p, e in projects.items()
                    if not (e.get("db_path") and Path(e["db_path"]).exists())
                ]:
                    del projects[gone]
                projects[project] = {
                    "db_path": db_key,
                    "last_seen": datetime.now(UTC).isoformat(),
                }
                _write(path, data)
            _registered_projects.add(key)
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

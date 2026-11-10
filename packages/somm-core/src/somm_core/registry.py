"""Fleet registry — which project databases live on this machine.

Metered-plan quotas (see somm_core.plans) are shared by every project
using the same provider account, so pacing must aggregate usage across
ALL local project databases, not just the current one. The registry is
how they find each other: every `somm.llm()` init upserts its project's
db_path into ``~/.somm/registry.json``.

Strictly local metadata (project name, db paths, last-seen timestamps) —
nothing leaves the machine. Paths whose database has vanished are pruned
whenever a project registers.

Entry format. A project can write to more than one ledger (a scratch run,
a moved checkout), so each entry keeps every path it has registered::

    {"db_path": "<most recent>", "last_seen": "<iso>",
     "db_paths": {"<path>": "<iso last seen>", ...}}

``db_path`` stays the most recently registered path, so older somm releases
read the file unchanged. An entry without ``db_paths`` (the format before
0.24) is read as a single-path entry.
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


def entry_paths(entry: dict) -> dict[str, str]:
    """``{path: last_seen}`` for every path an entry records, old format too."""
    out: dict[str, str] = {}
    raw = entry.get("db_paths")
    if isinstance(raw, dict):
        out.update({str(k): str(v or "") for k, v in raw.items() if k})
    current = entry.get("db_path")
    if current and current not in out:
        out[str(current)] = str(entry.get("last_seen") or "")
    return out


def _entry(paths: dict[str, str], current: str | None = None) -> dict:
    """Rebuild an entry from its paths; ``current`` defaults to the newest."""
    if current is None or current not in paths:
        current = max(paths, key=lambda k: paths[k])
    return {"db_path": current, "last_seen": paths[current], "db_paths": dict(sorted(paths.items()))}


def _drop_vanished(projects: dict) -> int:
    """Remove paths that no longer exist, and projects left with none.
    Returns the number of paths removed."""
    removed = 0
    for project in list(projects):
        entry = projects[project]
        paths = entry_paths(entry) if isinstance(entry, dict) else {}
        kept = {k: v for k, v in paths.items() if Path(k).exists()}
        removed += len(paths) - len(kept)
        if not kept:
            del projects[project]
        elif len(kept) != len(paths) or "db_paths" not in entry:
            current = entry.get("db_path")
            projects[project] = _entry(kept, current if current in kept else None)
    return removed


def prune_registry() -> int:
    """Drop registered paths whose database no longer exists. Never raises."""
    try:
        path = registry_path()
        with _registry_lock(path):
            data = _load(path)
            projects = data.get("projects", {})
            before = {k: entry_paths(v) for k, v in projects.items()}
            pruned = _drop_vanished(projects)
            if pruned or before != {k: entry_paths(v) for k, v in projects.items()}:
                data["projects"] = projects
                _write(path, data)
        return pruned
    except Exception:
        return 0


def registry_entries() -> dict[str, dict[str, str]]:
    """``{project: {path: last_seen}}`` exactly as registered, vanished paths
    included — the raw view `somm ledgers` reports on."""
    return {
        str(project): entry_paths(entry)
        for project, entry in _load(registry_path())["projects"].items()
        if isinstance(entry, dict)
    }


def registered_db_path(project: str) -> Path | None:
    """The project's most recently registered DB that still exists, if any."""
    try:
        entry = _load(registry_path()).get("projects", {}).get(project)
        if not entry:
            return None
        current = entry.get("db_path")
        if current and Path(current).exists():
            return Path(current)
        paths = entry_paths(entry)
        for raw in sorted(paths, key=lambda k: paths[k], reverse=True):
            if Path(raw).exists():
                return Path(raw)
        return None
    except Exception:
        return None


def register_project(project: str, db_path: Path) -> None:
    """Record that this project writes to ``db_path`` (kept beside any other
    path it has written to). Never raises — registry maintenance must not
    break `somm.llm()`."""
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
                _drop_vanished(projects)
                paths = entry_paths(projects.get(project) or {})
                paths[db_key] = datetime.now(UTC).isoformat()
                projects[project] = _entry(paths, db_key)
                _write(path, data)
            _registered_projects.add(key)
    except Exception:
        pass


def forget_project(project: str, *, path: Path | None = None, dry_run: bool = False) -> list[str]:
    """Remove a project's registration (or just one of its paths) and return
    the paths removed. Never touches a database file. Raises KeyError for a
    project or path that is not registered."""
    reg = registry_path()
    with _registry_lock(reg):
        data = _load(reg)
        projects = data["projects"]
        if project not in projects:
            raise KeyError(f"project {project!r} is not registered")
        paths = entry_paths(projects[project])
        if path is None:
            removed = sorted(paths)
        else:
            wanted = {str(path), str(Path(path).expanduser().resolve())}
            removed = [p for p in paths if p in wanted]
            if not removed:
                raise KeyError(f"{path} is not registered for {project!r}")
        if not dry_run:
            kept = {k: v for k, v in paths.items() if k not in removed}
            if kept:
                current = projects[project].get("db_path")
                projects[project] = _entry(kept, current if current in kept else None)
            else:
                del projects[project]
            _write(reg, data)
    return removed


def adopt_ledger(project: str, db_path: Path, *, dry_run: bool = False) -> bool:
    """Register an existing ledger for ``project``. Returns False when it was
    already registered. The project's current path is kept if it has one, so
    adopting an old ledger never redirects where `--project` reads; its
    last-seen time is the file's modification time."""
    db_key = str(Path(db_path).expanduser().resolve())
    if not Path(db_key).is_file():
        raise FileNotFoundError(db_key)
    reg = registry_path()
    with _registry_lock(reg):
        data = _load(reg)
        projects = data["projects"]
        entry = projects.get(project) or {}
        paths = entry_paths(entry)
        if db_key in paths:
            return False
        mtime = datetime.fromtimestamp(Path(db_key).stat().st_mtime, UTC).isoformat()
        if not dry_run:
            current = entry.get("db_path")
            paths[db_key] = mtime
            keep = current if current and Path(current).exists() else db_key
            projects[project] = _entry(paths, keep)
            _write(reg, data)
    return True


def registered_ledgers() -> list[tuple[str, Path]]:
    """``(project, db_path)`` for every registered ledger path that still exists
    (a project that writes to two ledgers appears twice)."""
    out: list[tuple[str, Path]] = []
    for project, paths in registry_entries().items():
        for raw in sorted(paths):
            p = Path(raw)
            if p.exists():
                out.append((project, p))
    return out


def fleet_db_paths(include: Path | None = None) -> list[Path]:
    """Every registered project DB that still exists, deduplicated.
    `include` (usually the current project's DB) is always present."""
    seen: dict[str, Path] = {}
    if include is not None:
        p = Path(include).resolve()
        if p.exists():
            seen[str(p)] = p
    for _project, p in registered_ledgers():
        seen.setdefault(str(p), p)
    return list(seen.values())

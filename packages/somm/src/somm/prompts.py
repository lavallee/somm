"""Prompt versioning — first-class Prompt objects.

A Prompt is `(workload_id, version, hash, body)`. Hash is content-addressed
(SHA-256 of body → 16 chars). Version is human-readable ("v1", "v1.2",
"v2").

Behavior:
- `llm.register_prompt(workload, body)` commits a new prompt version.
  If the body hash matches an existing prompt's hash, returns that prompt
  (idempotent).
  Otherwise: new minor bump from the latest version (v1 → v1.1, v1.3 → v1.4)
  unless `bump="major"` is set.
- `llm.prompt(workload, version="latest")` fetches. "latest" = highest
  version by the retired_at=NULL entries.
- `llm.prompt(workload, version="v2")` fetches a specific pinned version.

Prompts are never mutated — a "change" is a new version. Retirement is
soft (`retired_at` timestamp) so historical calls stay analyzable.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from somm_core.models import Prompt
from somm_core.parse import prompt_id as _prompt_id

if TYPE_CHECKING:
    from somm_core.repository import Repository


class PromptNotFound(Exception):
    pass


_VERSION_RE = re.compile(r"^v(\d+)(?:\.(\d+))?$")


def register_prompt(
    repo: Repository,
    workload_id: str,
    body: str,
    bump: str = "minor",
) -> Prompt:
    """Commit a prompt body. Returns the resulting Prompt.

    Idempotent: identical body (by hash) returns the existing entry without
    creating a new version.

    Args:
        bump: 'minor' (default), 'major', or an explicit version like 'v3'.
    """
    pid = _prompt_id(body)

    # Existing hash match → return existing (idempotent)
    with repo._open() as conn:
        row = conn.execute(
            "SELECT id, workload_id, version, hash, body, created_at, retired_at "
            "FROM prompts WHERE id = ? AND workload_id = ?",
            (pid, workload_id),
        ).fetchone()
    if row:
        return _row_to_prompt(row)

    # Find latest active version for this workload → bump
    latest = _latest_version(repo, workload_id)
    if bump.startswith("v") and _VERSION_RE.match(bump):
        new_version = bump
    else:
        new_version = _bump(latest, bump)

    with repo._open() as conn:
        conn.execute(
            "INSERT INTO prompts (id, workload_id, version, hash, body) VALUES (?, ?, ?, ?, ?)",
            (pid, workload_id, new_version, pid, body),
        )

    return Prompt(
        id=pid,
        workload_id=workload_id,
        version=new_version,
        hash=pid,
        body=body,
        created_at=datetime.now(UTC),
    )


def get_prompt(
    repo: Repository,
    workload_id: str,
    version: str = "latest",
) -> Prompt:
    """Fetch a prompt. Raises PromptNotFound if missing."""
    with repo._open() as conn:
        if version == "latest":
            # Tie-break on rowid — SQLite's CURRENT_TIMESTAMP is second-
            # resolution and multiple inserts in the same second collide.
            row = conn.execute(
                "SELECT id, workload_id, version, hash, body, created_at, retired_at "
                "FROM prompts WHERE workload_id = ? AND retired_at IS NULL "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (workload_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, workload_id, version, hash, body, created_at, retired_at "
                "FROM prompts WHERE workload_id = ? AND version = ?",
                (workload_id, version),
            ).fetchone()
    if not row:
        raise PromptNotFound(f"no prompt for workload {workload_id!r} version {version!r}")
    return _row_to_prompt(row)


def retire_prompt(repo: Repository, prompt_id: str) -> None:
    """Soft-retire a prompt. Historical calls still reference it."""
    with repo._open() as conn:
        conn.execute(
            "UPDATE prompts SET retired_at = ? WHERE id = ? AND retired_at IS NULL",
            (datetime.now(UTC).isoformat(), prompt_id),
        )


# ---------------------------------------------------------------------------


def _latest_version(repo: Repository, workload_id: str) -> str | None:
    with repo._open() as conn:
        row = conn.execute(
            "SELECT version FROM prompts WHERE workload_id = ? AND retired_at IS NULL "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (workload_id,),
        ).fetchone()
    return row[0] if row else None


def _bump(current: str | None, how: str) -> str:
    """Bump semver-ish. v1 -> v1.1 (minor); v1.3 -> v2 (major)."""
    if current is None:
        return "v1"
    m = _VERSION_RE.match(current)
    if not m:
        return "v1"
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) else 0
    if how == "major":
        return f"v{major + 1}"
    # minor
    return f"v{major}.{minor + 1}" if minor else f"v{major}.1"


def _row_to_prompt(row: tuple) -> Prompt:
    return Prompt(
        id=row[0],
        workload_id=row[1],
        version=row[2],
        hash=row[3],
        body=row[4],
        created_at=_maybe_ts(row[5]),
        retired_at=_maybe_ts(row[6]),
    )


def _maybe_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except ValueError:
        return None

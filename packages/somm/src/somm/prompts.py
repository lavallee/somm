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
- `llm.set_prompt_label(workload, "production", version="v2")` moves a
  mutable label pointer. Rollback is the same primitive: move the label back
  to an older prompt id/version and the move is recorded in history.
- `llm.prompt(workload, version="latest")` fetches. "latest" = highest
  version by the retired_at=NULL entries.
- `llm.prompt(workload, label="production")` resolves via prompt_labels.
  Explicit label wins over version. Weighted labels without a bucket key
  resolve to their highest-weight variant.
- `llm.prompt(workload, version="v2")` fetches a specific pinned version.
- `llm.fork_prompt(...)` creates a new prompt in the same workload with
  parent_prompt_id set to the source prompt id. It is lineage, not a new
  workload.

Prompts are never mutated — a "change" is a new version. Retirement is
soft (`retired_at` timestamp) so historical calls stay analyzable.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from somm_core.models import Prompt
from somm_core.parse import prompt_id as _prompt_id
from somm_core.parse import stable_hash

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
    existing = _prompt_by_id(repo, workload_id, _prompt_id(body))
    if existing is not None:
        return existing

    prompt = _register_prompt(repo, workload_id, body, bump=bump)
    set_label(repo, workload_id, "latest", prompt.id, updated_by="register_prompt")
    return prompt


def _register_prompt(
    repo: Repository,
    workload_id: str,
    body: str,
    bump: str = "minor",
    parent_prompt_id: str | None = None,
) -> Prompt:
    pid = _prompt_id(body)

    # Existing hash match → return existing (idempotent)
    existing = _prompt_by_id(repo, workload_id, pid)
    if existing is not None:
        return existing

    # Find latest active version for this workload → bump
    latest = _latest_version(repo, workload_id)
    new_version = (
        bump if bump.startswith("v") and _VERSION_RE.match(bump) else _bump(latest, bump)
    )

    with repo._open() as conn:
        conn.execute(
            "INSERT INTO prompts (id, workload_id, version, hash, body, parent_prompt_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pid, workload_id, new_version, pid, body, parent_prompt_id),
        )

    return Prompt(
        id=pid,
        workload_id=workload_id,
        version=new_version,
        hash=pid,
        body=body,
        created_at=datetime.now(UTC),
        parent_prompt_id=parent_prompt_id,
    )


def get_prompt(
    repo: Repository,
    workload_id: str,
    version: str = "latest",
    *,
    label: str | None = None,
) -> Prompt:
    """Fetch a prompt by version or label.

    Explicit label wins over version. Missing versions or labels raise
    PromptNotFound to preserve the existing get_prompt contract; use get_label
    when a missing label should return None.
    """
    if label is not None:
        prompt = resolve_label(repo, workload_id, label, bucket_key=None)
        if prompt is None:
            raise PromptNotFound(
                f"no prompt label {label!r} for workload {workload_id!r}"
            )
        return prompt

    with repo._open() as conn:
        if version == "latest":
            # Tie-break on rowid — SQLite's CURRENT_TIMESTAMP is second-
            # resolution and multiple inserts in the same second collide.
            row = conn.execute(
                "SELECT id, workload_id, version, hash, body, "
                "created_at, retired_at, parent_prompt_id "
                "FROM prompts WHERE workload_id = ? AND retired_at IS NULL "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (workload_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, workload_id, version, hash, body, "
                "created_at, retired_at, parent_prompt_id "
                "FROM prompts WHERE workload_id = ? AND version = ?",
                (workload_id, version),
            ).fetchone()
    if not row:
        raise PromptNotFound(f"no prompt for workload {workload_id!r} version {version!r}")
    return _row_to_prompt(row)


def set_label(
    repo: Repository,
    workload_id: str,
    label: str,
    prompt_id: str,
    updated_by: str | None = None,
) -> None:
    """Move a label to a prompt and append history.

    Rollback is this same primitive: call set_label with an older prompt_id.
    """
    with repo._open() as conn:
        row = conn.execute(
            "SELECT 1 FROM prompts WHERE id = ? AND workload_id = ?",
            (prompt_id, workload_id),
        ).fetchone()
        if row is None:
            raise PromptNotFound(
                f"prompt {prompt_id!r} does not exist for workload {workload_id!r}"
            )

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                INSERT INTO prompt_labels (workload_id, label, prompt_id, updated_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(workload_id, label) DO UPDATE SET
                    prompt_id = excluded.prompt_id,
                    weights_json = NULL,
                    updated_at = CURRENT_TIMESTAMP,
                    updated_by = excluded.updated_by
                """,
                (workload_id, label, prompt_id, updated_by),
            )
            conn.execute(
                """
                INSERT INTO prompt_label_history (workload_id, label, prompt_id, moved_by)
                VALUES (?, ?, ?, ?)
                """,
                (workload_id, label, prompt_id, updated_by),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def set_label_weights(
    repo: Repository,
    workload_id: str,
    label: str,
    weights: dict[str, float | int],
    updated_by: str | None = None,
) -> None:
    """Move a label to a deterministic weighted prompt distribution.

    Keys may be prompt versions (``v1.2``) or prompt ids. We store normalized
    weights keyed by prompt id and keep ``prompt_labels.prompt_id`` pointed at
    the highest-weight variant as the no-bucket fallback.
    """
    normalized = _normalize_label_weights(repo, workload_id, weights)
    fallback_id = max(normalized.items(), key=lambda item: (item[1], item[0]))[0]
    weights_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    with repo._open() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                INSERT INTO prompt_labels (
                    workload_id, label, prompt_id, weights_json, updated_by
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workload_id, label) DO UPDATE SET
                    prompt_id = excluded.prompt_id,
                    weights_json = excluded.weights_json,
                    updated_at = CURRENT_TIMESTAMP,
                    updated_by = excluded.updated_by
                """,
                (workload_id, label, fallback_id, weights_json, updated_by),
            )
            conn.execute(
                """
                INSERT INTO prompt_label_history (workload_id, label, prompt_id, moved_by)
                VALUES (?, ?, ?, ?)
                """,
                (workload_id, label, fallback_id, updated_by),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def get_label(repo: Repository, workload_id: str, label: str) -> Prompt | None:
    """Resolve a prompt label, returning None when the label is unknown."""
    return resolve_label(repo, workload_id, label, bucket_key=None)


def resolve_label(
    repo: Repository,
    workload_id: str,
    label: str,
    bucket_key: str | None,
) -> Prompt | None:
    """Resolve a label to a Prompt, applying weighted A/B selection if present.

    Weighted labels use ``stable_hash(bucket_key)`` for deterministic bucketing.
    When ``bucket_key`` is None, resolution falls back to the highest-weight
    variant, matching ``get_prompt(label=...)``'s deterministic behavior.
    """
    with repo._open() as conn:
        row = conn.execute(
            """
            SELECT prompt_id, weights_json
            FROM prompt_labels
            WHERE workload_id = ? AND label = ?
            """,
            (workload_id, label),
        ).fetchone()
    if row is None:
        return None

    prompt_id = row[0]
    weights_json = row[1] if len(row) > 1 else None
    if weights_json:
        weights = _loads_weights(weights_json)
        if weights:
            prompt_id = _choose_weighted_prompt_id(weights, bucket_key) or prompt_id

    return _prompt_by_id(repo, workload_id, prompt_id)


def list_labels(repo: Repository, workload_id: str) -> dict[str, Prompt]:
    """Return all labels for a workload keyed by label name."""
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT l.label, p.id, p.workload_id, p.version, p.hash, p.body,
                   p.created_at, p.retired_at, p.parent_prompt_id
            FROM prompt_labels AS l
            JOIN prompts AS p ON p.id = l.prompt_id AND p.workload_id = l.workload_id
            WHERE l.workload_id = ?
            ORDER BY l.label
            """,
            (workload_id,),
        ).fetchall()
    return {row[0]: _row_to_prompt(row[1:]) for row in rows}


def list_label_pointers(repo: Repository, workload_id: str) -> dict[str, dict]:
    """Return label pointer metadata, including weighted distributions."""
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT l.label, l.prompt_id, l.weights_json, p.version
            FROM prompt_labels AS l
            JOIN prompts AS p ON p.id = l.prompt_id AND p.workload_id = l.workload_id
            WHERE l.workload_id = ?
            ORDER BY l.label
            """,
            (workload_id,),
        ).fetchall()
        version_rows = conn.execute(
            "SELECT id, version FROM prompts WHERE workload_id = ?",
            (workload_id,),
        ).fetchall()
    versions_by_id = {row[0]: row[1] for row in version_rows}
    out: dict[str, dict] = {}
    for label, prompt_id, weights_json, version in rows:
        weights = _loads_weights(weights_json)
        out[label] = {
            "prompt_id": prompt_id,
            "version": version,
            "weights": weights,
            "versions": {pid: versions_by_id.get(pid, pid) for pid in weights},
        }
    return out


def label_history(repo: Repository, workload_id: str, label: str) -> list[dict]:
    """Return the append-only audit trail for a prompt label."""
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT id, workload_id, label, prompt_id, moved_at, moved_by
            FROM prompt_label_history
            WHERE workload_id = ? AND label = ?
            ORDER BY moved_at ASC, id ASC
            """,
            (workload_id, label),
        ).fetchall()
    return [
        {
            "id": row[0],
            "workload_id": row[1],
            "label": row[2],
            "prompt_id": row[3],
            "moved_at": row[4],
            "moved_by": row[5],
        }
        for row in rows
    ]


def fork_prompt(
    repo: Repository,
    workload_id: str,
    from_version_or_label: str,
    new_body: str,
    updated_by: str | None = None,
) -> Prompt:
    """Branch a variant in the same workload with recorded lineage."""
    source = _resolve_prompt_ref(repo, workload_id, from_version_or_label)
    prompt = _register_prompt(
        repo,
        workload_id,
        new_body,
        parent_prompt_id=source.id,
    )
    set_label(
        repo,
        workload_id,
        "latest",
        prompt.id,
        updated_by=updated_by or "fork_prompt",
    )
    return prompt


def prompt_ids_for_workload(repo: Repository, workload_id: str) -> set[str]:
    """Return registered prompt ids for a workload."""
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT id FROM prompts WHERE workload_id = ?",
            (workload_id,),
        ).fetchall()
    return {row[0] for row in rows}


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


def _prompt_by_id(repo: Repository, workload_id: str, prompt_id: str) -> Prompt | None:
    with repo._open() as conn:
        row = conn.execute(
            "SELECT id, workload_id, version, hash, body, "
            "created_at, retired_at, parent_prompt_id "
            "FROM prompts WHERE id = ? AND workload_id = ?",
            (prompt_id, workload_id),
        ).fetchone()
    return _row_to_prompt(row) if row else None


def _resolve_prompt_ref(repo: Repository, workload_id: str, version_or_label: str) -> Prompt:
    labeled = get_label(repo, workload_id, version_or_label)
    if labeled is not None:
        return labeled
    return get_prompt(repo, workload_id, version=version_or_label)


def _normalize_label_weights(
    repo: Repository,
    workload_id: str,
    weights: dict[str, float | int],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for ref, raw_weight in weights.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ValueError(f"invalid weight for {ref!r}: {raw_weight!r}") from None
        if weight <= 0:
            raise ValueError(f"weight for {ref!r} must be > 0")
        prompt = _prompt_by_id(repo, workload_id, ref)
        if prompt is None:
            prompt = get_prompt(repo, workload_id, version=ref)
        totals[prompt.id] = totals.get(prompt.id, 0.0) + weight
    if not totals:
        raise ValueError("weighted label must include at least one prompt")
    total = sum(totals.values())
    return {pid: weight / total for pid, weight in sorted(totals.items())}


def _loads_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, float] = {}
    for prompt_id, weight in loaded.items():
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out[str(prompt_id)] = value
    return out


def _choose_weighted_prompt_id(
    weights: dict[str, float],
    bucket_key: str | None,
) -> str | None:
    if not weights:
        return None
    if bucket_key is None:
        return max(weights.items(), key=lambda item: (item[1], item[0]))[0]

    total = sum(weights.values())
    if total <= 0:
        return None
    space = 16**16
    point = int(stable_hash(str(bucket_key)), 16) / space * total
    cumulative = 0.0
    chosen = None
    for prompt_id, weight in sorted(weights.items()):
        cumulative += weight
        chosen = prompt_id
        if point < cumulative:
            return prompt_id
    return chosen


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
        parent_prompt_id=row[7] if len(row) > 7 else None,
    )


def _maybe_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except ValueError:
        return None

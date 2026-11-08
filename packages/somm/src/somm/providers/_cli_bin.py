"""Locate a CLI executor's binary (``claude``, ``codex``) and explain a miss.

The seat executors used to be "configured" iff ``shutil.which`` found the
binary on PATH. A systemd user unit, a cron job or a launchd agent runs
with a PATH that lacks ``~/.local/bin``, so a CLI that was installed and
signed in read as ``provider 'claude-cli' not configured`` with no hint
(Frampton on Chiba, somm 0.19.3). Resolution now tries, in order:

1. an explicit path — ``SOMM_CLAUDE_BIN`` / ``SOMM_CODEX_BIN`` or the
   ``claude_cli_bin`` / ``codex_cli_bin`` keys under ``[tool.somm]``;
2. ``shutil.which`` on the current PATH;
3. the well-known per-user and package-manager bin dirs below.

A miss reports exactly what was searched and how to point at the binary.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Where the vendors' installers put the binary when it is not on the unit's
# PATH. Order is the search order; keep it short so the miss message stays
# readable. Tests monkeypatch this to () to simulate "not installed".
WELL_KNOWN_BIN_DIRS: tuple[str, ...] = (
    "~/.local/bin",
    "~/.npm-global/bin",
    "~/.volta/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
)


@dataclass(frozen=True, slots=True)
class BinaryLookup:
    """Where a CLI binary was found, or why it was not.

    ``path`` is the executable to run (``None`` on a miss). ``detail`` is a
    one-line, operator-facing explanation either way: the resolved path, or
    what was searched and which setting points somm at the binary.
    """

    path: str | None
    detail: str

    @property
    def found(self) -> bool:
        return self.path is not None


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def resolve_cli_binary(name: str, *, explicit: str | None, env_var: str) -> BinaryLookup:
    """Resolve ``name`` (e.g. ``"claude"``) to an executable path.

    ``explicit`` is the configured override (already merged from env and
    ``[tool.somm]`` by ``Config``); ``env_var`` is named in the miss message
    so the fix is one line away from the error.
    """
    if explicit:
        candidate = os.path.expanduser(explicit)
        if _is_executable(candidate):
            return BinaryLookup(candidate, candidate)
        if os.sep not in candidate:
            # A bare name is still a request to look it up, just under that name.
            found = shutil.which(candidate)
            if found:
                return BinaryLookup(found, found)
        return BinaryLookup(
            None,
            f"{name} CLI not found: {env_var}={explicit!r} is not an executable file",
        )

    found = shutil.which(name)
    if found:
        return BinaryLookup(found, found)

    for bin_dir in WELL_KNOWN_BIN_DIRS:
        candidate = str(Path(bin_dir).expanduser() / name)
        if _is_executable(candidate):
            return BinaryLookup(candidate, candidate)

    searched = ", ".join(WELL_KNOWN_BIN_DIRS) if WELL_KNOWN_BIN_DIRS else "no well-known dirs"
    return BinaryLookup(
        None,
        f"{name} CLI not found: not on PATH, not in {searched}; "
        f"set {env_var}=/path/to/{name}",
    )


__all__ = ["WELL_KNOWN_BIN_DIRS", "BinaryLookup", "resolve_cli_binary"]

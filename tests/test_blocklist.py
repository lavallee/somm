"""Blocklist CI guard — fails if internal project names leak into the repo.

Run on every CI build (and pre-commit locally). Scans source + docs +
examples + test fixtures for strings that would identify the author's
personal projects or data domains. Keep the blocklist here in this file
(not in .gitignore); it's a safety net, not a secret.

Add banned strings to BANNED below. Keep the list narrow — false
positives waste contributor time. `CONTEXT_ALLOW` exempts files where
the string has a legitimate non-leak meaning.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Internal project names from author's private codebases. Do NOT commit
# these strings to ANY tracked file. Matched word-boundary-aware to
# avoid false positives on substrings of innocent English words
# (e.g. "tern" is a substring of "pattern").
BANNED = [
    "barnowl",
    "paperboy",
    "ScoutLLM",
    # NB: "tern" and "jay" are handled via word-boundary regex below to
    # avoid matching "pattern", "staging", "jay walk". They're still
    # banned as standalone words.
]

BANNED_WORDS = [
    "tern",
    "jay",
    "communities",  # user's internal project name, not the dictionary word
    "butterfly",
    "landscapes",
    "magpie",
    "catbird",
    "downunder",
    "keel",
    "weaver",
    "crow",
]

# Paths to scan. Anything outside this set is implicitly trusted.
SCAN_DIRS = ["packages", "examples", "docs", "tests"]
SCAN_FILES = ["README.md", "CHANGELOG.md", "PLAN.md", "TODOS.md"]

# File extensions to check. Binary/generated files skipped.
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".txt",
    ".css",
    ".html",
    ".js",
    ".ts",
    ".tsx",
}

# Specific files allowed to mention banned terms (e.g., this file itself,
# because it defines the blocklist).
ALLOW_PATHS = {
    "tests/test_blocklist.py",
}


def _iter_tracked_files() -> list[Path]:
    """Walk SCAN_DIRS + SCAN_FILES, returning text files."""
    out: list[Path] = []
    for name in SCAN_DIRS:
        root = REPO_ROOT / name
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in TEXT_SUFFIXES:
                out.append(p)
    for name in SCAN_FILES:
        p = REPO_ROOT / name
        if p.exists() and p.suffix in TEXT_SUFFIXES:
            out.append(p)
    return out


def test_no_banned_exact_names():
    """Exact-match banned strings must not appear in tracked files."""
    offenders: list[tuple[str, str, int]] = []
    for path in _iter_tracked_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOW_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for banned in BANNED:
            if banned in text:
                # Report first line with the match
                for i, line in enumerate(text.splitlines(), start=1):
                    if banned in line:
                        offenders.append((rel, banned, i))
                        break
    assert not offenders, "internal names leaked into tracked files:\n" + "\n".join(
        f"  {rel}:{line}  -> {name!r}" for rel, name, line in offenders
    )


def test_no_banned_words_with_boundaries():
    """Banned standalone words must not appear as whole words."""
    offenders: list[tuple[str, str, int]] = []
    patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in BANNED_WORDS]
    for path in _iter_tracked_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOW_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for word, pat in zip(BANNED_WORDS, patterns, strict=True):
                if pat.search(line):
                    offenders.append((rel, word, i))
                    break  # one report per line
    assert not offenders, "internal word-bounded names leaked:\n" + "\n".join(
        f"  {rel}:{line}  -> {word!r}" for rel, word, line in offenders
    )


def test_no_personal_paths():
    """No /home/<someone>/ or /Users/<someone>/ paths should ship."""
    offenders: list[tuple[str, str, int]] = []
    pattern = re.compile(r"/home/marc\b|/Users/[a-zA-Z_-]+/")
    for path in _iter_tracked_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOW_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            m = pattern.search(line)
            if m:
                offenders.append((rel, m.group(0), i))
    assert not offenders, "personal paths leaked:\n" + "\n".join(
        f"  {rel}:{line}  -> {path!r}" for rel, path, line in offenders
    )

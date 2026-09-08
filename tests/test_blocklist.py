"""Blocklist CI guard — fails if internal names or personal paths leak into the repo.

Run on every CI build (and pre-commit locally). Scans *every tracked text
file* for strings that would identify the author's private projects,
machines, or filesystem layout. Keep the blocklist here in this file (not
in .gitignore); it's a safety net, not a secret.

Two rules make this guard trustworthy:

1. **Deny by default.** Coverage comes from ``git ls-files``, not from a
   list of directories. A directory allowlist silently exempts every new
   top-level directory added later — which is exactly how ``site/`` and
   ``skillopt/`` came to ship absolute local paths.
2. **The detector must not leak.** Banned tokens are assembled from
   fragments at import time so the private strings never appear as
   literals in this public file. Read ``_token`` calls as the joined
   string; don't "tidy" them back into single literals.

Add banned strings to BANNED / BANNED_WORDS below. Keep the lists narrow —
false positives waste contributor time. ALLOW_PATHS exempts files where a
match has a legitimate non-leak meaning.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _token(*parts: str) -> str:
    """Join fragments into a banned token.

    Assembling at runtime keeps the private string out of this file's own
    text, so the leak detector does not itself leak. (Borrowed from
    Platform-Studio/orchestra's ``scripts/audit_public_repo.py``.)
    """
    return "".join(parts)


# Internal project names from the author's private codebases. Do NOT commit
# these strings to ANY tracked file. Matched as exact substrings.
BANNED = [
    _token("barn", "owl"),
    _token("paper", "boy"),
    _token("Scout", "LLM"),
]

# Banned as standalone words only, so innocent English (e.g. "pattern"
# containing "tern", or "staging" containing "jay") does not trip the guard.
BANNED_WORDS = [
    _token("te", "rn"),
    _token("j", "ay"),
    _token("commun", "ities"),  # internal project name, not the dictionary word
    _token("butter", "fly"),
    _token("land", "scapes"),
    _token("mag", "pie"),
    _token("cat", "bird"),
    _token("down", "under"),
    _token("ke", "el"),
    _token("wea", "ver"),
    _token("cr", "ow"),
    _token("scr", "ibe"),
    _token("ma", "lo"),
    _token("star", "board"),
    _token("or", "ca"),
    _token("ste", "ve"),
    _token("e", "no"),
    _token("del", "phi"),
    _token("stet", "son"),
    _token("imo", "gene"),
    _token("traw", "ler"),
    # Private CEO agents. Distinct from the sibling *systems* somm names on
    # purpose (Fab, George, Spindle, Milton, Chip): those are published
    # integration points and must stay greppable. Agents are not.
    _token("arch", "ie"),
    _token("astr", "id"),
]

# Private infrastructure: host names and network addresses.
BANNED_HOSTS = [
    _token("dash", "-main"),
    _token("100.86.", "199."),
]

# File extensions to check. Binary/generated files skipped.
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".sql", ".yaml", ".yml", ".json", ".sh",
    ".txt", ".css", ".html", ".js", ".ts", ".tsx", ".cfg", ".ini", ".rst",
}

# Files exempt from every rule. Use sparingly — a whole-file exemption also
# hides personal paths and hosts, not just the name you meant to allow.
ALLOW_PATHS = {
    # This file defines the blocklist.
    "tests/test_blocklist.py",
    # Generated third-party data: public model catalog ids can collide with
    # banned words (e.g. openrouter's "mancer/weaver") without being leaks.
    "packages/somm-core/src/somm_core/data/pricing_bundle.json",
}

# Preferred over ALLOW_PATHS: exempt one file from one token, leaving every
# other rule in force for it. {relative path: {token, ...}}.
ALLOW_TOKENS: dict[str, set[str]] = {}


def _is_camel_case_hit(matched: str) -> bool:
    """True when the hit is really a camelCase identifier, e.g. ``cRow``.

    ``\bcrow\b`` with IGNORECASE matches ``cRow`` because ``c`` and ``R``
    are both word characters, so no boundary separates them. Prose forms
    (``crow``, ``Crow``, ``CROW``) are kept; a lowercase run followed by an
    interior capital is not a word, it is an identifier.
    """
    if matched.islower() or matched.isupper() or matched.istitle():
        return False
    return any(
        ch.isupper() and matched[:i].islower()
        for i, ch in enumerate(matched)
        if i > 0
    )

def _iter_tracked_files() -> list[Path]:
    """Return every tracked text file. Deny by default — no directory allowlist."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    out: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        path = REPO_ROOT / rel
        if path.suffix in TEXT_SUFFIXES and path.is_file():
            out.append(path)
    return out


def _scannable() -> list[tuple[str, str]]:
    """Yield (relative path, text) for each tracked text file not allowlisted."""
    items: list[tuple[str, str]] = []
    for path in _iter_tracked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOW_PATHS:
            continue
        try:
            items.append((rel, path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return items


def _allowed(rel: str, token: str) -> bool:
    """True when this one file is exempt from this one token."""
    return token.lower() in {t.lower() for t in ALLOW_TOKENS.get(rel, set())}


def test_no_banned_exact_names():
    """Exact-match banned strings must not appear in tracked files."""
    offenders: list[tuple[str, str, int]] = []
    for rel, text in _scannable():
        for banned in BANNED:
            if _allowed(rel, banned):
                continue
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
    patterns = [
        (word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
        for word in BANNED_WORDS
    ]
    for rel, text in _scannable():
        for i, line in enumerate(text.splitlines(), start=1):
            for word, pat in patterns:
                if _allowed(rel, word):
                    continue
                m = pat.search(line)
                if m and not _is_camel_case_hit(m.group(0)):
                    offenders.append((rel, word, i))
                    break  # one report per line
    assert not offenders, "internal word-bounded names leaked:\n" + "\n".join(
        f"  {rel}:{line}  -> {word!r}" for rel, word, line in offenders
    )


def test_no_private_hosts():
    """Private host names and network addresses must not ship."""
    offenders: list[tuple[str, str, int]] = []
    for rel, text in _scannable():
        for i, line in enumerate(text.splitlines(), start=1):
            for host in BANNED_HOSTS:
                if _allowed(rel, host):
                    continue
                if host.lower() in line.lower():
                    offenders.append((rel, host, i))
                    break
    assert not offenders, "private infrastructure identifiers leaked:\n" + "\n".join(
        f"  {rel}:{line}  -> {host!r}" for rel, host, line in offenders
    )


def test_no_personal_paths():
    """No /home/<someone>/, /Users/<someone>/, or /media/… paths should ship."""
    offenders: list[tuple[str, str, int]] = []
    pattern = re.compile(
        r"/home/(?!user\b|you\b|USER\b|username\b|runner\b)[a-zA-Z0-9_-]+"
        r"|/Users/(?!you\b|user\b|USER\b|username\b)[a-zA-Z0-9_-]+"
        r"|/media/[a-zA-Z0-9_-]+"
        r"|~/\.gstack"
    )
    for rel, text in _scannable():
        for i, line in enumerate(text.splitlines(), start=1):
            m = pattern.search(line)
            if m:
                offenders.append((rel, m.group(0), i))
    assert not offenders, "personal paths leaked:\n" + "\n".join(
        f"  {rel}:{line}  -> {path!r}" for rel, path, line in offenders
    )


def test_guard_covers_every_tracked_directory():
    """Coverage must be deny-by-default: every tracked text file is scanned.

    Regression test for the directory-allowlist model this guard replaced,
    under which ``site/`` and ``skillopt/`` were never scanned.
    """
    scanned = {rel for rel, _ in _scannable()}
    tracked = {
        p.relative_to(REPO_ROOT).as_posix()
        for p in _iter_tracked_files()
    } - ALLOW_PATHS
    assert scanned == tracked, (
        "tracked text files not covered by the blocklist scan:\n"
        + "\n".join(f"  {rel}" for rel in sorted(tracked - scanned))
    )

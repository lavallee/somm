"""Guard: the `--help` text of every command and subcommand must not change by accident.

The snapshot lives in ``fixtures/cli_help.txt``. When you change a flag on
purpose, regenerate it and review the diff:

    UPDATE_CLI_SNAPSHOT=1 python -m pytest packages/somm/tests/test_cli_help_snapshot.py

Without that variable the test only compares.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from somm.cli import build_parser

FIXTURE = Path(__file__).parent / "fixtures" / "cli_help.txt"


def _walk(parser: argparse.ArgumentParser, path: tuple[str, ...]):
    yield path, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            seen: set[int] = set()
            for name, sub in action.choices.items():
                if id(sub) in seen:  # an alias points at the same parser
                    continue
                seen.add(id(sub))
                yield from _walk(sub, (*path, name))


def render_help() -> str:
    prev = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "100"
    try:
        chunks = []
        for path, parser in _walk(build_parser(), ("somm",)):
            chunks.append(f"===== {' '.join(path)} =====\n{parser.format_help()}")
        return "\n".join(chunks)
    finally:
        if prev is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = prev


def test_cli_help_matches_snapshot():
    text = render_help()
    if os.environ.get("UPDATE_CLI_SNAPSHOT"):
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(text)
    assert FIXTURE.exists(), "run once with UPDATE_CLI_SNAPSHOT=1"
    assert text == FIXTURE.read_text(), (
        "a command's --help changed; if deliberate, regenerate with "
        "UPDATE_CLI_SNAPSHOT=1 and review the diff"
    )

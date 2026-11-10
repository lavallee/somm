"""`--project` is accepted at the root, before a subcommand and after it, on every command."""

from __future__ import annotations

import argparse

import pytest
from somm.cli import _common, build_parser


def _placeholder(action: argparse.Action) -> list[str]:
    if action.choices:
        value = str(next(iter(action.choices)))
    elif action.type is int:
        value = "1"
    elif action.type is float:
        value = "0.5"
    else:
        value = "x"
    if not action.option_strings:
        count = 1 if action.nargs is None else (1 if action.nargs in ("+",) else 0)
        return [value] * count
    if action.nargs == 0:
        return [action.option_strings[0]]
    return [action.option_strings[0], value]


def _required(parser: argparse.ArgumentParser) -> list[str]:
    argv: list[str] = []
    grouped: set[int] = set()
    for group in parser._mutually_exclusive_groups:
        if group.required:
            argv += _placeholder(group._group_actions[0])
        grouped.update(id(a) for a in group._group_actions)
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if id(action) in grouped:
            continue
        if action.option_strings and not action.required:
            continue
        if not action.option_strings and action.nargs in ("?", "*"):
            continue
        argv += _placeholder(action)
    return argv


def _leaves(parser, path=()):
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subs:
        yield path, parser
        return
    seen: set[int] = set()
    for name, sub in subs[0].choices.items():
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        yield from _leaves(sub, (*path, name))


def _takes_project(parser: argparse.ArgumentParser) -> bool:
    return any("--project" in a.option_strings and a.dest == "project" for a in parser._actions)


def _cases():
    root = build_parser()
    for path, leaf in _leaves(root):
        if path[0] == "chatgpt" or not _takes_project(leaf):
            continue
        yield pytest.param(path, leaf, id=" ".join(path))


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


@pytest.mark.parametrize("path,leaf", list(_cases()))
def test_project_accepted_at_every_position(path, leaf):
    rest = _required(leaf)
    # Spelled after the leaf command (the only place the leaf itself takes it).
    assert _parse([*path, *rest, "--project", "p"]).project == "p"
    # At the root, before any command.
    assert _parse(["--project", "p", *path, *rest]).project == "p"
    # Between each group and its subcommand.
    for depth in range(1, len(path)):
        argv = [*path[:depth], "--project", "p", *path[depth:], *rest]
        assert _parse(argv).project == "p", argv
    # Absent everywhere: None, so SOMM_PROJECT / the directory decides.
    assert _parse([*path, *rest]).project is None


def test_project_may_repeat_and_last_wins():
    args = _parse(["--project", "a", "inbox", "--project", "b", "list", "--project", "c"])
    assert args.project == "c"


def test_somm_project_env_is_used_when_no_flag(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(_common, "_load_config", lambda **kw: seen.update(kw) or object())
    monkeypatch.setenv("SOMM_PROJECT", "from-env")
    _common.load_config(project=None)
    assert seen["project"] == "from-env" and seen["prefer_registry"] is True
    _common.load_config(project="flag")
    assert seen["project"] == "flag"

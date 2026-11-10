"""somm mcp / somm skill: register the MCP server and skill with coding agents."""

from __future__ import annotations

import argparse


def _cmd_connect(args: argparse.Namespace) -> int:
    from somm import connect

    targets = [args.harness] if args.harness else connect.found_harnesses()
    if not targets:
        print("no supported harness (claude, codex) on PATH; pass --harness")
        return 1
    if args.connect_cmd == "status":
        print("\n".join(connect.status_lines(targets)))
        return 0
    action = "add" if args.connect_cmd == "install" else "remove"
    for h in targets:
        if args.connect_group == "mcp":
            print(connect.mcp_apply(h, action, args.scope, dry_run=args.print))
        else:
            print(connect.skill_apply(h, action, dry_run=args.print))
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    for group, noun, helptext in (
        ("mcp", "the somm MCP server", "register the somm MCP server with coding agents"),
        ("skill", "the somm skill", "link the somm skill into coding agents' skill directories"),
    ):
        pg = sub.add_parser(group, help=helptext)
        pg_sub = pg.add_subparsers(dest="connect_cmd", required=True)
        for name, text in (
            ("install", f"install {noun}"),
            ("uninstall", f"remove {noun}"),
            ("status", f"show whether {noun} is set up"),
        ):
            pc = pg_sub.add_parser(name, help=text)
            pc.add_argument(
                "--harness",
                choices=["claude", "codex"],
                default=None,
                help="default: every supported harness found on PATH",
            )
            if name != "status":
                pc.add_argument(
                    "--print", action="store_true", help="show what would be done; change nothing"
                )
            if name != "status" and group == "mcp":
                pc.add_argument(
                    "--scope",
                    choices=["user", "project"],
                    default="user",
                    help="claude only (codex is user-level); default: user",
                )
            pc.set_defaults(func=_cmd_connect, connect_group=group)

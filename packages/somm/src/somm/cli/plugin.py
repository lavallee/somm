"""somm plugin: list and inspect plugins, hooks and providers."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys


def _provider_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "pinned-only"


def _reference_plugin_register_name(name: str) -> str:
    return f"somm.plugins.{name}.register()"


def _reference_plugin_register_fn(spec: dict):
    module = importlib.import_module(spec["module"])
    return module.register


def _cmd_plugin_list(args: argparse.Namespace) -> int:
    from somm import hooks, plugins
    from somm.providers.registry import (
        BUILTIN_PROVIDER_SPECS,
        load_entrypoint_provider_specs,
    )

    print("REFERENCE PLUGINS")
    if not plugins.REFERENCE_PLUGINS:
        print("  no reference plugins available")
    else:
        print(f"  {'name':<16} {'phase':<14} {'enable':<42} needs")
        for name, spec in sorted(plugins.REFERENCE_PLUGINS.items()):
            extra = spec.get("extra")
            needs = f"pip install somm[{extra}]" if extra else "-"
            print(
                f"  {name:<16} {spec['phase']:<14} "
                f"{_reference_plugin_register_name(name):<42} {needs}"
            )
            print(f"    {spec['summary']}")

    print()
    print("ACTIVE HOOKS")
    hooks.load_entry_points()
    active = hooks.registered_hooks()
    if not any(active.get(phase) for phase in hooks.HOOK_PHASES):
        print("  no active hooks")
    else:
        for phase in hooks.HOOK_PHASES:
            rows = active.get(phase, [])
            print(f"  {phase}")
            if not rows:
                print("    none")
                continue
            print(f"    {'priority':>8}  callable")
            for qualname, priority in rows:
                print(f"    {priority:>8}  {qualname}")

    print()
    print("PROVIDERS")
    print("  built-in")
    print(f"    {'name':<18} rank")
    for spec in BUILTIN_PROVIDER_SPECS:
        print(f"    {spec.name:<18} {_provider_rank(spec.default_order_rank)}")
    entrypoint_specs = load_entrypoint_provider_specs()
    print("  entry-point")
    if not entrypoint_specs:
        print("    no entry-point providers")
    else:
        print(f"    {'name':<18} rank")
        for spec in entrypoint_specs:
            print(f"    {spec.name:<18} {_provider_rank(spec.default_order_rank)}")
    return 0


def _cmd_plugin_info(args: argparse.Namespace) -> int:
    from somm import plugins

    spec = plugins.REFERENCE_PLUGINS.get(args.name)
    if spec is None:
        valid = ", ".join(sorted(plugins.REFERENCE_PLUGINS)) or "none"
        print(
            f"Unknown reference plugin {args.name!r}. Valid plugins: {valid}.",
            file=sys.stderr,
        )
        return 2

    register_fn = _reference_plugin_register_fn(spec)
    signature = inspect.signature(register_fn)
    extra = spec.get("extra")
    print(f"name: {args.name}")
    print(f"summary: {spec['summary']}")
    print(f"phase: {spec['phase']}")
    print(f"module: {spec['module']}")
    print(f"register: register{signature}")
    print(f"enable: {_reference_plugin_register_name(args.name)}")
    print(f"extra: {f'pip install somm[{extra}]' if extra else '-'}")
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    pp = sub.add_parser("plugin", help="list and inspect plugins, hooks, and providers")
    pp_sub = pp.add_subparsers(dest="plugin_cmd", required=True)

    pplg = pp_sub.add_parser("list", help="list reference plugins, active hooks, and providers")
    pplg.set_defaults(func=_cmd_plugin_list)

    ppi = pp_sub.add_parser("info", help="show one reference plugin")
    ppi.add_argument("name", help="reference plugin name")
    ppi.set_defaults(func=_cmd_plugin_info)

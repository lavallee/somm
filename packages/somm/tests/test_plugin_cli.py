from __future__ import annotations

import pytest
from somm import hooks
from somm.cli import build_parser, main
from somm.plugins import REFERENCE_PLUGINS
from somm.providers.registry import BUILTIN_PROVIDER_SPECS


@pytest.fixture(autouse=True)
def _reset_hooks():
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    saved_hooks = {
        phase: list(hooks._hooks_by_phase[phase]) for phase in hooks.HOOK_PHASES
    }
    saved_index = hooks._next_insertion_index
    saved_entry_points_loaded = hooks._entry_points_loaded
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase].clear()
    hooks._next_insertion_index = 0
    hooks._entry_points_loaded = False
    yield
    hooks.shutdown_hooks(wait=True)
    hooks.set_correlation_provider(None)
    for phase in hooks.HOOK_PHASES:
        hooks._hooks_by_phase[phase][:] = saved_hooks[phase]
    hooks._next_insertion_index = saved_index
    hooks._entry_points_loaded = saved_entry_points_loaded


def test_plugin_list_includes_reference_plugins_and_builtin_providers(capsys):
    rc = main(["plugin", "list"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "REFERENCE PLUGINS" in out
    assert "ACTIVE HOOKS" in out
    assert "PROVIDERS" in out
    for name in REFERENCE_PLUGINS:
        assert name in out
    for spec in BUILTIN_PROVIDER_SPECS:
        assert spec.name in out


def test_plugin_list_shows_registered_hook(capsys):
    def sample_hook(_event):
        return None

    hooks.register_hook(hooks.POST_CALL, sample_hook, priority=42)

    rc = main(["plugin", "list"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "post_call" in out
    assert "sample_hook" in out
    assert "42" in out


def test_plugin_info_cache_shows_summary_and_signature(capsys):
    rc = main(["plugin", "info", "cache"])
    out = capsys.readouterr().out

    assert rc == 0
    assert REFERENCE_PLUGINS["cache"]["summary"] in out
    assert "register(" in out
    assert "ttl_s" in out
    assert "maxsize" in out


def test_plugin_info_unknown_errors_nonzero(capsys):
    rc = main(["plugin", "info", "nope"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "Unknown reference plugin 'nope'" in captured.err
    assert "cache" in captured.err


def test_plugin_commands_parse():
    parser = build_parser()
    parser.parse_args(["plugin", "list"])
    parser.parse_args(["plugin", "info", "cache"])

"""Helpers shared by more than one command module."""

from __future__ import annotations

from somm_core.config import load as _load_config


def load_config(project: str | None = None, **kwargs):
    """The CLI's config: ``--project X`` run outside X's checkout reads the
    ledger registered for X in ``~/.somm/registry.json`` instead of silently
    reading this directory's ledger and reporting nothing."""
    return _load_config(project=project, prefer_registry=True, **kwargs)



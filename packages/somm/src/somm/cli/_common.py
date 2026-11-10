"""Helpers shared by more than one command module."""

from __future__ import annotations

import argparse
import os

from somm_core.config import load as _load_config


def load_config(project: str | None = None, **kwargs):
    """The CLI's config: ``--project X`` run outside X's checkout reads the
    ledger registered for X in ``~/.somm/registry.json`` instead of silently
    reading this directory's ledger and reporting nothing.

    ``SOMM_PROJECT`` counts as ``--project`` here, so it reaches the registry
    the same way."""
    project = project or os.environ.get("SOMM_PROJECT") or None
    return _load_config(project=project, prefer_registry=True, **kwargs)


def add_project(parser: argparse.ArgumentParser, help: str | None = None) -> None:
    """Add ``--project`` to ``parser`` so it is accepted at any position.

    The default is ``SUPPRESS`` so a value given at one level is never
    overwritten by an absent flag at another (``somm --project x inbox list``,
    ``somm inbox --project x list`` and ``somm inbox list --project x`` all
    read the same). The root parser supplies the ``None`` default.
    """
    parser.add_argument("--project", default=argparse.SUPPRESS, help=help)

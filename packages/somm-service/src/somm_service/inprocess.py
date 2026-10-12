"""Compatibility shim — see ``somm.workers.inprocess``."""

from somm.workers.inprocess import build_workers_factory, start_inprocess_scheduler

__all__ = ["build_workers_factory", "start_inprocess_scheduler"]

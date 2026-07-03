"""In-process intelligence loop — the scheduler without the web server.

Deliberately imports only the workers (httpx-weight), NOT the starlette
web stack: library-only deployments enable the loop with
SOMM_INPROCESS_WORKERS=1 and shouldn't need web-server dependencies
importable to do it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.config import Config
    from somm_core.repository import Repository


def build_workers_factory(cfg: Config, repo: Repository):
    """Create a factory that returns a worker instance for a given job name."""
    from somm.client import build_default_providers

    from somm_service.workers import (
        AgentWorker,
        ModelIntelWorker,
        ShadowEvalWorker,
    )

    def factory(job_name: str):
        if job_name == "model_intel":
            return ModelIntelWorker(repo, ollama_url=cfg.ollama_url)
        if job_name == "shadow_eval":
            # A workload's gold provider is an explicit choice — it must not
            # be filtered by SOMM_PROVIDER_ORDER (routing preference) or the
            # pinned-only exclusions (CLI executors). full=True reaches
            # anything configured.
            return ShadowEvalWorker(repo, providers=build_default_providers(cfg, full=True))
        if job_name == "agent":
            return AgentWorker(repo)
        return None

    return factory


def start_inprocess_scheduler(cfg: Config, repo: Repository):
    """Start the background scheduler inside the current process.

    This is what `somm serve` runs, minus the web server — for library-only
    deployments that still want the intelligence loop (model-intel refresh,
    online-eval grading, recommendations) without a dedicated service.
    Enabled from the library via SOMM_INPROCESS_WORKERS=1. Returns the
    running Scheduler; caller owns stop().
    """
    from somm_service.workers import Scheduler

    scheduler = Scheduler(repo, build_workers_factory(cfg, repo))
    scheduler.start()
    return scheduler

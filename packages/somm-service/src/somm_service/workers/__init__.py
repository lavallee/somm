"""Scheduled workers: model_intel (v0.3a), shadow_eval (v0.3b), agent (v0.3c).

All workers are callable classes with `.run_once()`. A lightweight scheduler
in `_runner.py` polls the `jobs` table and invokes due workers atomically
(lease-based, crash-safe).
"""

from somm_service.workers._runner import DEFAULT_JOBS, Scheduler
from somm_service.workers.agent import AgentWorker, Recommendation
from somm_service.workers.hf_intel import HuggingFaceIntelWorker
from somm_service.workers.model_intel import ModelIntelWorker
from somm_service.workers.shadow_eval import ShadowConfig, ShadowEvalWorker

__all__ = [
    "ModelIntelWorker",
    "HuggingFaceIntelWorker",
    "ShadowEvalWorker",
    "ShadowConfig",
    "AgentWorker",
    "Recommendation",
    "Scheduler",
    "DEFAULT_JOBS",
]

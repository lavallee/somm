"""Compatibility shim — the intelligence-loop workers live in ``somm.workers``.

They moved in 0.17 so `somm loop run` and `SOMM_INPROCESS_WORKERS=1` work with
the library alone; the web service only adds the dashboard, the HTTP API, and
the long-running scheduler host. Existing imports of
``somm_service.workers.<module>`` keep resolving to the same module objects.
"""

from __future__ import annotations

import sys

from somm import workers as _workers
from somm.workers import (
    DEFAULT_JOBS,
    AgentWorker,
    HuggingFaceIntelWorker,
    ModelIntelWorker,
    Recommendation,
    Scheduler,
    ShadowConfig,
    ShadowEvalWorker,
    _heartbeat,
    _runner,
    agent,
    hf_intel,
    model_intel,
    shadow_eval,
)

for _name, _module in (
    ("_heartbeat", _heartbeat),
    ("_runner", _runner),
    ("agent", agent),
    ("hf_intel", hf_intel),
    ("model_intel", model_intel),
    ("shadow_eval", shadow_eval),
):
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [
    "ModelIntelWorker",
    "HuggingFaceIntelWorker",
    "ShadowEvalWorker",
    "ShadowConfig",
    "AgentWorker",
    "Recommendation",
    "Scheduler",
    "DEFAULT_JOBS",
    "_workers",
]

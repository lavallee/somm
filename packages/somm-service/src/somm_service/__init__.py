"""somm-service — web admin + HTTP API + background workers.

v0.1 ships the minimal web dashboard + /api/stats. Workers (model_intel,
shadow_eval, agent) come in D3+.
"""

from somm_service.app import create_app, run_server

__all__ = ["create_app", "run_server"]

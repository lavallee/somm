"""somm-core — shared schema, repository, config, parse, version across all somm packages."""

from __future__ import annotations

import importlib

from somm_core.version import SCHEMA_VERSION, VERSION

_EXPORTS: dict[str, str] = {
    # models
    "Call": "somm_core.models",
    "CallOutcome": "somm_core.models",
    "Campaign": "somm_core.models",
    "CampaignEvent": "somm_core.models",
    "Dataset": "somm_core.models",
    "DatasetItem": "somm_core.models",
    "Decision": "somm_core.models",
    "EmbedResult": "somm_core.models",
    "EvalReceipt": "somm_core.models",
    "FailureClass": "somm_core.models",
    "ModelAlias": "somm_core.models",
    "Outcome": "somm_core.models",
    "PrivacyClass": "somm_core.models",
    "Prompt": "somm_core.models",
    "SommResult": "somm_core.models",
    "ToolCall": "somm_core.models",
    "Workload": "somm_core.models",
    # graders
    "GradeScores": "somm_core.graders",
    "grade_response_pair": "somm_core.graders",
    "judge_score": "somm_core.graders",
    "json_overlap": "somm_core.graders",
    "structural_score": "somm_core.graders",
    "text_similarity": "somm_core.graders",
    # pricing
    "cost_for_call": "somm_core.pricing",
    "list_intel": "somm_core.pricing",
    "merge_intel_capabilities": "somm_core.pricing",
    "seed_known_pricing": "somm_core.pricing",
    "sync_bundled_pricing": "somm_core.pricing",
    "write_intel": "somm_core.pricing",
    # plans
    "BurnRate": "somm_core.plans",
    "CatalogEntry": "somm_core.plans",
    "LearnedLimitUpdate": "somm_core.plans",
    "LimitStatus": "somm_core.plans",
    "ObservedCeiling": "somm_core.plans",
    "Plan": "somm_core.plans",
    "PlanLimit": "somm_core.plans",
    "learn_observed_limits": "somm_core.plans",
    "limit_statuses": "somm_core.plans",
    "load_catalog": "somm_core.plans",
    "load_plans": "somm_core.plans",
    "observed_ceilings": "somm_core.plans",
    "payg_burn_rates": "somm_core.plans",
    "plan_for": "somm_core.plans",
    "recent_ok_calls": "somm_core.plans",
    # registry/repository/schema
    "Repository": "somm_core.repository",
    "current_schema_version": "somm_core.schema",
    "ensure_schema": "somm_core.schema",
    "fleet_db_paths": "somm_core.registry",
    "register_project": "somm_core.registry",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'somm_core' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "VERSION",
    "SCHEMA_VERSION",
    *_EXPORTS,
]

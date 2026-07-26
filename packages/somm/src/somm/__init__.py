"""somm — self-hosted LLM telemetry, routing, and intelligence loop."""

import importlib

from somm.errors import (
    SommBadRequest,
    SommBudgetExceeded,
    SommError,
    SommNoCapableProvider,
    SommPrivacyViolation,
    SommProvidersExhausted,
    SommStrictMode,
    describe_error,
)

_CORE_MODEL_EXPORTS = {"EmbedResult", "Outcome", "PrivacyClass", "SommResult"}
_OUTCOME_EVIDENCE_EXPORTS = {
    "OutcomeEvidenceAssessment",
    "OutcomeEvidenceState",
    "assess_outcome_snapshot",
}


def llm(*args, **kwargs):
    from somm.client import llm as _llm

    return _llm(*args, **kwargs)


def extract_json(*args, **kwargs):
    from somm_core.parse import extract_json as _extract_json

    return _extract_json(*args, **kwargs)


def __getattr__(name: str):
    if name == "SommLLM":
        from somm.client import SommLLM

        globals()[name] = SommLLM
        return SommLLM
    if name in _CORE_MODEL_EXPORTS:
        module = importlib.import_module("somm_core.models")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _OUTCOME_EVIDENCE_EXPORTS:
        module = importlib.import_module("somm.outcome_evidence")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "record_procedure_outcome":
        from somm.procedure_outcomes import record_procedure_outcome

        globals()[name] = record_procedure_outcome
        return record_procedure_outcome
    if name == "hooks":
        return importlib.import_module("somm.hooks")
    if name == "workloads":
        return importlib.import_module("somm.workloads")
    if name == "harnesses":
        return importlib.import_module("somm.harnesses")
    if name == "provenance":
        from somm.provenance import provenance

        globals()[name] = provenance
        return provenance
    raise AttributeError(f"module 'somm' has no attribute {name!r}")


__all__ = [
    "SommLLM",
    "llm",
    "hooks",
    "workloads",
    "harnesses",
    "provenance",
    "extract_json",
    "EmbedResult",
    "Outcome",
    "PrivacyClass",
    "SommResult",
    "SommError",
    "SommBadRequest",
    "SommBudgetExceeded",
    "SommNoCapableProvider",
    "SommPrivacyViolation",
    "SommProvidersExhausted",
    "SommStrictMode",
    "describe_error",
    "OutcomeEvidenceAssessment",
    "OutcomeEvidenceState",
    "assess_outcome_snapshot",
    "record_procedure_outcome",
]

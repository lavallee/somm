"""somm — self-hosted LLM telemetry, routing, and intelligence loop."""

from somm_core import EmbedResult, Outcome, PrivacyClass, SommResult
from somm_core.parse import extract_json

from somm import hooks
from somm.client import SommLLM, llm
from somm.errors import (
    SommBadRequest,
    SommBudgetExceeded,
    SommError,
    SommNoCapableProvider,
    SommPrivacyViolation,
    SommProvidersExhausted,
    SommStrictMode,
)
from somm.provenance import provenance

__all__ = [
    "SommLLM",
    "llm",
    "hooks",
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
]

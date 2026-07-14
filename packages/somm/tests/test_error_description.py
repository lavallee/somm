"""Stable error metadata for task runners and service adapters."""

from somm.errors import (
    SommAuthError,
    SommBudgetExceeded,
    SommProvidersExhausted,
    SommRateLimited,
    describe_error,
)


def test_describe_rate_limit() -> None:
    assert describe_error(SommRateLimited("slow down", retry_after_s=17)) == {
        "schema_version": 1,
        "code": "SOMM_PROVIDER_RATE_LIMIT",
        "message": "slow down",
        "retryable": True,
        "retry_after_s": 17.0,
    }


def test_describe_exhaustion_uses_next_cooldown() -> None:
    result = describe_error(SommProvidersExhausted("all cooled", next_cool_in_s=4.5))
    assert result["retryable"] is True
    assert result["retry_after_s"] == 4.5


def test_describe_budget_fields() -> None:
    result = describe_error(
        SommBudgetExceeded("daily cap", workload="writer", spent_usd=2.0, cap_usd=2.0)
    )
    assert result == {
        "schema_version": 1,
        "code": "SOMM_BUDGET_EXCEEDED",
        "message": "daily cap",
        "retryable": False,
        "workload": "writer",
        "spent_usd": 2.0,
        "cap_usd": 2.0,
    }


def test_describe_fatal_error() -> None:
    result = describe_error(SommAuthError("bad key"))
    assert result["code"] == "SOMM_PROVIDER_AUTH"
    assert result["retryable"] is False

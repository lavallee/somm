"""FailureClass classification of Outcome values.

The split between capability_* (model unfit) and detractor_* (provider/network
flaky) is the load-bearing distinction for adequacy queries — verify every
Outcome value lands in the bucket steve's reporter's-notebook framework
expects.
"""

from __future__ import annotations

import pytest
from somm_core.models import FailureClass, Outcome


@pytest.mark.parametrize(
    ("outcome", "expected_class"),
    [
        (Outcome.OK, FailureClass.NONE),
        (Outcome.BAD_JSON, FailureClass.CAPABILITY_PAYLOAD),
        (Outcome.OFF_TASK, FailureClass.CAPABILITY_PAYLOAD),
        (Outcome.EMPTY, FailureClass.CAPABILITY_EMPTY),
        (Outcome.TIMEOUT, FailureClass.DETRACTOR_TIMEOUT),
        (Outcome.RATE_LIMIT, FailureClass.DETRACTOR_RATE_LIMIT),
        (Outcome.UPSTREAM_ERROR, FailureClass.DETRACTOR_UPSTREAM),
        (Outcome.EXHAUSTED, FailureClass.META_EXHAUSTED),
        (Outcome.UNKNOWN, FailureClass.UNKNOWN),
    ],
)
def test_outcome_failure_class_mapping(outcome: Outcome, expected_class: FailureClass) -> None:
    assert outcome.failure_class is expected_class


@pytest.mark.parametrize(
    "outcome",
    [Outcome.BAD_JSON, Outcome.OFF_TASK, Outcome.EMPTY],
)
def test_capability_signals_are_capability(outcome: Outcome) -> None:
    assert outcome.is_capability_signal is True
    assert outcome.is_detractor is False


@pytest.mark.parametrize(
    "outcome",
    [Outcome.TIMEOUT, Outcome.RATE_LIMIT, Outcome.UPSTREAM_ERROR],
)
def test_detractors_are_not_capability(outcome: Outcome) -> None:
    assert outcome.is_capability_signal is False
    assert outcome.is_detractor is True


@pytest.mark.parametrize(
    "outcome",
    [Outcome.OK, Outcome.EXHAUSTED, Outcome.UNKNOWN],
)
def test_neither_bucket(outcome: Outcome) -> None:
    assert outcome.is_capability_signal is False
    assert outcome.is_detractor is False


def test_failure_class_predicates_self_consistent() -> None:
    """The class-level predicates must agree with the outcome-level ones."""
    for outcome in Outcome:
        fc = outcome.failure_class
        assert fc.is_capability_signal == outcome.is_capability_signal
        assert fc.is_detractor == outcome.is_detractor

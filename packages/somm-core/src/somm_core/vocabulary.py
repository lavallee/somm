"""The enumerations a downstream tool maps somm's values against.

A tool that translates somm rows into another vocabulary (a contract mapper, a
dashboard colour table) can check its mapping against :func:`vocabulary` when
it starts, instead of failing on the first row that carries a value it has not
seen. Plain data; built from the enums where they exist.

``vocabulary_version`` changes only when a key is removed or a meaning
changes. Adding a value or a key does not bump it. Sets marked ``open`` are
not closed: somm writes values its own code does not enumerate (an exception
class name, a provider name), so map the listed values and have a default.
"""

from __future__ import annotations

from somm_core.boundary import REFUSAL_ERROR_KIND, TIER_NAMES
from somm_core.cost import UNKNOWN_COST_SOURCE
from somm_core.models import Outcome
from somm_core.plans import VALID_MODES, VALID_UNITS
from somm_core.read import ROUTE_SOURCES, VERDICT_BY_FIELD, VERDICT_FIELD
from somm_core.version import SCHEMA_VERSION, VERSION

VOCABULARY_VERSION = 1


def vocabulary() -> dict:
    """The stable value sets, with the somm and schema versions that wrote them."""
    return {
        "vocabulary_version": VOCABULARY_VERSION,
        "somm_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        # calls.outcome: closed. failure_class says whose fault a non-ok call is.
        "outcomes": {
            o.value: {
                "failure_class": o.failure_class.value,
                "capability_signal": o.is_capability_signal,
                "detractor": o.is_detractor,
            }
            for o in Outcome
        },
        # calls.error_kind: open (an exception class name, or an outcome value
        # for a harness attempt). These are the names somm itself writes.
        "error_kinds": {
            "open": True,
            "known": sorted(
                {"EmptyResponse", "SommStructuredError", REFUSAL_ERROR_KIND, "SommProvidersExhausted"}
            ),
        },
        # calls.kind: closed. NULL is an ordinary call (generate, stream,
        # structured, embed); "decide" is a decision-model call (0030) whose
        # label, p_top, abstained and result_json columns are set.
        "call_kinds": ["decide"],
        # call_updates.field: open (callers append their own). somm writes
        # "outcome" (result.mark), "verdict" (the right label of a decide
        # row), "verdict:<question>" (of one question on a decide_many row)
        # and "verdict_by" (the grader call that wrote a row's verdicts).
        "call_update_fields": {
            "open": True,
            "known": ["outcome", VERDICT_FIELD, VERDICT_BY_FIELD],
            "patterns": [VERDICT_FIELD + ":<question>"],
        },
        # calls.data_tier, strictest first: closed.
        "data_tiers": list(TIER_NAMES),
        # routing_json.route_source -> True when somm's evidence chose the route.
        "route_sources": dict(ROUTE_SOURCES),
        # calls.cost_source: open. A value starting with ``unknown_prefix`` means
        # the price was not known (cost_usd 0.0 is then "unknown", not "free").
        "cost_sources": {
            "open": True,
            "known": sorted(
                [
                    "somm:model_intel",
                    UNKNOWN_COST_SOURCE,
                    "local-included-unpriced",
                    "openrouter:free-model",
                    "chatgpt-subscription",
                    "otlp",
                ]
            ),
            "patterns": ["provider:<name>", "<provider>:response"],
            "unknown_prefix": "unknown:",
        },
        # The cost provenance columns: closed.
        "cost_basis": ["computed", "reported", "unknown"],
        "cost_kind": ["marginal", "notional", "included", "unknown"],
        "cost_accuracy": ["estimated", "unknown"],
        "observation_roles": ["production", "shadow_gold", "shadow_judge", "imported"],
        "origins": ["native", "harness", "foreign_imported"],
        # Units: what the call row itself counts, and what a plan limit may meter.
        "meter_units": {
            "call": ["tokens_in", "tokens_out"],
            "plan_limit": list(VALID_UNITS),
        },
        "plan_modes": list(VALID_MODES),
    }

"""Decisions: a call that returns a typed label and a probability, not text.

``SommLLM.decide`` delegates here. The question is plain Python:

- ``ask`` alone: a yes/no predicate. Labels are ``"yes"`` and ``"no"``.
- ``ask`` + ``labels``: a choice among named labels (a list of names, or a
  dict of name -> description of when it applies).
- ``ask`` + ``scale``: a score on a stated scale, an ordered list of level
  descriptions, lowest first. The label is the most probable level.

One provider serves it (TypeSafe System One, ``somm.providers.typesafe``),
pinned. Every call writes one ``calls`` row with ``kind = 'decide'`` plus the
chosen ``label``, its probability ``p_top``, ``abstained``, and the full
answer in ``result_json``. Cost is input tokens at the bundled price.
See docs/decisions.md.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from somm_core import Outcome, cost_for_call
from somm_core.cost import cost_provenance
from somm_core.models import Call, MarkHook
from somm_core.parse import stable_hash
from somm_core.read import VERDICT_FIELD
from somm_core.version import VERSION

from somm import hooks
from somm.errors import (
    SommError,
    SommPrivacyViolation,
    SommProviderNotConfigured,
    SommTransientError,
)
from somm.providers.typesafe import TypeSafeProvider

if TYPE_CHECKING:
    from somm.client import SommLLM

KIND = "decide"
YES, NO = "yes", "no"


@dataclass(slots=True)
class DecisionResult:
    """What ``SommLLM.decide`` returns. A plain value, like ``SommResult``.

    ``label`` is the chosen label and ``p`` the probability the model gave
    it; ``probabilities`` covers every label. ``abstained`` is True when ``p``
    fell below the caller's ``min_p``; the label is still reported.
    ``confidence`` is the vendor's concentration statistic (choice and score
    only); ``score`` is the expected level index (score only).
    """

    label: str
    p: float
    probabilities: dict[str, float]
    abstained: bool
    shape: str  # "yes_no" | "choice" | "score"
    provider: str
    model: str
    call_id: str
    latency_ms: int
    tokens_in: int
    cost_usd: float
    confidence: float | None = None
    score: float | None = None
    outcome: Outcome = Outcome.OK
    _on_verdict: MarkHook | None = field(default=None, repr=False, compare=False)

    def verdict(self, truth: str) -> DecisionResult:
        """Record which label was actually right, for calibration.

        Appends a ``call_updates`` row (field ``verdict``). Never raises:
        recording a verdict must not break the caller. Returns self.
        """
        if self._on_verdict is not None:
            try:
                self._on_verdict((self.call_id, str(truth)))
            except Exception:  # noqa: BLE001
                pass
        return self


def _question(ask: str, labels, scale) -> tuple[str, dict, list[str]]:
    """``(shape, wire question, label names)`` from the caller's arguments."""
    if not isinstance(ask, str) or not ask.strip():
        raise ValueError("decide() needs `ask`: the question, as text")
    if labels is not None and scale is not None:
        raise ValueError("decide() takes `labels` (a choice) or `scale` (a score), not both")
    if labels is not None:
        criteria = dict(labels) if isinstance(labels, dict) else {str(n): None for n in labels}
        if len(criteria) < 2 or not all(isinstance(k, str) and k for k in criteria):
            raise ValueError("`labels` needs at least two distinct, non-empty names")
        return "choice", {"type": "choice", "instructions": ask, "criteria": criteria}, list(criteria)
    if scale is not None:
        levels = [str(s) for s in scale]
        if not 2 <= len(levels) <= 10 or len(set(levels)) != len(levels):
            raise ValueError("`scale` needs 2 to 10 distinct level descriptions, lowest first")
        return "score", {"type": "score", "instructions": ask, "criteria": levels}, levels
    return "yes_no", {"type": "noul", "instructions": ask}, [YES, NO]


def _read_answer(shape: str, answer: dict, names: list[str]) -> dict:
    """The typed answer as ``label``, ``p``, ``probabilities`` and extras."""
    if shape == "yes_no":
        p_yes = min(max(float(answer["noul"]), 0.0), 1.0)
        probs = {YES: p_yes, NO: 1.0 - p_yes}
        confidence = score = None
    else:
        raw = answer["probabilities"]
        if shape == "score":
            # Keyed by level index ("0", "1", ...); report the caller's words.
            probs = {names[int(k)]: float(v) for k, v in raw.items()}
            score = float(answer["score"])
        else:
            probs = {str(k): float(v) for k, v in raw.items()}
            score = None
        confidence = answer.get("confidence")
        confidence = float(confidence) if confidence is not None else None
    if shape == "choice" and answer.get("choice") in probs:
        label = answer["choice"]
    else:
        label = max(probs, key=probs.__getitem__)
    return {"label": label, "p": probs[label], "probabilities": probs,
            "confidence": confidence, "score": score}


def _provider(llm: SommLLM) -> TypeSafeProvider:
    injected = getattr(llm, "_decision_provider", None)
    if injected is not None:
        return injected
    key = getattr(llm.config, "typesafe_api_key", None)
    if not key:
        raise SommProviderNotConfigured(
            "decide() needs a TypeSafe key: set TYPESAFE_API_KEY", provider="typesafe"
        )
    llm._decision_provider = TypeSafeProvider(api_key=key)
    return llm._decision_provider


def decide(
    llm: SommLLM,
    text: Any,
    *,
    ask: str,
    labels: list[str] | dict[str, str] | None = None,
    scale: list[str] | None = None,
    workload: str = "default",
    context: Any = None,
    min_p: float | None = None,
    model: str | None = None,
    session_id: str | None = None,
    parent_call_id: str | None = None,
) -> DecisionResult:
    """See ``SommLLM.decide``."""
    from somm.boundary import check_route
    from somm.client import _failure_outcome, _format_error_detail, _route_record

    shape, question, names = _question(ask, labels, scale)
    if min_p is not None and not 0.0 <= float(min_p) <= 1.0:
        raise ValueError("min_p must be between 0 and 1")
    state = text if context is None else {"text": text, "context": context}
    prompt_hash = stable_hash(json.dumps({"state": state, "question": question}, sort_keys=True,
                                         default=str))
    wl = llm._require_workload(workload)
    provider = _provider(llm)
    want = model or provider.default_model

    # Admission: the data boundary (a refusal is recorded, then raised), then
    # the daily budget. Same gates as generate(), one route.
    boundary = llm._resolve_boundary(wl, None)
    try:
        route = check_route(provider=provider.name, model=want, boundary=boundary,
                            workload=workload, context="decide()")
    except SommPrivacyViolation as exc:
        llm._record_refusal(wl, workload, exc, model=want, prompt_hash=prompt_hash,
                            session_id=session_id, parent_call_id=parent_call_id, mode="pinned")
        raise
    llm._enforce_budget(wl)

    call_id = str(uuid.uuid4())
    ts = datetime.now(UTC)
    error: Exception | None = None
    resp = None
    try:
        resp = provider.decide(state, question, model=want)
        try:
            typed = _read_answer(shape, resp.answer, names)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise SommTransientError(
                f"typesafe answer did not match the question: {exc}", model=resp.model
            ) from exc
    except Exception as exc:  # recorded below, then raised
        error = exc

    served = resp.model if resp is not None else want
    tokens_in = resp.tokens_in if resp is not None else 0
    cost_usd = cost_for_call(llm.repo, provider.name, served, tokens_in, 0)
    basis, cost_kind, accuracy, source, pricing_version = cost_provenance(
        llm.repo, provider.name, served, reported=False, source=None
    )
    call = Call(
        id=call_id,
        ts=ts,
        project=llm.config.project,
        workload_id=wl.id,
        prompt_id=None,
        provider=provider.name,
        model=served,
        tokens_in=tokens_in,
        tokens_out=resp.tokens_out if resp is not None else 0,
        latency_ms=resp.latency_ms if resp is not None else 0,
        cost_usd=cost_usd,
        outcome=Outcome.OK if error is None else _failure_outcome(error),
        error_kind=None if error is None else type(error).__name__,
        error_detail=None if error is None else _format_error_detail(error, provider.name, served),
        prompt_hash=prompt_hash,
        response_hash=stable_hash(json.dumps(resp.answer, sort_keys=True) if resp else ""),
        correlation_id=hooks.current_correlation_id(),
        call_site=hooks.current_call_site(),
        client_version=VERSION,
        session_id=session_id,
        parent_call_id=parent_call_id,
        cost_basis=basis,
        cost_kind=cost_kind,
        cost_accuracy=accuracy,
        cost_source=source,
        pricing_version=pricing_version,
        routing_json=json.dumps({
            "route_source": "pin" if model else "default",
            "route": _route_record("pinned", [provider.name, want],
                                   None if error else [provider.name, served], [], None),
        }),
        data_tier=route.tier.value,
        kind=KIND,
    )
    if error is None:
        abstained = min_p is not None and typed["p"] < float(min_p)
        call.label, call.p_top, call.abstained = typed["label"], typed["p"], abstained
        call.result_json = json.dumps({"shape": shape, "min_p": min_p, **typed}, sort_keys=True)
    llm._finish_call(call, workload=workload, short_circuited=None)
    if error is not None:
        if isinstance(error, SommError):
            error.provider, error.model, error.call_id = provider.name, served, call_id
        raise error

    return DecisionResult(
        label=typed["label"],
        p=typed["p"],
        probabilities=typed["probabilities"],
        abstained=abstained,
        shape=shape,
        provider=provider.name,
        model=served,
        call_id=call_id,
        latency_ms=call.latency_ms,
        tokens_in=tokens_in,
        cost_usd=cost_usd,
        confidence=typed["confidence"],
        score=typed["score"],
        _on_verdict=MarkHook(lambda pair: record_verdict(llm, *pair)),
    )


def record_verdict(llm: SommLLM, call_id: str, truth: str) -> None:
    """Record that ``truth`` was the right label for decision ``call_id``.

    For verdicts that arrive later than the result object (a reviewer, a
    downstream outcome). The latest verdict for a call wins in
    ``somm_core.read.decision_calibration``.
    """
    llm._writer.flush()  # the row may still be queued
    llm.repo.record_call_update(call_id, field=VERDICT_FIELD, value=str(truth))
    if llm._mirror_repo is not None:
        try:
            llm._mirror_repo.record_call_update(call_id, field=VERDICT_FIELD, value=str(truth))
        except Exception:  # noqa: BLE001 — the mirror is a convenience copy
            pass

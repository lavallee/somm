"""Provider/routing error taxonomy.

Transient errors cool the offending (provider, model) and let the router try
the next one. Fatal errors bubble up — the caller must handle them. Every
exception carries a canonical SOMM_* code so error messages stay consistent
(canonical operator-facing error format).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any


class SommError(Exception):
    """Base for all somm exceptions.

    ``provider``, ``model`` and ``call_id`` say which route failed and which
    ledger row recorded it, when somm knows; they are None otherwise.

    When the error came from a provider's HTTP answer, the upstream detail
    rides along, each None when the provider did not say:

    - ``status_code``: the exact upstream HTTP status (a 403 stays 403).
    - ``provider_code``, ``provider_type``, ``provider_param``: the ``code``,
      ``type`` and ``param`` of the provider's error object, as strings.
    - ``provider_request_id``: the provider's request id, for its support.
    - ``retry_after_s``: seconds the provider asked us to wait
      (``Retry-After`` or its rate-limit reset headers).

    None of these carries the provider's error *message*, which can echo the
    prompt; that stays in the (truncated) exception text.
    """

    code: str = "SOMM_ERROR"
    provider: str | None = None
    model: str | None = None
    call_id: str | None = None
    status_code: int | None = None
    provider_code: str | None = None
    provider_type: str | None = None
    provider_param: str | None = None
    provider_request_id: str | None = None
    retry_after_s: float | None = None


class SommProviderError(SommError):
    """Raised by a provider adapter. Router decides whether to fall through."""


class SommTransientError(SommProviderError):
    """Transient — router should cool (provider, model) and try next."""

    def __init__(self, detail: str = "", cooldown_s: float = 60.0,
                 model: str = "") -> None:
        super().__init__(detail)
        self.cooldown_s = cooldown_s
        self.model = model  # which model failed; "" if provider-level


class SommTimeout(SommTransientError):
    code = "SOMM_PROVIDER_TIMEOUT"


class SommRateLimited(SommTransientError):
    code = "SOMM_PROVIDER_RATE_LIMIT"

    def __init__(self, detail: str = "", retry_after_s: float = 120.0, model: str = "") -> None:
        super().__init__(detail, cooldown_s=retry_after_s, model=model)
        self.retry_after_s = retry_after_s


class SommUpstream5xx(SommTransientError):
    code = "SOMM_PROVIDER_5XX"


class SommEmptyResponse(SommTransientError):
    """Model returned no usable content. Treat as transient so routing tries next."""

    code = "SOMM_EMPTY_RESPONSE"


class SommInsufficientCredit(SommTransientError):
    """Provider rejected the call because its balance/quota is exhausted.

    Providers surface this as an auth-ish or bad-request-ish status (Anthropic
    400 "credit balance is too low", OpenAI 429 ``insufficient_quota``), but
    it's neither a malformed request nor a bad key — it's a *billing state* on
    that one provider. The request itself is fine and would succeed elsewhere,
    so this is transient: the router cools the lapsed provider and falls through
    to the next. The cooldown is long because a balance won't replenish in
    seconds; we'd rather sideline the provider than hammer it.
    """

    code = "SOMM_PROVIDER_INSUFFICIENT_CREDIT"

    def __init__(self, detail: str = "", cooldown_s: float = 3600.0,
                 model: str = "") -> None:
        super().__init__(detail, cooldown_s=cooldown_s, model=model)


class SommFatalError(SommProviderError):
    """Fatal — caller must fix. Don't retry, don't fall through."""


class SommAuthError(SommFatalError):
    code = "SOMM_PROVIDER_AUTH"


class SommBadRequest(SommFatalError):
    code = "SOMM_PROVIDER_BAD_REQUEST"


MODEL_NOT_FOUND_COOLDOWN_S = 3600.0
"""Base cooldown for a (provider, model) slot the provider says it does not
serve. A withdrawn model does not come back in seconds; the router's
escalation ladder lengthens this for a slot that keeps failing."""


class SommModelNotFound(SommBadRequest):
    """The provider does not serve this model (404, or a 400 naming the model).

    Still a ``SommBadRequest``: a pinned call, the proxy and anyone catching
    ``SommBadRequest`` see what they always saw. The router and OpenRouter's
    roster walk treat it as a dead *slot*, not a bad request: they cool that
    (provider, model) on the escalating ladder and try the next candidate. A
    withdrawn model used to be fatal and never cooled, so it was retried on
    every call and aborted the chain each time.
    """

    def __init__(self, detail: str = "", *, model: str = "",
                 cooldown_s: float = MODEL_NOT_FOUND_COOLDOWN_S) -> None:
        super().__init__(detail)
        self.model = model
        self.cooldown_s = cooldown_s


# Wording that means "no such model" in a 400 body. Narrow on purpose, like
# the billing phrases: a false positive would route past a malformed request.
_MODEL_NOT_FOUND_SIGNALS = (
    "model_not_found",
    "is not a valid model id",
    "no endpoints found",
)


def _looks_like_model_not_found(body: str) -> bool:
    low = body.lower()
    return any(sig in low for sig in _MODEL_NOT_FOUND_SIGNALS)


class SommProviderNotConfigured(SommError, ValueError):
    """A call pinned a provider this machine has not configured.

    Raised before any network call. Still a ``ValueError``, which is what
    this case raised before it had a name, so existing handlers keep working.
    """

    code = "SOMM_PROVIDER_NOT_CONFIGURED"

    def __init__(self, detail: str = "", *, provider: str | None = None) -> None:
        super().__init__(detail)
        self.provider = provider


class SommRefused(SommProviderError):
    """The model or the provider's filter declined to answer.

    Seen as a ``stop_reason`` of ``refusal`` (Anthropic) or
    ``content_filter`` (OpenAI-compatible). The call itself was delivered and
    is recorded as such; this is raised only for ``raise_on_failure=True``.
    Retrying the same prompt on the same model rarely helps.

    It carries what a wire format needs to report the refusal as an answer:
    ``stop_reason``, any ``text`` the model did produce, ``tokens_in`` /
    ``tokens_out``, and (from the base) the served ``model``, ``provider``
    and ``call_id``. Each is None when not known.
    """

    code = "SOMM_UPSTREAM_REFUSED"

    def __init__(
        self,
        detail: str = "",
        *,
        stop_reason: str = "",
        text: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        call_id: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.stop_reason = stop_reason
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        if model is not None:
            self.model = model
        if provider is not None:
            self.provider = provider
        if call_id is not None:
            self.call_id = call_id


class SommBudgetExceeded(SommFatalError):
    """Fail-closed budget gate refused the call before dispatch.

    Raised pre-request when a workload's accumulated daily spend has reached
    its configured ``budget_cap_usd_daily`` and ``budget_fail_closed`` is on.
    Deliberately FATAL and non-transient: the router must NOT cool a provider
    and fall through (that would defeat the ceiling). No telemetry row is
    written — a blocked call is not a spend event.
    """

    code = "SOMM_BUDGET_EXCEEDED"

    def __init__(self, detail: str = "", *, workload: str = "",
                 spent_usd: float = 0.0, cap_usd: float = 0.0) -> None:
        super().__init__(detail)
        self.workload = workload
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


class SommProvidersExhausted(SommFatalError):
    """Every configured (provider, model) is either cooled or failed this round."""

    code = "SOMM_PROVIDERS_EXHAUSTED"

    def __init__(self, detail: str = "", next_cool_in_s: float | None = None) -> None:
        super().__init__(detail)
        self.next_cool_in_s = next_cool_in_s


class SommNoCapableProvider(SommFatalError):
    """No provider in the chain exposes a model with the required capabilities.

    Raised before any network call. Carries the missing capabilities and the
    list of (provider, model, reason) pairs the router skipped so operators
    can understand the gap and fix it explicitly (enable a provider, swap
    default model, etc.) rather than discover it via a late 400.
    """

    code = "SOMM_NO_CAPABLE_PROVIDER"

    def __init__(
        self,
        detail: str = "",
        required: list[str] | None = None,
        skipped: list[tuple[str, str, str]] | None = None,
    ) -> None:
        super().__init__(detail)
        self.required = list(required or [])
        self.skipped = list(skipped or [])


class SommStrictMode(SommError):
    """Strict mode refused an unregistered workload/prompt.

    The code matches the operator-facing message and docs page
    (docs/errors/SOMM_WORKLOAD_UNREGISTERED.md) — every raise site is an
    unregistered workload; the class name describes the mode that
    enforces it.
    """

    code = "SOMM_WORKLOAD_UNREGISTERED"


class SommPrivacyViolation(SommError):
    """A request was refused because no route lies within its data boundary.

    Raised before any provider call for a ``privacy_class=private`` workload
    (boundary ``local``) or any workload whose ``data_boundary`` excludes
    every candidate route. Never retried; never falls through to a looser
    route. ``data_boundary`` is the tier that applied, ``boundary_source``
    where it came from, and ``refused`` the routes turned away, each as
    ``{"provider", "model", "tier", "basis", "origin"}``.
    """

    code = "SOMM_PRIVACY_VIOLATION"

    def __init__(
        self,
        detail: str = "",
        *,
        workload: str | None = None,
        data_boundary: str | None = None,
        boundary_source: str | None = None,
        refused: list[dict] | None = None,
    ) -> None:
        super().__init__(detail)
        self.workload = workload
        self.data_boundary = data_boundary
        self.boundary_source = boundary_source
        self.refused = list(refused or [])


class SommStructuredError(SommError):
    """Strict structured-output generation failed after corrective retries.

    ``delivery_failure`` is True when the last attempt never produced model
    output at all (empty response, upstream error, exhaustion) — a provider or
    throttling problem, not a schema problem. Callers should back off rather
    than loosen the schema; ``last_outcome`` says which it was.
    """

    code = "SOMM_STRUCTURED_OUTPUT_FAILED"

    def __init__(
        self,
        detail: str = "",
        *,
        last_raw: str = "",
        last_error: str = "",
        last_outcome: str | None = None,
        delivery_failure: bool = False,
        last_result: Any = None,
    ) -> None:
        super().__init__(detail)
        self.last_raw = last_raw
        self.last_error = last_error
        self.last_outcome = last_outcome
        self.delivery_failure = delivery_failure
        self.last_result = last_result


def describe_error(error: SommError) -> dict[str, Any]:
    """Return a stable, JSON-safe error description for external runners.

    Exception class names are an implementation detail. Task runners and
    service adapters should dispatch on the canonical ``code`` and can use the
    retry metadata here without parsing a traceback or provider message.
    """
    # Exhausted credit is transient for the *router* (another provider can
    # serve the call) but not for a caller: retrying the same provider fails
    # until someone tops it up.
    retryable = isinstance(
        error, (SommTransientError, SommProvidersExhausted)
    ) and not isinstance(error, SommInsufficientCredit)
    description: dict[str, Any] = {
        "schema_version": 1,
        "code": error.code,
        "message": str(error),
        "retryable": retryable,
    }

    retry_after_s = getattr(error, "retry_after_s", None)
    if retry_after_s is None:
        retry_after_s = getattr(error, "next_cool_in_s", None)
    if retry_after_s is None and retryable and isinstance(error, SommTransientError):
        retry_after_s = error.cooldown_s
    if retry_after_s is not None:
        description["retry_after_s"] = float(retry_after_s)

    for field in (
        "provider",
        "model",
        "call_id",
        "status_code",
        "provider_code",
        "provider_type",
        "provider_param",
        "provider_request_id",
        "stop_reason",
        "workload",
        "spent_usd",
        "cap_usd",
        "required",
        "skipped",
        "last_outcome",
        "delivery_failure",
        "data_boundary",
        "boundary_source",
        "refused",
    ):
        value = getattr(error, field, None)
        if value not in (None, "", [], ()):
            description[field] = value
    return description


# Phrases providers use when the account is out of money/quota. Matched
# case-insensitively against the raw error body. Kept deliberately narrow:
# a false positive would silently fall through on a genuinely malformed
# request, so we only match wording that unambiguously means "billing".
_INSUFFICIENT_CREDIT_SIGNALS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient credit",
    "insufficient balance",
    "credit balance is too low",
    "credit balance too low",
    "exceeded your current quota",
    # MiniMax Token Plan, sent as a 429 "rate_limit_error": "Token Plan usage
    # limit reached: Upgrade your Token Plan or purchase Credits for more
    # usage. (2056)". A plan that is used up for the window, not a burst.
    "usage limit reached",
    "purchase credits",
    # OpenAI billing 429s name the cause in ``error.code``
    # (https://developers.openai.com/api/docs/guides/error-codes).
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    # Anthropic: the monthly spend-cap 429 (``error.details.error_code``) and
    # the owner-set spend-limit 400 ("You have reached your specified API
    # usage limits"); https://platform.claude.com/docs/en/api/rate-limits.
    "enforced_spend_limit_reached",
    "api usage limits",
)

# Vendor-specific numeric codes that mean "plan or balance exhausted" even
# when the surrounding wording changes. MiniMax puts the code at the end of
# the message, "(2056)", and in ``base_resp.status_code`` on 200 bodies.
#   2056 — MiniMax: Token Plan usage limit reached (window quota used up).
# MiniMax 1008 ("insufficient balance") is caught by the phrase list.
_PLAN_EXHAUSTED_BODY_CODES: frozenset[str] = frozenset({"2056"})
_PAREN_CODE = re.compile(r"\((\d{3,5})\)")


def _body_codes(body: str) -> set[str]:
    """Numeric vendor codes carried by an error body, as strings."""
    codes: set[str] = set()
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            for key in ("code", "status_code"):
                if err.get(key) not in (None, ""):
                    codes.add(str(err[key]))
            msg = err.get("message")
            if isinstance(msg, str):
                codes.update(_PAREN_CODE.findall(msg))
        base = data.get("base_resp")
        if isinstance(base, dict) and base.get("status_code") not in (None, ""):
            codes.add(str(base["status_code"]))
    else:
        codes.update(_PAREN_CODE.findall(body))
    return codes


def looks_like_insufficient_credit(body: str) -> bool:
    """True if a provider error body signals an exhausted balance/quota.

    Used by provider adapters to reclassify an otherwise-fatal billing 400/429
    as a transient SommInsufficientCredit so the router skips the lapsed
    provider instead of aborting the whole fallback chain. Matches the phrase
    list and the vendor codes in ``_PLAN_EXHAUSTED_BODY_CODES``.
    """
    low = body.lower()
    if any(sig in low for sig in _INSUFFICIENT_CREDIT_SIGNALS):
        return True
    return bool(_body_codes(body) & _PLAN_EXHAUSTED_BODY_CODES)


# ---------------------------------------------------------------------------
# HTTP → taxonomy. One mapping for every HTTP provider.
#
# Each adapter used to carry its own copy of this ladder, and the copies
# disagreed: one knew about exhausted credit, another about HTTP-date
# Retry-After, a third about neither. The router can only treat providers
# uniformly if they classify uniformly, so the ladder lives here and adapters
# call it.


_MAX_WAIT_S = 86400.0
_FIELD_MAX = 256


def _header(headers: Any, name: str) -> str | None:
    """Case-insensitive lookup of a lower-case ``name`` in an ``httpx.Headers``,
    a plain dict, or a test fake; None when absent or unreadable."""
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is None:
            value = next((v for k, v in headers.items() if str(k).lower() == name), None)
    except Exception:  # noqa: BLE001 - a fake that is not a mapping
        return None
    return None if value is None else str(value)


def _seconds_until(raw: str) -> float | None:
    """An RFC 3339 time or an HTTP-date, as seconds from now (never negative)."""
    try:
        then = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            then = parsedate_to_datetime(raw)
        except Exception:  # noqa: BLE001
            return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return max(0.0, (then - datetime.now(UTC)).total_seconds())


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _duration_seconds(raw: str) -> float | None:
    """OpenAI's reset durations: ``1s``, ``6m0s``."""
    raw = raw.strip()
    parts = _DURATION_PART.findall(raw)
    if not parts or "".join(n + u for n, u in parts) != raw:
        return None
    return sum(float(n) * _DURATION_UNIT_S[u] for n, u in parts)


# Rate-limit reset headers, read on a 429 that has no Retry-After. Anthropic
# sends RFC 3339 times (https://platform.claude.com/docs/en/api/rate-limits,
# "Response headers"); OpenAI sends durations such as ``6m0s``
# (https://developers.openai.com/api/docs/guides/rate-limits).
_RESET_TIME_HEADERS = tuple(
    f"anthropic-ratelimit-{kind}-reset"
    for kind in ("requests", "tokens", "input-tokens", "output-tokens")
)
_RESET_DURATION_HEADERS = ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """Parse ``Retry-After`` as delta-seconds or an HTTP-date. ``None`` if absent
    or unparseable."""
    raw = _header(headers, "retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _seconds_until(raw)


def _reset_seconds(headers: Any) -> float | None:
    """The latest rate-limit reset the provider announced: the call can fail
    until every exhausted limit is back, and the headers do not say which one
    was exhausted."""
    waits: list[float | None] = []
    for name in _RESET_TIME_HEADERS + _RESET_DURATION_HEADERS:
        raw = _header(headers, name)
        if raw:
            parse = _seconds_until if name in _RESET_TIME_HEADERS else _duration_seconds
            waits.append(parse(raw))
    known = [w for w in waits if w is not None]
    return max(known) if known else None


def _short(value: Any) -> str | None:
    if value is None or value == "" or isinstance(value, (dict, list)):
        return None
    return str(value)[:_FIELD_MAX]


def _upstream_fields(status: int | None, body: str, headers: Any) -> dict[str, Any]:
    """The machine-readable parts of a provider's error answer.

    Read only from documented error objects: OpenAI's
    ``error.{code,type,param}``
    (https://developers.openai.com/api/docs/guides/error-codes); Anthropic's
    ``error.type``, ``error.details.error_code`` and body ``request_id``
    (https://platform.claude.com/docs/en/api/errors,
    https://platform.claude.com/docs/en/api/rate-limits); OpenRouter's numeric
    ``error.code`` (https://openrouter.ai/docs/api-reference/errors); and
    MiniMax's ``base_resp.status_code``, which the adapter already read for
    plan exhaustion. The request id is Anthropic's ``request-id`` or OpenAI's
    ``x-request-id`` header. The error ``message`` is never copied: it can
    echo the prompt.
    """
    try:
        data = json.loads(body) if body else None
    except (TypeError, ValueError):
        data = None
    if not isinstance(data, dict):
        data = {}
    err = data.get("error")
    fields: dict[str, Any] = {"status_code": status}
    if isinstance(err, dict):
        fields["provider_code"] = _short(err.get("code"))
        fields["provider_type"] = _short(err.get("type"))
        fields["provider_param"] = _short(err.get("param"))
        details = err.get("details")
        if fields["provider_code"] is None and isinstance(details, dict):
            fields["provider_code"] = _short(details.get("error_code"))
    base = data.get("base_resp")
    if fields.get("provider_code") is None and isinstance(base, dict) and base.get("status_code"):
        fields["provider_code"] = _short(base["status_code"])
    request_id = _header(headers, "request-id") or _header(headers, "x-request-id")
    fields["provider_request_id"] = _short(request_id or data.get("request_id"))
    wait = retry_after_seconds(headers)
    if wait is None and status == 429:
        wait = _reset_seconds(headers)
    fields["retry_after_s"] = None if wait is None else min(wait, _MAX_WAIT_S)
    return fields


def _attach(exc: SommError, fields: Mapping[str, Any]) -> SommError:
    for name, value in fields.items():
        if value is not None:
            setattr(exc, name, value)
    return exc


def http_error(
    status: int,
    body: str,
    *,
    provider: str,
    model: str,
    headers: Mapping[str, str] | None = None,
    overloaded_statuses: tuple[int, ...] = (529,),
    detail: str | None = None,
) -> SommError:
    """The SommError a non-2xx provider answer means, with the upstream fields
    (``status_code``, ``provider_*``, ``retry_after_s``) filled.

    ``detail`` replaces the message, which otherwise quotes up to 200
    characters of the body, for an adapter that keeps provider text out of
    its messages or has a better hint.
    """
    body = body or ""
    fields = _upstream_fields(status, body, headers)
    snippet = body[:200]

    def say(default: str) -> str:
        return default if detail is None else detail

    exc: SommError
    if looks_like_insufficient_credit(body) or status == 402:
        # Out of money or quota: a billing state on this provider, not a bad
        # request or a bad key. Transient for the router, which falls through;
        # 402 Payment Required means this whatever the body says.
        exc = SommInsufficientCredit(say(f"{provider} out of credit on {model}: {snippet}"), model=model)
    elif status in (401, 403):
        exc = SommAuthError(say(f"{provider} auth failed ({status}): {snippet}"))
    elif status == 400 and not _looks_like_model_not_found(body):
        exc = SommBadRequest(say(f"{provider} 400 on {model}: {snippet}"))
    elif status in (400, 404):
        exc = SommModelNotFound(
            say(f"{provider} {status} — model {model!r} not found: {snippet}"), model=model
        )
    elif status == 408:
        exc = SommTimeout(say(f"{provider} 408 on {model}"), cooldown_s=60.0, model=model)
    elif status == 429:
        wait = fields["retry_after_s"]
        exc = SommRateLimited(
            say(f"{provider} 429 on {model}: {snippet}"),
            retry_after_s=120.0 if wait is None else wait,
            model=model,
        )
    elif status in overloaded_statuses:
        exc = SommTransientError(
            say(f"{provider} overloaded ({status}) on {model}"), cooldown_s=60.0, model=model
        )
    elif 500 <= status < 600:
        exc = SommUpstream5xx(say(f"{provider} {status} on {model}"), cooldown_s=30.0, model=model)
    else:
        exc = SommTransientError(
            say(f"{provider} unexpected {status} on {model}: {snippet}"), cooldown_s=30.0, model=model
        )
    return _attach(exc, fields)


def classify_http_status(
    status: int,
    body: str,
    *,
    provider: str,
    model: str,
    headers: Mapping[str, str] | None = None,
    overloaded_statuses: tuple[int, ...] = (529,),
    detail: str | None = None,
) -> None:
    """Raise the SommError that an HTTP status from a provider means.

    Returns normally on 2xx. ``body`` is the response text (it is truncated
    for the message; the whole thing is scanned for billing phrases and
    parsed for the provider's error object). ``overloaded_statuses`` are
    provider-specific "try again shortly" codes outside the 5xx range
    (Anthropic's 529). The raised error carries the upstream fields; see
    :func:`http_error`.
    """
    if 200 <= status < 300:
        return
    raise http_error(
        status, body, provider=provider, model=model, headers=headers,
        overloaded_statuses=overloaded_statuses, detail=detail,
    )


def classify_body_error(err: object, *, provider: str, model: str) -> None:
    """Raise for an error object a provider returned inside a 2xx body.

    OpenRouter and some gateways answer ``200 {"error": {...}}``. The shape is
    OpenAI-ish: ``code`` may be an HTTP status, ``type`` and ``message`` are
    free text.
    """
    if isinstance(err, dict):
        msg = str(err.get("message", ""))
        code = err.get("code")
        typ = str(err.get("type", ""))
        exc: SommError
        if str(code) in _PLAN_EXHAUSTED_BODY_CODES or looks_like_insufficient_credit(f"{code} {typ} {msg}"):
            exc = SommInsufficientCredit(f"{provider} out of credit on {model}: {msg}", model=model)
        elif code == 404 or _looks_like_model_not_found(f"{code} {msg}"):
            exc = SommModelNotFound(f"{provider} body-404 on {model}: {msg}", model=model)
        elif code == 429 or "rate" in msg.lower():
            exc = SommRateLimited(f"{provider} body-429 on {model}: {msg}", retry_after_s=120.0, model=model)
        elif "auth" in typ.lower() or "auth" in msg.lower():
            exc = SommAuthError(f"{provider} body-auth on {model}: {msg}")
        else:
            exc = SommTransientError(f"{provider} body-error on {model}: {msg}", cooldown_s=60.0, model=model)
        # The HTTP status was 2xx, so only the error object's own fields apply.
        fields = _upstream_fields(None, json.dumps({"error": err}, default=str), None)
        raise _attach(exc, fields)
    raise SommTransientError(f"{provider} body-error on {model}: {err}", cooldown_s=60.0, model=model)


class SommDeferred(SommError):
    """No inference was attempted. The application owns resumption/queueing."""

    code = "SOMM_WORK_DEFERRED"

    def __init__(self, decision: dict):
        self.decision = decision
        super().__init__("Stronger-model work deferred to conserve weekly ChatGPT usage")

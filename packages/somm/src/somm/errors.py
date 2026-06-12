"""Provider/routing error taxonomy.

Transient errors cool the offending (provider, model) and let the router try
the next one. Fatal errors bubble up — the caller must handle them. Every
exception carries a canonical SOMM_* code so error messages stay consistent
(per PLAN.md DX canonical error format).
"""

from __future__ import annotations


class SommError(Exception):
    """Base for all somm exceptions."""

    code: str = "SOMM_ERROR"


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

    def __init__(self, detail: str = "", retry_after_s: float = 120.0) -> None:
        super().__init__(detail, cooldown_s=retry_after_s)
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
    """Strict mode refused an unregistered workload/prompt."""

    code = "SOMM_STRICT_MODE"


class SommPrivacyViolation(SommError):
    """Attempted upstream call on a privacy_class=private workload."""

    code = "SOMM_PRIVACY_VIOLATION"


# Phrases providers use when the account is out of money/quota. Matched
# case-insensitively against the raw error body. Kept deliberately narrow:
# a false positive would silently fall through on a genuinely malformed
# request, so we only match wording that unambiguously means "billing".
_INSUFFICIENT_CREDIT_SIGNALS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient credit",
    "credit balance is too low",
    "credit balance too low",
    "exceeded your current quota",
)


def looks_like_insufficient_credit(body: str) -> bool:
    """True if a provider error body signals an exhausted balance/quota.

    Used by provider adapters to reclassify an otherwise-fatal billing 400/429
    as a transient SommInsufficientCredit so the router skips the lapsed
    provider instead of aborting the whole fallback chain.
    """
    low = body.lower()
    return any(sig in low for sig in _INSUFFICIENT_CREDIT_SIGNALS)

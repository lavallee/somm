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

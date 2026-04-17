"""Drop-in shim for codebases with a legacy `FooLLM.generate()` wrapper class.

Target call site shape:
    result = llm.generate(prompt, system="", max_tokens=256, provider=None)
    result.text, result.provider, result.model  # legacy fields

The shim accepts the same signature + workload tag, returns an object
with legacy attributes PLUS somm-native extras (call_id, cost_usd,
tokens_in, tokens_out, latency_ms, outcome).

The intent: enable a one-line swap at import time:

    from somm.compat import GenericLLMCompat as FooLLM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from somm.client import SommLLM


@dataclass(slots=True)
class LegacyLLMResult:
    """Legacy-shape result. Mirrors what most custom-wrapper codebases
    already return, with somm-native fields added non-breakingly.
    """

    text: str
    provider: str
    model: str
    # somm-native extras — non-breaking additions:
    call_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    outcome: str = "ok"
    error_kind: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class GenericLLMCompat:
    """Shim that preserves `(prompt, system, max_tokens, provider) -> result`
    while adding full somm telemetry + routing underneath.

    Accepts an optional `workload` argument on every call. If the call-site
    code doesn't pass workload, a default ("default") is used — which is
    fine for observe mode but noisy in strict mode.

    Args:
        project: project name (falls back to SOMM_PROJECT env or "default").
        mode: "observe" (default) or "strict".
        providers: explicit provider list; otherwise uses the default chain.
    """

    def __init__(
        self,
        project: str | None = None,
        mode: str | None = None,
        providers: list | None = None,
        default_workload: str = "default",
    ) -> None:
        from somm.client import SommLLM

        self._llm: SommLLM = SommLLM(project=project, mode=mode, providers=providers)
        self._default_workload = default_workload

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 256,
        provider: str | None = None,
        workload: str | None = None,
        temperature: float = 0.2,
        model: str | None = None,
    ) -> LegacyLLMResult:
        """Same signature as most legacy wrappers. Returns LegacyLLMResult."""
        r = self._llm.generate(
            prompt=prompt,
            system=system,
            workload=workload or self._default_workload,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            provider=provider,
        )
        return LegacyLLMResult(
            text=r.text,
            provider=r.provider,
            model=r.model,
            call_id=r.call_id,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            latency_ms=r.latency_ms,
            cost_usd=r.cost_usd,
            outcome=r.outcome.value,
            error_kind=r.error_kind,
            raw=r.raw or {},
        )

    def extract_structured(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
        workload: str | None = None,
    ) -> dict | list:
        """Return parsed JSON or {"raw": ..., "_somm_parse_err": True}."""
        return self._llm.extract_structured(
            prompt=prompt,
            system=system,
            workload=workload or self._default_workload,
            max_tokens=max_tokens,
        )

    def probe_providers(self, n: int = 4) -> list[str]:
        """Legacy alias for parallel_slots. Returns a striped slot assignment."""
        return self._llm.parallel_slots(n)

    # Optional: OpenAI-ish `chat` method for codebases that used it.
    def chat(self, messages: list[dict], **kwargs) -> LegacyLLMResult:
        """Accepts OpenAI-style messages; splits into system/user prompt."""
        system = ""
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system = content
            else:
                user_parts.append(content)
        return self.generate(
            prompt="\n\n".join(user_parts),
            system=system,
            **kwargs,
        )

    # ------------------------------------------------------------------

    @property
    def llm(self) -> SommLLM:
        """Direct handle for somm-native operations (register_prompt,
        enable_shadow, etc.) when the legacy API isn't enough.
        """
        return self._llm

    def close(self) -> None:
        """Flush telemetry + stop writer thread. Optional for short-lived processes."""
        self._llm.close()

    def __enter__(self) -> GenericLLMCompat:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

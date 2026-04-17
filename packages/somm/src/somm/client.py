"""SommLLM — the library entry point.

D1 minimal: .generate() against a single provider (ollama), telemetry writes
via WriterQueue, demo-mode default (auto-registers unknown workloads with a
warning), call_id in result for provenance.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from somm_core import Outcome, SommResult
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.models import Call, Prompt
from somm_core.parse import extract_json, stable_hash
from somm_core.repository import Repository

from somm.errors import SommStrictMode as _SommStrictMode
from somm.prompts import get_prompt, register_prompt
from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import SommProvider, SommRequest
from somm.providers.minimax import MinimaxProvider
from somm.providers.ollama import OllamaProvider
from somm.providers.openai import OpenAIProvider
from somm.providers.openrouter import OpenRouterProvider
from somm.routing import ProviderHealthTracker, Router
from somm.slots import parallel_slots as _parallel_slots
from somm.telemetry import WriterQueue

SommStrictMode = _SommStrictMode  # re-export; new canonical lives in somm.errors


class SommLLM:
    """The library handle. One per process per project.

    Default mode is 'observe' (DX default — TTHW first). Strict mode enforces
    workload/prompt registration; enable via `SommLLM(mode="strict")` or env
    `SOMM_MODE=strict`.
    """

    def __init__(
        self,
        project: str | None = None,
        mode: str | None = None,
        providers: list[SommProvider] | None = None,
        config: Config | None = None,
    ) -> None:
        self.config = config or load_config(project=project)
        if mode is not None:
            self.config.mode = mode

        self.repo = Repository(self.config.db_path)
        self._tracker = ProviderHealthTracker(self.repo)
        self.providers: list[SommProvider] = providers or self._default_providers()
        self.router = Router(self.providers, self._tracker)
        self._writer = WriterQueue(self.repo, self.config.spool_dir)
        self._writer.start()

    def _default_providers(self) -> list[SommProvider]:
        """Build the default provider chain from config.

        Order (sovereign-first):
          1. ollama (local; no network)
          2. openrouter (free roster + cooldowns)
          3. minimax (if key)
          4. anthropic (if key)
          5. openai / openai-compatible (if key)

        Every commercial-API provider is opt-in via its env var. Library
        works offline with just ollama.
        """
        chain: list[SommProvider] = [
            OllamaProvider(
                base_url=self.config.ollama_url,
                default_model=self.config.ollama_model,
            )
        ]
        if self.config.openrouter_api_key:
            chain.append(
                OpenRouterProvider(
                    api_key=self.config.openrouter_api_key,
                    roster=self.config.openrouter_roster,
                    tracker=self._tracker,
                )
            )
        if self.config.minimax_api_key:
            chain.append(
                MinimaxProvider(
                    api_key=self.config.minimax_api_key,
                    default_model=self.config.minimax_model,
                )
            )
        if self.config.anthropic_api_key:
            chain.append(
                AnthropicProvider(
                    api_key=self.config.anthropic_api_key,
                    default_model=self.config.anthropic_model,
                )
            )
        if self.config.openai_api_key:
            chain.append(
                OpenAIProvider(
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                    default_model=self.config.openai_model,
                )
            )
        return chain

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
    ) -> SommResult:
        """Run one LLM call. Writes telemetry synchronously at the row level.

        demo mode: auto-registers unknown workloads as 'ad_hoc' equivalents.
        strict mode: raises SommStrictMode if workload isn't registered.
        """
        wl = self.repo.workload_by_name(workload, self.config.project)
        if wl is None:
            if self.config.mode == "strict":
                raise SommStrictMode(
                    f"SOMM_WORKLOAD_UNREGISTERED\n\n"
                    f"Problem: This call used workload {workload!r}, but it is not registered.\n"
                    f"Cause: strict mode requires workload metadata before calls are logged.\n"
                    f"Fix:\n"
                    f"  somm workload add {workload} --from-example structured-extraction\n"
                    f"  # or switch to observe mode:\n"
                    f"  export SOMM_MODE=observe\n"
                    f"Docs: docs/errors/SOMM_WORKLOAD_UNREGISTERED.md"
                )
            wl = self.repo.register_workload(name=workload, project=self.config.project)

        req = SommRequest(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )

        call_id = str(uuid.uuid4())
        ts = datetime.now(UTC)
        outcome = Outcome.OK
        error_kind: str | None = None
        tokens_in = tokens_out = latency_ms = 0
        actual_model = model or ""
        actual_provider = ""
        text = ""

        if provider is not None:
            chosen = self._pick_provider(provider)
            try:
                resp = chosen.generate(req)
                text = resp.text
                actual_provider = chosen.name
                actual_model = resp.model
                tokens_in = resp.tokens_in
                tokens_out = resp.tokens_out
                latency_ms = resp.latency_ms
                if not text.strip():
                    outcome = Outcome.EMPTY
            except Exception as exc:
                outcome = Outcome.UPSTREAM_ERROR
                error_kind = type(exc).__name__
                actual_provider = chosen.name
        else:
            try:
                router_result = self.router.dispatch(req)
                resp = router_result.response
                text = resp.text
                actual_provider = router_result.provider
                actual_model = resp.model
                tokens_in = resp.tokens_in
                tokens_out = resp.tokens_out
                latency_ms = resp.latency_ms
                if not text.strip():
                    outcome = Outcome.EMPTY
            except Exception as exc:
                outcome = (
                    Outcome.EXHAUSTED
                    if type(exc).__name__ == "SommProvidersExhausted"
                    else Outcome.UPSTREAM_ERROR
                )
                error_kind = type(exc).__name__

        result = SommResult(
            text=text,
            provider=actual_provider,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=0.0,  # D3: cost calc lands with model_intel worker
            call_id=call_id,
            outcome=outcome,
            error_kind=error_kind,
        )

        call = Call(
            id=call_id,
            ts=ts,
            project=self.config.project,
            workload_id=wl.id,
            prompt_id=None,  # D2b: prompt versioning lands with register_prompt
            provider=actual_provider,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=0.0,
            outcome=outcome,
            error_kind=error_kind,
            prompt_hash=stable_hash(prompt),
            response_hash=stable_hash(text),
        )
        self._writer.submit(call)
        return result

    # ------------------------------------------------------------------

    def _pick_provider(self, name: str | None) -> SommProvider:
        if name:
            for p in self.providers:
                if p.name == name:
                    return p
            raise ValueError(f"provider {name!r} not configured")
        return self.providers[0]

    # ------------------------------------------------------------------
    # Structured output

    def extract_structured(
        self,
        prompt: str,
        system: str = "",
        workload: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.1,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict | list:
        """Call the LLM and extract JSON from the response.

        Handles markdown fences, bracket-balanced extraction, qwen2.5 double-
        quote quirk, and `<think>` blocks (already stripped by adapters).

        Returns the parsed dict or list. On parse failure, returns
        `{"raw": <text>, "_somm_parse_err": True}` so the caller can distinguish
        between "LLM said nothing parseable" and "LLM said something parseable".
        """
        result = self.generate(
            prompt=prompt,
            system=system,
            workload=workload,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            provider=provider,
        )
        parsed = extract_json(result.text)
        if parsed is None:
            result.mark(Outcome.BAD_JSON)
            return {"raw": result.text, "_somm_parse_err": True}
        return parsed

    # ------------------------------------------------------------------
    # Prompt versioning

    def register_prompt(
        self,
        workload: str,
        body: str,
        bump: str = "minor",
    ) -> Prompt:
        """Commit a prompt body for a named workload. Idempotent on hash match.

        Args:
            workload: workload name (must exist, or be auto-registered in observe mode).
            body: the prompt body.
            bump: "minor" (default), "major", or an explicit version "vN".
        """
        wl = self._require_workload(workload)
        return register_prompt(self.repo, wl.id, body, bump=bump)

    def prompt(self, workload: str, version: str = "latest") -> Prompt:
        """Fetch a prompt by workload + version.

        Use in calling code:
            body = llm.prompt("claim_extract", version="latest").body
            result = llm.generate(body, workload="claim_extract")
        """
        wl = self._require_workload(workload)
        return get_prompt(self.repo, wl.id, version=version)

    def _require_workload(self, name: str):
        wl = self.repo.workload_by_name(name, self.config.project)
        if wl is None:
            if self.config.mode == "strict":
                raise _SommStrictMode(
                    f"SOMM_WORKLOAD_UNREGISTERED\n\n"
                    f"Problem: Workload {name!r} is not registered.\n"
                    f"Cause: strict mode requires workload metadata first.\n"
                    f"Fix:\n"
                    f"  somm.llm().repo.register_workload(name={name!r}, project=...)\n"
                    f"Docs: docs/errors/SOMM_WORKLOAD_UNREGISTERED.md"
                )
            wl = self.repo.register_workload(name=name, project=self.config.project)
        return wl

    # ------------------------------------------------------------------
    # Parallel-worker slot assignment

    def parallel_slots(self, n: int) -> list[str]:
        """Return a striped assignment of provider names for n parallel workers.

        Preserves sovereignty-first ordering and avoids stampeding one
        provider. Cooled providers are excluded. Use:

            assignments = llm.parallel_slots(4)
            # e.g. ['ollama', 'openrouter', 'ollama', 'openrouter']
            for i, provider_name in enumerate(assignments):
                spawn_worker(i, provider=provider_name)
        """
        return _parallel_slots(self.providers, n, tracker=self._tracker)

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drain the writer queue and stop the thread. Optional for short-lived processes."""
        self._writer.flush()
        self._writer.stop()

    def __enter__(self) -> SommLLM:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def llm(**kwargs) -> SommLLM:
    """Factory matching the plan's `somm.llm(project=...)` signature."""
    return SommLLM(**kwargs)

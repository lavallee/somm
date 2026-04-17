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
from somm_core.models import Call
from somm_core.parse import stable_hash
from somm_core.repository import Repository

from somm.providers.base import SommProvider, SommRequest
from somm.providers.ollama import OllamaProvider
from somm.telemetry import WriterQueue


class SommStrictMode(Exception):
    """Raised in strict mode for unregistered workloads/prompts."""


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
        self.providers: list[SommProvider] = providers or [
            OllamaProvider(
                base_url=self.config.ollama_url,
                default_model=self.config.ollama_model,
            )
        ]
        self._writer = WriterQueue(self.repo, self.config.spool_dir)
        self._writer.start()

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

        chosen = self._pick_provider(provider)
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
        text = ""

        try:
            resp = chosen.generate(req)
            text = resp.text
            actual_model = resp.model
            tokens_in = resp.tokens_in
            tokens_out = resp.tokens_out
            latency_ms = resp.latency_ms
            if not text.strip():
                outcome = Outcome.EMPTY
        except Exception as exc:
            outcome = Outcome.UPSTREAM_ERROR
            error_kind = type(exc).__name__

        result = SommResult(
            text=text,
            provider=chosen.name,
            model=actual_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=0.0,  # D1: cost calc lands in D2 w/ model_intel
            call_id=call_id,
            outcome=outcome,
            error_kind=error_kind,
        )

        call = Call(
            id=call_id,
            ts=ts,
            project=self.config.project,
            workload_id=wl.id,
            prompt_id=None,  # D1: prompt versioning lands in D2
            provider=chosen.name,
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

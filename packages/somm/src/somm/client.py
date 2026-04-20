"""SommLLM — the library entry point.

D1 minimal: .generate() against a single provider (ollama), telemetry writes
via WriterQueue, demo-mode default (auto-registers unknown workloads with a
warning), call_id in result for provenance.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime

from somm_core import Outcome, SommResult, cost_for_call
from somm_core.pricing import seed_known_pricing
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.models import Call, Prompt
from somm_core.parse import (
    ThinkStreamStripper,
    extract_json,
    infer_capabilities,
    stable_hash,
)
from somm_core.repository import Repository

from somm.errors import SommStrictMode as _SommStrictMode
from somm.prompts import get_prompt, register_prompt
from somm.providers.anthropic import AnthropicProvider
from somm.providers.base import SommProvider, SommRequest
from somm.providers.gemini import GeminiProvider
from somm.providers.minimax import MinimaxProvider
from somm.providers.ollama import OllamaProvider
from somm.providers.openai import OpenAIProvider
from somm.providers.openrouter import OpenRouterProvider
from somm.routing import ProviderHealthTracker, Router
from somm.slots import parallel_slots as _parallel_slots
from somm.telemetry import WriterQueue

SommStrictMode = _SommStrictMode  # re-export; new canonical lives in somm.errors

# Track which (workload, date) pairs have already emitted a budget-exceeded
# warning so we only warn once per workload per day per process.
_warned_budget_exceeded: set[tuple[str, date]] = set()


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

        mirror_repo: Repository | None = None
        if self.config.cross_project_enabled:
            mirror_repo = Repository(self.config.global_db_path)
            # Mirror workload registrations as well so global rollups can
            # resolve names rather than showing "(unregistered)".
            _mirror_workloads(self.repo, mirror_repo)

        self._writer = WriterQueue(self.repo, self.config.spool_dir, mirror_repo=mirror_repo)
        self._writer.start()
        self._mirror_repo = mirror_repo

        # Decisions are ALWAYS mirrored to the global store — unlike calls
        # (per-project for privacy), advisory memory is explicitly
        # cross-project: "last time I picked a vision model, here's why."
        # We lazily create the handle so cold starts don't touch the global
        # path unless a decision is actually recorded.
        self._decision_mirror: Repository | None = mirror_repo

        # Seed pricing on first use so cost tracking works out of the box.
        seed_known_pricing(self.repo)

    def register_workload(self, **kwargs):
        """Register a workload in the project repo AND mirror-if-enabled."""
        wl = self.repo.register_workload(project=self.config.project, **kwargs)
        if self._mirror_repo is not None:
            try:
                self._mirror_repo.register_workload(project=self.config.project, **kwargs)
            except Exception:  # noqa: BLE001
                pass
        return wl

    def _default_providers(self) -> list[SommProvider]:
        """Build the provider chain from config.

        Default order (sovereign-first): ollama → openrouter → minimax →
        anthropic → openai. Override with SOMM_PROVIDER_ORDER env var
        (comma-separated, e.g. "openrouter,minimax,ollama").

        Every commercial-API provider is opt-in via its env var. Library
        works offline with just ollama.
        """
        available: dict[str, SommProvider] = {}
        available["ollama"] = OllamaProvider(
            base_url=self.config.ollama_url,
            default_model=self.config.ollama_model,
        )
        if self.config.openrouter_api_key:
            available["openrouter"] = OpenRouterProvider(
                api_key=self.config.openrouter_api_key,
                roster=self.config.openrouter_roster,
                tracker=self._tracker,
            )
        if self.config.minimax_api_key:
            available["minimax"] = MinimaxProvider(
                api_key=self.config.minimax_api_key,
                default_model=self.config.minimax_model,
            )
        if self.config.anthropic_api_key:
            available["anthropic"] = AnthropicProvider(
                api_key=self.config.anthropic_api_key,
                default_model=self.config.anthropic_model,
            )
        if self.config.openai_api_key:
            available["openai"] = OpenAIProvider(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                default_model=self.config.openai_model,
            )
        if self.config.gemini_api_key:
            available["gemini"] = GeminiProvider(
                api_key=self.config.gemini_api_key,
                default_model=self.config.gemini_model,
            )

        if self.config.provider_order:
            # Exclusive: ONLY providers in the list, in the listed order.
            # If you set SOMM_PROVIDER_ORDER=openrouter,minimax,ollama
            # then anthropic is NOT in the chain, even if its key is set.
            chain = [available[p] for p in self.config.provider_order if p in available]
            return chain if chain else list(available.values())

        # Default: ollama → openrouter → minimax → anthropic → gemini → openai
        default_order = ["ollama", "openrouter", "minimax", "anthropic", "gemini", "openai"]
        return [available[p] for p in default_order if p in available]

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str | list[dict],
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
        capabilities_required: list[str] | None = None,
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

        effective_caps = _merge_caps(
            wl.capabilities_required,
            capabilities_required,
            infer_capabilities(prompt),
        )

        req = SommRequest(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            capabilities_required=effective_caps,
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
                # Preferred provider failed — fall through to the full
                # router chain instead of giving up. "provider=X" means
                # "try X first", not "only X". This is critical for
                # parallel workers: if one provider goes down mid-batch,
                # those workers recover via fallthrough instead of
                # producing empty results.
                from somm.errors import SommFatalError
                if isinstance(exc, SommFatalError):
                    # Auth errors etc. — don't retry, but still try router
                    pass
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
                except Exception as fallback_exc:
                    outcome = Outcome.UPSTREAM_ERROR
                    error_kind = type(exc).__name__
                    actual_provider = chosen.name
                    if hasattr(exc, "model") and exc.model:
                        actual_model = exc.model
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
                if hasattr(exc, "model") and exc.model:
                    actual_model = exc.model
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
            cost_usd=cost_for_call(self.repo, actual_provider, actual_model, tokens_in, tokens_out),
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
            cost_usd=result.cost_usd,
            outcome=outcome,
            error_kind=error_kind,
            prompt_hash=stable_hash(prompt),
            response_hash=stable_hash(text),
        )
        self._writer.submit(call)

        # Feature 3: warn if daily budget cap is exceeded.
        if wl.budget_cap_usd_daily is not None and result.cost_usd > 0:
            today = date.today()
            budget_key = (wl.name, today)
            if budget_key not in _warned_budget_exceeded:
                with self.repo._open() as conn:
                    row = conn.execute(
                        "SELECT COALESCE(SUM(cost_usd), 0) FROM calls "
                        "WHERE workload_id = ? AND date(ts) = date('now')",
                        (wl.id,),
                    ).fetchone()
                daily_cost = row[0] if row else 0.0
                if daily_cost > wl.budget_cap_usd_daily:
                    _warned_budget_exceeded.add(budget_key)
                    print(
                        f"[somm] WARNING: workload {wl.name!r} daily cost "
                        f"${daily_cost:.4f} exceeds budget cap "
                        f"${wl.budget_cap_usd_daily:.4f}.",
                        file=sys.stderr,
                    )

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
    # Streaming

    def stream(
        self,
        prompt: str | list[dict],
        system: str = "",
        workload: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.2,
        model: str | None = None,
        provider: str | None = None,
    ) -> Iterator[str]:
        """Stream text deltas from the LLM. Yields user-visible text chunks
        (with `<think>` blocks stripped across chunk boundaries).

        Telemetry is written after the stream completes. No mid-stream
        router fallback in v0.1 — first non-cooled provider handles the
        whole stream or errors out.

        Usage:
            for piece in llm.stream("tell a story", workload="story"):
                print(piece, end="", flush=True)
        """
        from somm.providers.base import SommRequest

        wl = self._require_workload(workload)
        req = SommRequest(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )

        chosen = self._pick_stream_provider(provider)
        call_id = str(uuid.uuid4())
        ts = datetime.now(UTC)
        stripper = ThinkStreamStripper()

        t0 = time.monotonic()
        collected = []
        outcome = Outcome.OK
        error_kind: str | None = None
        tokens_in = tokens_out = 0
        actual_model = model or ""

        try:
            for chunk in chosen.stream(req):
                if chunk.text:
                    visible = stripper.feed(chunk.text)
                    if visible:
                        collected.append(visible)
                        yield visible
                if chunk.done:
                    tail = stripper.flush()
                    if tail:
                        collected.append(tail)
                        yield tail
                    break
            # Streaming providers don't always give token counts — estimate
            # from length as a fallback.
            text = "".join(collected)
            if not actual_model:
                actual_model = chosen.name
            tokens_in = chosen.estimate_tokens(prompt, actual_model) + chosen.estimate_tokens(
                system, actual_model
            )
            tokens_out = chosen.estimate_tokens(text, actual_model)
            if not text.strip():
                outcome = Outcome.EMPTY
        except Exception as exc:
            outcome = Outcome.UPSTREAM_ERROR
            error_kind = type(exc).__name__
            text = "".join(collected)
            raise
        finally:
            latency_ms = int((time.monotonic() - t0) * 1000)
            full_text = "".join(collected)
            call = Call(
                id=call_id,
                ts=ts,
                project=self.config.project,
                workload_id=wl.id,
                prompt_id=None,
                provider=chosen.name,
                model=actual_model or chosen.name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost_for_call(
                    self.repo,
                    chosen.name,
                    actual_model or chosen.name,
                    tokens_in,
                    tokens_out,
                ),
                outcome=outcome,
                error_kind=error_kind,
                prompt_hash=stable_hash(prompt),
                response_hash=stable_hash(full_text),
            )
            self._writer.submit(call)

    def _pick_stream_provider(self, name: str | None):
        if name:
            return self._pick_provider(name)
        # First non-cooled provider for streams (no mid-stream fallback).
        for p in self.providers:
            if not self._tracker.get(p.name).is_cooling():
                return p
        # If all cooled, use the first and let it raise.
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

    def enable_shadow(
        self,
        workload: str,
        gold_provider: str,
        gold_model: str,
        sample_rate: float = 0.02,
        budget_usd_daily: float = 1.0,
        max_grades_per_run: int = 20,
    ) -> None:
        """Opt a workload in to shadow-eval.

        Off by default per sovereignty + privacy. Private workloads
        (privacy_class=PRIVATE) cannot be shadow-graded — the schema view
        enforces this + the worker defense-in-depth checks again.
        """
        wl = self._require_workload(workload)
        from somm_core.models import PrivacyClass

        if wl.privacy_class == PrivacyClass.PRIVATE:
            from somm.errors import SommPrivacyViolation

            raise SommPrivacyViolation(
                f"SOMM_PRIVACY_VIOLATION\n\n"
                f"Problem: workload {workload!r} is privacy_class=private; shadow-eval forbidden.\n"
                f"Cause: shadow-eval re-sends prompt/response to gold-model provider.\n"
                f"Fix: downgrade privacy_class OR keep shadow off for this workload.\n"
                f"Docs: docs/errors/SOMM_PRIVACY_VIOLATION.md"
            )
        self.repo.set_shadow_config(
            wl.id,
            {
                "gold_provider": gold_provider,
                "gold_model": gold_model,
                "sample_rate": sample_rate,
                "budget_usd_daily": budget_usd_daily,
                "max_grades_per_run": max_grades_per_run,
            },
        )

    def disable_shadow(self, workload: str) -> None:
        wl = self._require_workload(workload)
        self.repo.set_shadow_config(wl.id, None)

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


def _merge_caps(*sources: list[str] | None) -> list[str]:
    """Merge capability lists deterministically — preserves declaration order
    of the first source to name each capability."""
    seen: dict[str, None] = {}
    for src in sources:
        if not src:
            continue
        for cap in src:
            if cap and cap not in seen:
                seen[cap] = None
    return list(seen.keys())


def _mirror_workloads(src: Repository, dst: Repository) -> None:
    """Copy workloads rows from src to dst (idempotent on id). Called once
    on SommLLM init when cross_project_enabled is set."""
    try:
        with src._open() as s_conn:
            rows = s_conn.execute(
                "SELECT id, name, project, description, input_schema_json, "
                "output_schema_json, quality_criteria_json, budget_cap_usd_daily, "
                "privacy_class, created_at, shadow_config_json, "
                "capabilities_required_json FROM workloads"
            ).fetchall()
        if not rows:
            return
        with dst._open() as d_conn:
            d_conn.executemany(
                """
                INSERT OR IGNORE INTO workloads
                    (id, name, project, description,
                     input_schema_json, output_schema_json, quality_criteria_json,
                     budget_cap_usd_daily, privacy_class, created_at, shadow_config_json,
                     capabilities_required_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    except Exception:  # noqa: BLE001 — best-effort mirror
        pass

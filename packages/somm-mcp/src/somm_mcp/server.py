"""FastMCP server — exposes 10 tools to coding agents.

Tool catalog:
  somm_stats              — rolled-up call counts per (workload, provider, model)
  somm_search_calls       — query the call log by filters
  somm_recommend          — open recommendations + shadow-eval ranking
                            (+ cold-start sommelier candidates when sparse)
  somm_advise             — sommelier: free-form candidate ranking by
                            capability / price / provider / workload
  somm_record_decision    — persist the outcome of a sommelier conversation
                            (always cross-project mirrored)
  somm_search_decisions   — recall prior decisions by question / workload /
                            provider (default scope: global)
  somm_register_workload  — commit a workload (+ optional privacy/schemas)
  somm_register_prompt    — commit a new prompt version for a workload
  somm_compare            — run a prompt through N models side-by-side (provider-dependent)
  somm_replay             — replay a stored call against a different model (provider-dependent)

The handlers are thin — they delegate to `somm-core.Repository`,
`somm.sommelier`, `somm.prompts`, and `somm.client`.

Providers parameter is optional — if omitted, compare/replay return a
structured error rather than being hidden from the catalog, so tool
discovery stays predictable across deployments.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from somm.prompts import register_prompt
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.models import PrivacyClass
from somm_core.parse import stable_hash
from somm_core.repository import Repository

if TYPE_CHECKING:
    from somm.providers.base import SommProvider


def build_server(
    config: Config | None = None,
    providers: list[SommProvider] | None = None,
) -> FastMCP:
    cfg = config or load_config()
    repo = Repository(cfg.db_path)
    provider_map: dict[str, SommProvider] = {p.name: p for p in providers} if providers else {}
    server = FastMCP("somm")

    # ------------------------------------------------------------------
    # somm_stats (always available)

    @server.tool()
    def somm_stats(since_days: int = 7) -> dict:
        """Rolled-up call counts + token + cost + failure stats per (workload, provider, model).

        Args:
            since_days: Window in days (default 7).

        Returns:
            dict with 'project', 'window_days', and 'rows' (list).
        """
        rows = repo.stats_by_workload(cfg.project, since_days=since_days)
        return {"project": cfg.project, "window_days": since_days, "rows": rows}

    # ------------------------------------------------------------------
    # somm_search_calls

    @server.tool()
    def somm_search_calls(
        workload: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        outcome: str | None = None,
        since_days: int = 7,
        limit: int = 50,
    ) -> dict:
        """Query the `calls` table. Filter by workload name / provider / model /
        outcome / time window. Returns most-recent-first.

        Use before compare/replay to find a representative call_id.
        """
        rows = _search_calls(
            repo,
            cfg.project,
            workload=workload,
            provider=provider,
            model=model,
            outcome=outcome,
            since_days=since_days,
            limit=limit,
        )
        return {"project": cfg.project, "count": len(rows), "rows": rows}

    # ------------------------------------------------------------------
    # somm_recommend

    @server.tool()
    def somm_recommend(workload: str, since_days: int = 30) -> dict:
        """Return open recommendations for a workload + top models by shadow-eval quality.

        Combines agent-emitted recommendations with a ranked list of models that
        have been shadow-graded on this workload. Low-confidence when shadow data
        is sparse.

        Args:
            workload: workload name (must be registered in the current project).
            since_days: window for shadow-eval aggregation (default 30).
        """
        wl = repo.workload_by_name(workload, cfg.project)
        if wl is None:
            return {"error": f"workload {workload!r} not registered in project {cfg.project!r}"}

        recs = _open_recommendations(repo, wl.id)
        ranked = _shadow_ranking(repo, wl.id, since_days=since_days)

        # Cold-start fallback: no shadow data yet → ask the sommelier for
        # candidates grounded in model_intel + capability coverage + prior
        # cross-project decisions. Better than an empty list.
        cold_start: list[dict] = []
        prior_decisions: list[dict] = []
        if not ranked:
            from somm.sommelier import AdviseConstraints, advise

            cold_constraints = AdviseConstraints(
                capabilities=list(wl.capabilities_required),
                workload=wl.name,
                limit=5,
            )
            cold_start = [c.as_dict() for c in advise(repo, cold_constraints)]
            prior_decisions = _relevant_decisions(repo, workload=wl.name)

        return {
            "workload": workload,
            "project": cfg.project,
            "privacy_class": wl.privacy_class.value,
            "capabilities_required": list(wl.capabilities_required),
            "open_recommendations": recs,
            "shadow_rankings": ranked,
            "cold_start_candidates": cold_start,
            "prior_decisions": prior_decisions,
            "note": (
                "No shadow data yet — returning cold-start candidates "
                "from model_intel and any prior decisions recorded for "
                "this workload. Enable shadow-eval via "
                "SommLLM.enable_shadow(workload, gold_provider, gold_model) "
                "to start grading real calls."
            )
            if not ranked
            else None,
        }

    # ------------------------------------------------------------------
    # somm_advise  (sommelier — free-form candidate ranking)

    @server.tool()
    def somm_advise(
        question: str,
        capabilities: list[str] | None = None,
        providers: list[str] | None = None,
        max_price_in_per_1m: float | None = None,
        max_price_out_per_1m: float | None = None,
        min_context_window: int | None = None,
        free_only: bool = False,
        workload: str | None = None,
        limit: int = 8,
    ) -> dict:
        """Rank candidate (provider, model) pairs against the given constraints.

        Returns the top-N ranked candidates from model_intel with reasoning,
        plus any prior cross-project decisions whose question hashes match.
        The agent uses this to surface options + rationale to the user,
        then records the chosen outcome via `somm_record_decision`.

        Args:
            question: natural-language question (stored verbatim on any
              resulting decision; normalised + hashed for dedup).
            capabilities: required capability tokens (e.g. ["vision"]).
              Hard filter — known-incapable models are excluded.
            providers: whitelist of providers to consider.
            max_price_in_per_1m / max_price_out_per_1m: price ceilings in
              USD per 1M tokens. Null = no ceiling.
            min_context_window: minimum context window in tokens.
            free_only: shortcut — both input/output prices must be 0.
            workload: optional — looks up any shadow-eval scores for this
              workload so graded models get a ranking bonus.
            limit: max candidates to return (default 8).
        """
        from somm.sommelier import AdviseConstraints, consult

        constraints = AdviseConstraints(
            capabilities=list(capabilities or []),
            providers=list(providers) if providers else None,
            max_price_in_per_1m=max_price_in_per_1m,
            max_price_out_per_1m=max_price_out_per_1m,
            min_context_window=min_context_window,
            free_only=free_only,
            workload=workload,
            limit=limit,
        )
        result = consult(
            repo,
            question=question,
            constraints=constraints,
            project=cfg.project,
            global_repo=_global_repo(cfg),
        )
        return result.as_dict()

    # ------------------------------------------------------------------
    # somm_record_decision

    @server.tool()
    def somm_record_decision(
        question: str,
        rationale: str,
        candidates: list[dict],
        chosen_provider: str | None = None,
        chosen_model: str | None = None,
        workload: str | None = None,
        constraints: dict | None = None,
        agent: str | None = None,
    ) -> dict:
        """Record the outcome of a sommelier conversation.

        Decisions are *always* mirrored to the cross-project global store
        (unlike call telemetry, which is per-project by default). The value
        is explicitly cross-project: "last time I picked a vision model,
        this is what I chose and why."

        Args:
            question: the original question text (stored verbatim; normalised
              for dedup).
            rationale: the "why" in the user's (or agent's) own words.
              Keep short — this is the payload future sessions will read.
            candidates: the candidates considered, usually the `candidates`
              list from somm_advise serialised as-is.
            chosen_provider / chosen_model: the pick (both optional — a
              decision may legitimately be "none of these, recheck later").
            workload: optional workload scope.
            constraints: the constraint dict from somm_advise (stored for
              audit / dedup).
            agent: the caller identifier (e.g. "claude-code-malo").
        """
        from somm.sommelier import build_decision

        wl = None
        if workload:
            wl = repo.workload_by_name(workload, cfg.project)

        decision = build_decision(
            question=question,
            candidates=candidates,
            rationale=rationale,
            project=cfg.project,
            chosen_provider=chosen_provider,
            chosen_model=chosen_model,
            workload=workload,
            workload_id=wl.id if wl else None,
            constraints=constraints,
            agent=agent,
        )

        # Primary write
        repo.record_decision(decision)
        mirrored = False
        # Always mirror to the global store, even when cross-project is off
        # for calls — decisions are advisory memory by definition.
        try:
            global_repo = _global_repo(cfg)
            if global_repo is not None:
                global_repo.record_decision(decision)
                mirrored = True
        except Exception as e:  # noqa: BLE001 — mirror must not block
            return {
                "decision_id": decision.id,
                "mirrored": False,
                "mirror_error": str(e),
            }
        return {
            "decision_id": decision.id,
            "question_hash": decision.question_hash,
            "mirrored": mirrored,
            "chosen": {
                "provider": decision.chosen_provider,
                "model": decision.chosen_model,
            },
        }

    # ------------------------------------------------------------------
    # somm_search_decisions

    @server.tool()
    def somm_search_decisions(
        question: str | None = None,
        workload: str | None = None,
        chosen_provider: str | None = None,
        project: str | None = None,
        scope: str = "global",
        limit: int = 10,
    ) -> dict:
        """Look up past sommelier decisions.

        By default scope="global" queries the cross-project store — so a
        decision made in one project can inform reasoning in another. Use
        scope="project" to restrict to the current project's DB only.

        Args:
            question: free-text search; matches by question_hash first,
              then by LIKE on the natural-language text.
            workload: filter by workload name or id.
            chosen_provider: filter by the provider that was picked.
            project: filter by project name (any scope).
            scope: "global" (default) or "project".
            limit: max results (default 10).
        """
        if scope == "project":
            search_repo = repo
        else:
            g = _global_repo(cfg)
            search_repo = g if g is not None else repo

        decisions = search_repo.search_decisions(
            question=question,
            project=project,
            workload=workload,
            chosen_provider=chosen_provider,
            limit=limit,
        )
        return {
            "scope": scope,
            "count": len(decisions),
            "decisions": [_decision_as_dict(d) for d in decisions],
        }

    # ------------------------------------------------------------------
    # somm_register_workload

    @server.tool()
    def somm_register_workload(
        name: str,
        description: str = "",
        input_schema: dict | None = None,
        output_schema: dict | None = None,
        privacy_class: str = "internal",
        budget_cap_usd_daily: float | None = None,
    ) -> dict:
        """Register a workload in the current project. Idempotent on (name + schemas).

        Args:
            name: workload name (snake_case recommended).
            description: human-readable description.
            input_schema: optional JSON schema for the prompt input.
            output_schema: optional JSON schema for structured output.
            privacy_class: "public" | "internal" | "private" (default: internal).
              private workloads are banned from shadow-eval and any upstream
              egress by the router.
            budget_cap_usd_daily: optional per-workload daily ceiling.
        """
        try:
            pc = PrivacyClass(privacy_class)
        except ValueError:
            return {
                "error": f"invalid privacy_class {privacy_class!r}; use public|internal|private"
            }
        wl = repo.register_workload(
            name=name,
            project=cfg.project,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            privacy_class=pc,
            budget_cap_usd_daily=budget_cap_usd_daily,
        )
        return {
            "workload_id": wl.id,
            "name": wl.name,
            "privacy_class": wl.privacy_class.value,
            "project": cfg.project,
        }

    # ------------------------------------------------------------------
    # somm_register_prompt

    @server.tool()
    def somm_register_prompt(
        workload: str,
        body: str,
        bump: str = "minor",
    ) -> dict:
        """Commit a prompt body for a workload. Idempotent on body hash.

        Args:
            workload: workload name (must be registered).
            body: the prompt body.
            bump: "minor" (default), "major", or explicit like "v3".
        """
        wl = repo.workload_by_name(workload, cfg.project)
        if wl is None:
            return {"error": f"workload {workload!r} not registered"}
        p = register_prompt(repo, wl.id, body, bump=bump)
        return {
            "prompt_id": p.id,
            "workload": workload,
            "version": p.version,
            "hash": p.hash,
        }

    # ------------------------------------------------------------------
    # somm_compare

    @server.tool()
    def somm_compare(
        prompt: str,
        models: list[str],
        workload: str = "compare",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict:
        """Run a prompt through N models side-by-side. Non-routed; explicit picks.

        Args:
            prompt: the prompt text.
            models: list like ["ollama/gemma4:e4b", "openai/gpt-4o-mini"].
            workload: workload name to tag telemetry (default "compare").
            max_tokens, temperature: per-call params.

        Returns:
            dict with per-model result blocks (text, tokens, latency, cost, call_id).
        """
        if not provider_map:
            return {"error": "no providers configured; compare needs a provider chain"}
        specs = _parse_models(models)
        if not specs:
            return {"error": "no models supplied; example: ['ollama/gemma4:e4b']"}
        # Use the library to get cost + telemetry + strict-mode check in one shot.
        from somm.client import SommLLM

        llm = SommLLM(config=cfg, providers=list(provider_map.values()))
        try:
            out: list[dict] = []
            for provider_name, model in specs:
                if provider_name not in provider_map:
                    out.append(
                        {
                            "provider": provider_name,
                            "model": model,
                            "error": f"provider {provider_name!r} not in chain "
                            f"({list(provider_map)})",
                        }
                    )
                    continue
                try:
                    r = llm.generate(
                        prompt=prompt,
                        workload=workload,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model=model,
                        provider=provider_name,
                    )
                    out.append(
                        {
                            "provider": provider_name,
                            "model": model,
                            "text": r.text,
                            "tokens_in": r.tokens_in,
                            "tokens_out": r.tokens_out,
                            "latency_ms": r.latency_ms,
                            "cost_usd": r.cost_usd,
                            "outcome": r.outcome.value,
                            "call_id": r.call_id,
                        }
                    )
                except Exception as e:
                    out.append(
                        {
                            "provider": provider_name,
                            "model": model,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
            return {"prompt_hash": stable_hash(prompt), "results": out}
        finally:
            llm.close()

    # ------------------------------------------------------------------
    # somm_replay

    @server.tool()
    def somm_replay(
        call_id: str,
        with_provider: str,
        with_model: str,
        max_tokens: int = 256,
    ) -> dict:
        """Replay a stored call against a different (provider, model).

        Requires the original call to have its prompt captured in `samples`
        (per-workload opt-in). Private workloads (privacy_class=PRIVATE) are
        refused — the replay would send the prompt upstream.

        Args:
            call_id: UUID of the original call (from somm_search_calls).
            with_provider, with_model: target for the replay.

        Returns:
            dict with original + replay response + deltas.
        """
        if not provider_map:
            return {"error": "no providers configured; replay needs a provider chain"}

        original = _fetch_call_with_sample(repo, call_id)
        if not original:
            return {"error": f"call {call_id!r} not found"}
        if not original["prompt_body"]:
            return {
                "error": (
                    "original call has no captured prompt — enable per-workload "
                    "sample capture before replay"
                ),
                "call_id": call_id,
            }
        if original["privacy_class"] == PrivacyClass.PRIVATE.value:
            return {
                "error": (
                    "SOMM_PRIVACY_VIOLATION: original workload is privacy_class=private; "
                    "replay would egress prompt upstream — refused."
                ),
                "call_id": call_id,
            }
        if with_provider not in provider_map:
            return {"error": f"provider {with_provider!r} not in chain"}

        from somm.client import SommLLM

        llm = SommLLM(config=cfg, providers=list(provider_map.values()))
        try:
            r = llm.generate(
                prompt=original["prompt_body"],
                workload=original["workload_name"] or "replay",
                model=with_model,
                provider=with_provider,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return {
                "call_id": call_id,
                "original": {
                    "provider": original["provider"],
                    "model": original["model"],
                    "response": original["response_body"],
                    "latency_ms": original["latency_ms"],
                    "cost_usd": original["cost_usd"],
                    "tokens_in": original["tokens_in"],
                    "tokens_out": original["tokens_out"],
                },
                "replay": {
                    "provider": r.provider,
                    "model": r.model,
                    "response": r.text,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "outcome": r.outcome.value,
                    "call_id": r.call_id,
                },
                "deltas": {
                    "latency_ms": r.latency_ms - original["latency_ms"],
                    "cost_usd": r.cost_usd - (original["cost_usd"] or 0),
                    "tokens_out_pct": (
                        100
                        * (r.tokens_out - original["tokens_out"])
                        / max(1, original["tokens_out"])
                    ),
                },
            }
        finally:
            llm.close()

    return server


# ---------------------------------------------------------------------------
# Helpers


def _search_calls(
    repo: Repository,
    project: str,
    workload: str | None,
    provider: str | None,
    model: str | None,
    outcome: str | None,
    since_days: int,
    limit: int,
) -> list[dict]:
    q = [
        "SELECT c.id, c.ts, COALESCE(w.name, '(unregistered)'), c.provider, c.model, "
        "       c.tokens_in, c.tokens_out, c.latency_ms, c.cost_usd, c.outcome, "
        "       c.error_kind, c.prompt_hash, c.response_hash "
        "FROM calls c LEFT JOIN workloads w ON w.id = c.workload_id "
        "WHERE c.project = ? AND c.ts >= datetime('now', ?) "
    ]
    params: list = [project, f"-{since_days} days"]
    if workload:
        q.append("AND w.name = ? ")
        params.append(workload)
    if provider:
        q.append("AND c.provider = ? ")
        params.append(provider)
    if model:
        q.append("AND c.model = ? ")
        params.append(model)
    if outcome:
        q.append("AND c.outcome = ? ")
        params.append(outcome)
    q.append("ORDER BY c.ts DESC LIMIT ?")
    params.append(limit)

    with repo._open() as conn:
        rows = conn.execute("".join(q), params).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "workload": r[2],
            "provider": r[3],
            "model": r[4],
            "tokens_in": r[5],
            "tokens_out": r[6],
            "latency_ms": r[7],
            "cost_usd": r[8] or 0.0,
            "outcome": r[9],
            "error_kind": r[10],
            "prompt_hash": r[11],
            "response_hash": r[12],
        }
        for r in rows
    ]


def _open_recommendations(repo: Repository, workload_id: str) -> list[dict]:
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT id, action, evidence_json, expected_impact, confidence, created_at "
            "FROM recommendations "
            "WHERE workload_id = ? AND dismissed_at IS NULL AND applied_at IS NULL "
            "ORDER BY created_at DESC",
            (workload_id,),
        ).fetchall()
    out = []
    for r in rows:
        try:
            evidence = json.loads(r[2]) if r[2] else {}
        except json.JSONDecodeError:
            evidence = {}
        out.append(
            {
                "id": r[0],
                "action": r[1],
                "evidence": evidence,
                "expected_impact": r[3],
                "confidence": r[4],
                "created_at": r[5],
            }
        )
    return out


def _shadow_ranking(repo: Repository, workload_id: str, since_days: int) -> list[dict]:
    """Top-5 (provider, model) for this workload by shadow-eval score."""
    with repo._open() as conn:
        rows = conn.execute(
            """
            SELECT c.provider, c.model,
                   AVG(COALESCE(er.structural_score, er.embedding_score)) AS score,
                   AVG(c.latency_ms) AS latency_ms,
                   AVG(c.cost_usd)   AS cost_usd,
                   COUNT(er.id)      AS n_evals
            FROM eval_results er
            JOIN calls c ON c.id = er.call_id
            WHERE c.workload_id = ?
              AND c.ts >= datetime('now', ?)
              AND (er.structural_score IS NOT NULL OR er.embedding_score IS NOT NULL)
            GROUP BY c.provider, c.model
            ORDER BY score DESC, cost_usd ASC
            LIMIT 5
            """,
            (workload_id, f"-{since_days} days"),
        ).fetchall()
    return [
        {
            "provider": r[0],
            "model": r[1],
            "score": round(r[2] or 0, 3),
            "latency_ms_avg": round(r[3] or 0),
            "cost_usd_avg": round(r[4] or 0, 6),
            "n_evals": r[5],
        }
        for r in rows
    ]


def _fetch_call_with_sample(repo: Repository, call_id: str) -> dict | None:
    with repo._open() as conn:
        row = conn.execute(
            """
            SELECT c.id, c.provider, c.model, c.tokens_in, c.tokens_out,
                   c.latency_ms, c.cost_usd, c.outcome, c.workload_id,
                   w.name AS workload_name, w.privacy_class,
                   s.prompt_body, s.response_body
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            LEFT JOIN samples s ON s.call_id = c.id
            WHERE c.id = ?
            """,
            (call_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "provider": row[1],
        "model": row[2],
        "tokens_in": row[3],
        "tokens_out": row[4],
        "latency_ms": row[5],
        "cost_usd": row[6] or 0,
        "outcome": row[7],
        "workload_id": row[8],
        "workload_name": row[9],
        "privacy_class": row[10],
        "prompt_body": row[11],
        "response_body": row[12],
    }


def _global_repo(cfg: Config) -> Repository | None:
    """Open the cross-project global store on demand, or return None if
    creation fails. Decisions always target this path — we don't gate
    behind the cross_project_enabled flag (that flag is about calls)."""
    try:
        return Repository(cfg.global_db_path)
    except Exception:  # noqa: BLE001
        return None


def _relevant_decisions(
    repo: Repository,
    question: str | None = None,
    workload: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Lookup that prefers the global store but falls back to local."""
    cfg = load_config()
    for candidate in (_global_repo(cfg), repo):
        if candidate is None:
            continue
        try:
            results = candidate.search_decisions(
                question=question, workload=workload, limit=limit
            )
        except Exception:  # noqa: BLE001 — never let a missing table crash MCP
            results = []
        if results:
            return [_decision_as_dict(d) for d in results]
    return []


def _decision_as_dict(d) -> dict:
    return {
        "id": d.id,
        "ts": d.ts.isoformat(),
        "project": d.project,
        "workload": d.workload_name,
        "question": d.question,
        "question_hash": d.question_hash,
        "chosen_provider": d.chosen_provider,
        "chosen_model": d.chosen_model,
        "rationale": d.rationale,
        "candidates": d.candidates,
        "constraints": d.constraints,
        "agent": d.agent,
        "superseded_by": d.superseded_by,
        "outcome_note": d.outcome_note,
    }


def _parse_models(raw: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in raw:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue
            if "/" in token:
                p, _, m = token.partition("/")
            elif ":" in token:
                p, _, m = token.partition(":")
            else:
                p, m = token, ""
            if p:
                specs.append((p, m))
    return specs

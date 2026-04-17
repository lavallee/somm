"""FastMCP server — exposes 7 tools to coding agents.

Tool catalog:
  somm_stats              — rolled-up call counts per (workload, provider, model)
  somm_search_calls       — query the call log by filters
  somm_recommend          — open recommendations + shadow-eval candidates for a workload
  somm_register_workload  — commit a workload (+ optional privacy/schemas)
  somm_register_prompt    — commit a new prompt version for a workload
  somm_compare            — run a prompt through N models side-by-side (provider-dependent)
  somm_replay             — replay a stored call against a different model (provider-dependent)

The handlers are thin — they delegate to `somm-core.Repository` /
`somm.prompts` / `somm.client`. Same surface is reused by an HTTP
transport (via `somm serve`) in D4+.

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
        return {
            "workload": workload,
            "project": cfg.project,
            "privacy_class": wl.privacy_class.value,
            "open_recommendations": recs,
            "shadow_rankings": ranked,
            "note": (
                "If no shadow_rankings: enable shadow-eval via "
                "SommLLM.enable_shadow(workload, gold_provider, gold_model) "
                "and let the worker accumulate grades."
            )
            if not ranked
            else None,
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

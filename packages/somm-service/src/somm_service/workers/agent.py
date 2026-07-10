"""Agent worker — weekly analysis that emits recommendations.

Reads calls + eval_results + model_intel; produces rows in `recommendations`
scored by evidence strength. v0.3c ships 3 recommendation types:

1. switch_model — same workload has been shadow-graded on ≥2 models; one
   scores ≥10% higher (structural or text) AND costs ≤ the current cost
   OR latency is substantially better.
2. new_model_landed — model_intel saw a model appear in the last N days
   that hits a cheaper-or-bigger-ctx profile than the workload's current
   model. Surfaces as a "try this" candidate, not an auto-switch.
3. chronic_cooldown — a provider on a workload's chain has been cooled
   for >50% of the analysis window. Recommends re-ordering or removing.
4. adaptive_param_bump — PENDING BY DEFAULT. A (workload, provider, model) keeps
   returning empty with the `stripped_empty` signature (a reasoning model
   exhausting its output budget on thinking before answering). Writes a
   recommendation by default; only workloads with policy `auto_heal: true`
   receive an immediate learned_param_override.

Outputs to `recommendations` (schema v1). Config-driven window, threshold,
notification sink (stdout only in v0.3c; webhook/SMTP in v0.3d+).
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from somm_core.repository import Repository


_log = logging.getLogger("somm.workers.agent")


@dataclass(slots=True)
class Recommendation:
    workload_id: str
    action: str  # "switch_model" | "new_model_landed" | "chronic_cooldown"
    evidence: dict
    expected_impact: str
    confidence: float  # 0..1


class AgentWorker:
    """Weekly analysis worker. Produces recommendations."""

    name = "agent"

    def __init__(
        self,
        repo: Repository,
        window_days: int = 7,
        min_calls_for_consideration: int = 10,
        min_evals_for_switch: int = 5,
        quality_threshold: float = 0.1,  # 10% improvement minimum
        notify_sink: str = "stdout",
    ) -> None:
        self.repo = repo
        self.window_days = window_days
        self.min_calls = min_calls_for_consideration
        self.min_evals = min_evals_for_switch
        self.threshold = quality_threshold
        self.notify_sink = notify_sink

    def run_once(self) -> dict:
        """Produce recommendations; write to DB; notify. Returns summary."""
        recs: list[Recommendation] = []
        recs.extend(self._rec_switch_model())
        recs.extend(self._rec_new_model_landed())
        recs.extend(self._rec_chronic_cooldown())

        # Dedup against existing non-dismissed recs (same workload + action)
        written = 0
        for rec in recs:
            if not self._already_open(rec):
                self._write(rec)
                written += 1

        adaptive = self._rec_adaptive_param_bump()
        auto_applied = 0
        for rec in adaptive:
            if self._auto_heal_enabled(rec.workload_id):
                ev = rec.evidence
                self.repo.upsert_learned_override(
                    workload_id=rec.workload_id,
                    provider=ev["provider"],
                    model=ev["model"],
                    max_tokens_floor=ev["recommended_max_tokens_floor"],
                    failure_signature=ev["failure_signature"],
                    evidence=ev,
                    confidence=rec.confidence,
                )
                self._write(rec, applied=True)
                auto_applied += 1
            elif not self._already_open(rec):
                self._write(rec)
                written += 1

        all_recs = recs + adaptive
        if written or auto_applied:
            self._notify(all_recs)
        return {
            "considered": len(all_recs),
            "written": written,
            "auto_applied": auto_applied,
            "by_action": _count_by_action(all_recs),
        }

    # ------------------------------------------------------------------
    # Rec: switch_model based on shadow-eval deltas

    def _rec_switch_model(self) -> list[Recommendation]:
        """For each workload: if shadow evals show a graded model scores
        materially higher than the production model, recommend a switch.
        """
        out: list[Recommendation] = []
        since = _iso(self._window_start())

        with self.repo._open() as conn:
            # Per (workload_id, prod model/provider): aggregate eval scores
            # (structural_score and embedding_score are both 0..1-ish).
            prod_rows = conn.execute(
                """
                SELECT c.workload_id, c.provider, c.model,
                       AVG(COALESCE(er.structural_score, er.embedding_score)) AS score,
                       COUNT(er.id) AS n_evals,
                       AVG(c.latency_ms) AS latency_ms,
                       AVG(c.cost_usd)   AS cost_usd,
                       w.name
                FROM eval_results er
                JOIN calls c ON c.id = er.call_id
                JOIN workloads w ON w.id = c.workload_id
                WHERE er.judge_score IS NOT NULL OR er.structural_score IS NOT NULL
                   OR er.embedding_score IS NOT NULL
                GROUP BY c.workload_id, c.provider, c.model
                HAVING COUNT(er.id) >= ?
                """,
                (self.min_evals,),
            ).fetchall()

        by_workload: dict[str, list[dict]] = {}
        for r in prod_rows:
            by_workload.setdefault(r[0], []).append(
                {
                    "provider": r[1],
                    "model": r[2],
                    "score": r[3] or 0,
                    "n_evals": r[4],
                    "latency_ms": r[5] or 0,
                    "cost_usd": r[6] or 0,
                    "workload_name": r[7],
                }
            )

        for workload_id, entries in by_workload.items():
            if len(entries) < 2:
                continue
            # Sort by score desc
            entries.sort(key=lambda e: e["score"], reverse=True)
            best = entries[0]
            # Current production = most-used model (first call shows up most in
            # production). We approximate "current prod" as the model with most
            # calls in window_days.
            with self.repo._open() as conn:
                row = conn.execute(
                    """
                    SELECT provider, model FROM calls
                    WHERE workload_id = ? AND ts >= ?
                    GROUP BY provider, model ORDER BY COUNT(*) DESC LIMIT 1
                    """,
                    (workload_id, since),
                ).fetchone()
            if not row:
                continue
            current_provider, current_model = row
            current = next(
                (
                    e
                    for e in entries
                    if e["provider"] == current_provider and e["model"] == current_model
                ),
                None,
            )
            if (
                current is None
                or best["provider"] == current_provider
                and best["model"] == current_model
            ):
                continue

            score_delta = best["score"] - current["score"]
            if score_delta < self.threshold:
                continue
            # Only recommend if the new model is at least as cheap OR at least
            # 20% faster (avoid "better but 5x more expensive").
            cheaper = best["cost_usd"] <= current["cost_usd"] * 1.01
            faster = best["latency_ms"] < current["latency_ms"] * 0.8
            if not (cheaper or faster):
                continue

            out.append(
                Recommendation(
                    workload_id=workload_id,
                    action="switch_model",
                    evidence={
                        "workload": current["workload_name"],
                        "current": {
                            "provider": current_provider,
                            "model": current_model,
                            "score": round(current["score"], 3),
                            "cost_usd": round(current["cost_usd"], 6),
                            "latency_ms": round(current["latency_ms"]),
                        },
                        "candidate": {
                            "provider": best["provider"],
                            "model": best["model"],
                            "score": round(best["score"], 3),
                            "cost_usd": round(best["cost_usd"], 6),
                            "latency_ms": round(best["latency_ms"]),
                        },
                        "score_delta": round(score_delta, 3),
                        "n_evals": best["n_evals"],
                    },
                    expected_impact=_impact_str(current, best),
                    confidence=min(0.99, 0.5 + score_delta + (best["n_evals"] / 50)),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Rec: new model landed (from model_intel)

    def _rec_new_model_landed(self) -> list[Recommendation]:
        """If a new model appeared in model_intel within the window that's
        cheaper than the workload's current model (same provider family), flag it."""
        out: list[Recommendation] = []
        since = _iso(self._window_start())

        with self.repo._open() as conn:
            workloads = conn.execute(
                """
                SELECT DISTINCT c.workload_id, c.provider, c.model, w.name
                FROM calls c JOIN workloads w ON w.id = c.workload_id
                WHERE c.ts >= ?
                GROUP BY c.workload_id, c.provider, c.model
                HAVING COUNT(*) >= ?
                """,
                (since, self.min_calls),
            ).fetchall()

            for wl_id, provider, model, name in workloads:
                current = conn.execute(
                    "SELECT price_in_per_1m, price_out_per_1m, context_window "
                    "FROM model_intel WHERE provider = ? AND model = ?",
                    (provider, model),
                ).fetchone()
                if not current:
                    continue
                cur_in, cur_out = current[0] or 0, current[1] or 0
                if cur_in == 0 and cur_out == 0:
                    continue  # can't beat free; skip

                # Candidates: same provider, appeared recently, cheaper.
                # Require both prices present and non-negative so that
                # dynamic-pricing sentinels (OpenRouter's "-1" on meta-models
                # like openrouter/auto) don't rank as infinitely cheap.
                cands = conn.execute(
                    """
                    SELECT model, price_in_per_1m, price_out_per_1m, context_window, last_seen
                    FROM model_intel WHERE provider = ? AND model != ?
                      AND last_seen >= ?
                      AND price_in_per_1m IS NOT NULL AND price_out_per_1m IS NOT NULL
                      AND price_in_per_1m >= 0 AND price_out_per_1m >= 0
                      AND price_in_per_1m + price_out_per_1m < ?
                    ORDER BY (price_in_per_1m + price_out_per_1m) ASC LIMIT 1
                    """,
                    (provider, model, since, (cur_in + cur_out) * 0.8),
                ).fetchone()
                if not cands:
                    continue
                cand_model, cand_in, cand_out, cand_ctx, last_seen = cands
                out.append(
                    Recommendation(
                        workload_id=wl_id,
                        action="new_model_landed",
                        evidence={
                            "workload": name,
                            "current": {
                                "provider": provider,
                                "model": model,
                                "price_in_per_1m": cur_in,
                                "price_out_per_1m": cur_out,
                            },
                            "candidate": {
                                "provider": provider,
                                "model": cand_model,
                                "price_in_per_1m": cand_in,
                                "price_out_per_1m": cand_out,
                                "context_window": cand_ctx,
                                "last_seen": last_seen,
                            },
                        },
                        expected_impact=(
                            f"~{100 * (1 - (cand_in + cand_out) / max(0.001, cur_in + cur_out)):.0f}% "
                            f"lower cost; quality untested — run `somm compare <prompt> --models ...` to verify"
                        ),
                        confidence=0.4,  # untested quality → low confidence
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Rec: chronic cooldown

    def _rec_chronic_cooldown(self) -> list[Recommendation]:
        out: list[Recommendation] = []
        since = _iso(self._window_start())

        with self.repo._open() as conn:
            # Providers that have been cooling for >half the window on
            # workloads with >min_calls calls. Approximation: look at
            # current cooldown_until + consecutive_failures.
            rows = conn.execute(
                """
                SELECT ph.provider, ph.consecutive_failures,
                       ph.cooldown_until, ph.last_ok_at
                FROM provider_health ph
                WHERE ph.consecutive_failures >= 5
                """,
            ).fetchall()
            if not rows:
                return out
            # Find workloads that have called any of those providers a lot
            chronic_providers = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in chronic_providers)
            wls = conn.execute(
                f"""
                SELECT c.workload_id, w.name, c.provider, COUNT(*) AS n
                FROM calls c JOIN workloads w ON w.id = c.workload_id
                WHERE c.provider IN ({placeholders}) AND c.ts >= ?
                GROUP BY c.workload_id, c.provider
                HAVING COUNT(*) >= ?
                """,
                (*chronic_providers, since, self.min_calls),
            ).fetchall()

        for wl_id, name, provider, n in wls:
            out.append(
                Recommendation(
                    workload_id=wl_id,
                    action="chronic_cooldown",
                    evidence={
                        "workload": name,
                        "provider": provider,
                        "n_calls": n,
                        "note": f"provider {provider!r} has hit circuit-break threshold — "
                        "consider adding a fallback provider or reordering the chain",
                    },
                    expected_impact="reduce latency + exhausted-call rate on this workload",
                    confidence=0.7,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Rec: adaptive_param_bump — controlled self-healing for recurring capability failures.
    # Detects the `stripped_empty` signature (a reasoning model exhausting its
    # output budget on thinking tokens and returning empty) and prescribes a
    # higher max_tokens floor for that (workload, provider, model). run_once()
    # auto-applies only when the workload policy opts into auto_heal.

    def _rec_adaptive_param_bump(self) -> list[Recommendation]:
        out: list[Recommendation] = []
        candidates = self.repo.detect_capability_overflow(
            window_days=self.window_days,
            min_calls=self.min_calls,
        )
        for c in candidates:
            floor = c["recommended_max_tokens_floor"]
            # Skip if a learned override is already at/above this floor — avoids
            # rewriting an identical heal on every run.
            existing = self.repo.lookup_learned_override(
                c["workload_id"], c["model"], c["provider"]
            )
            if existing and (existing.get("max_tokens_floor") or 0) >= floor:
                continue
            out.append(
                Recommendation(
                    workload_id=c["workload_id"],
                    action="adaptive_param_bump",
                    evidence={
                        "workload": c["workload_name"],
                        "provider": c["provider"],
                        "model": c["model"],
                        "failure_signature": "capability_empty:stripped_empty",
                        "empty_rate": c["empty_rate"],
                        "n_calls": c["n_calls"],
                        "n_empty": c["n_empty"],
                        "n_stripped": c["n_stripped"],
                        "current_max_tokens": c["max_tokens_req"],
                        "recommended_max_tokens_floor": floor,
                    },
                    expected_impact=(
                        f"raise max_tokens floor to {floor} for "
                        f"{c['provider']}/{c['model']} on {c['workload_name']!r} — gives the "
                        f"model budget to finish, cutting the {round(c['empty_rate'] * 100)}% "
                        f"empty rate"
                    ),
                    confidence=c["confidence"],
                )
            )
        return out

    # ------------------------------------------------------------------

    def _already_open(self, rec: Recommendation) -> bool:
        with self.repo._open() as conn:
            row = conn.execute(
                "SELECT id FROM recommendations "
                "WHERE workload_id = ? AND action = ? AND dismissed_at IS NULL "
                "AND applied_at IS NULL "
                "ORDER BY created_at DESC LIMIT 1",
                (rec.workload_id, rec.action),
            ).fetchone()
        return row is not None

    def _write(self, rec: Recommendation, applied: bool = False) -> None:
        cols = "(workload_id, action, evidence_json, expected_impact, confidence"
        vals = "(?, ?, ?, ?, ?"
        params = [
            rec.workload_id,
            rec.action,
            json.dumps(rec.evidence),
            rec.expected_impact,
            rec.confidence,
        ]
        if applied:
            # Auto-applied self-heal: stamp applied_at so the recommendation
            # surfaces as already-done, not as an open suggestion.
            cols += ", applied_at"
            vals += ", CURRENT_TIMESTAMP"
        cols += ")"
        vals += ")"
        with self.repo._open() as conn:
            conn.execute(
                f"INSERT INTO recommendations {cols} VALUES {vals}",
                tuple(params),
            )

    def _auto_heal_enabled(self, workload_id: str) -> bool:
        with self.repo._open() as conn:
            row = conn.execute(
                "SELECT policy_json FROM workloads WHERE id = ?",
                (workload_id,),
            ).fetchone()
        if row is None or not row[0]:
            return False
        try:
            policy = json.loads(row[0])
        except json.JSONDecodeError:
            return False
        return bool(isinstance(policy, dict) and policy.get("auto_heal") is True)

    def _notify(self, recs: list[Recommendation]) -> None:
        if self.notify_sink == "stdout":
            print("\n=== somm agent: new recommendations ===")
            for r in recs:
                name = r.evidence.get("workload", r.workload_id[:10])
                print(f"  [{r.action}] {name}: {r.expected_impact} (confidence={r.confidence:.2f})")
            print("")

    def _window_start(self) -> datetime:
        return datetime.now(UTC) - timedelta(days=self.window_days)


# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _impact_str(current: dict, best: dict) -> str:
    parts = []
    score_delta = best["score"] - current["score"]
    parts.append(f"+{round(100 * score_delta)}% quality")
    if current["cost_usd"] > 0:
        cost_pct = 100 * (1 - best["cost_usd"] / current["cost_usd"])
        if cost_pct > 1:
            parts.append(f"-{round(cost_pct)}% cost")
    if current["latency_ms"] > 0:
        lat_pct = 100 * (1 - best["latency_ms"] / current["latency_ms"])
        if lat_pct > 1:
            parts.append(f"-{round(lat_pct)}% latency")
    return ", ".join(parts) if parts else "improvement observed"


def _count_by_action(recs: list[Recommendation]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        out[r.action] = out.get(r.action, 0) + 1
    return out


def median_or_zero(xs) -> float:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0

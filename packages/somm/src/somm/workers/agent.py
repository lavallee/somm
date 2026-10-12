"""Agent worker — weekly analysis that emits recommendations.

Reads calls + eval_results + model_intel; produces rows in `recommendations`
scored by evidence strength. v0.3c ships 3 recommendation types:

1. switch_model — same workload has been shadow-graded on ≥2 models; one
   scores ≥10% higher (structural or text) AND costs ≤ the current cost
   OR latency is substantially better.
2. new_model_landed — a same-provider model whose model_intel.first_seen is
   inside the window and later than the current model's, at least 20%
   cheaper. Surfaces as a "try this" candidate, not an auto-switch, and is
   proposed once per workload: dismissed is remembered.
3. chronic_cooldown — at least `chronic_failure_rate` of a workload's
   production calls to a provider in the window failed to deliver (upstream
   error, exhausted chain, timeout, rate limit). Recommends a fallback,
   re-ordering, or pruning dead roster models.
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


DELIVERY_FAILURE_OUTCOMES: tuple[str, ...] = ("upstream_error", "exhausted", "timeout", "rate_limit")
"""Outcomes that mean the provider did not deliver, as opposed to delivering badly."""


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
        chronic_failure_rate: float = 0.10,  # share of a workload's calls to a provider that fail
        notify_sink: str = "stdout",
    ) -> None:
        self.repo = repo
        self.window_days = window_days
        self.min_calls = min_calls_for_consideration
        self.min_evals = min_evals_for_switch
        self.threshold = quality_threshold
        self.chronic_failure_rate = chronic_failure_rate
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
                WHERE (
                    er.judge_score IS NOT NULL OR er.structural_score IS NOT NULL
                    OR er.embedding_score IS NOT NULL
                )
                  AND c.observation_role = 'production'
                  AND c.budget_eligible != 0
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
                      AND observation_role = 'production'
                      AND budget_eligible != 0
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
        """A cheaper model from the same provider landed after the one in use.

        "Landed" means ``model_intel.first_seen`` falls inside the window and
        is later than the current model's own ``first_seen``. Until 0.17.2
        this read ``last_seen``, which every intel pass refreshes, so the rule
        proposed every cheaper sibling already in the catalog — older
        generations included — once per workload, forever. A landing is also
        proposed once: a candidate already open, dismissed, or applied for the
        workload is not raised again.
        """
        out: list[Recommendation] = []
        since = _iso(self._window_start())

        with self.repo._open() as conn:
            workloads = conn.execute(
                """
                SELECT DISTINCT c.workload_id, c.provider, c.model, w.name
                FROM calls c JOIN workloads w ON w.id = c.workload_id
                WHERE c.ts >= ?
                  AND c.observation_role = 'production'
                  AND c.budget_eligible != 0
                GROUP BY c.workload_id, c.provider, c.model
                HAVING COUNT(*) >= ?
                """,
                (since, self.min_calls),
            ).fetchall()

            for wl_id, provider, model, name in workloads:
                current = conn.execute(
                    "SELECT price_in_per_1m, price_out_per_1m, context_window, first_seen "
                    "FROM model_intel WHERE provider = ? AND model = ?",
                    (provider, model),
                ).fetchone()
                if not current:
                    continue
                cur_in, cur_out = current[0] or 0, current[1] or 0
                cur_first_seen = current[3]
                if cur_in == 0 and cur_out == 0:
                    continue  # can't beat free; skip

                # Candidates: same provider, landed in the window and after the
                # current model, at least 20% cheaper. Both prices must be
                # present and non-negative so dynamic-pricing sentinels
                # (OpenRouter's "-1" on meta-models) don't rank as free.
                cands = conn.execute(
                    """
                    SELECT model, price_in_per_1m, price_out_per_1m, context_window,
                           first_seen, last_seen
                    FROM model_intel WHERE provider = ? AND model != ?
                      AND first_seen IS NOT NULL AND first_seen >= ?
                      AND (? IS NULL OR first_seen > ?)
                      AND price_in_per_1m IS NOT NULL AND price_out_per_1m IS NOT NULL
                      AND price_in_per_1m >= 0 AND price_out_per_1m >= 0
                      AND price_in_per_1m + price_out_per_1m < ?
                    ORDER BY (price_in_per_1m + price_out_per_1m) ASC
                    """,
                    (provider, model, since, cur_first_seen, cur_first_seen, (cur_in + cur_out) * 0.8),
                ).fetchall()
                chosen = None
                for cand in cands:
                    if not self._already_proposed(wl_id, "new_model_landed", cand[0]):
                        chosen = cand
                        break
                if chosen is None:
                    continue
                cand_model, cand_in, cand_out, cand_ctx, cand_first_seen, last_seen = chosen
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
                                "first_seen": cur_first_seen,
                            },
                            "candidate": {
                                "provider": provider,
                                "model": cand_model,
                                "price_in_per_1m": cand_in,
                                "price_out_per_1m": cand_out,
                                "context_window": cand_ctx,
                                "first_seen": cand_first_seen,
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
        """A provider that keeps failing to deliver for a workload.

        Answered from the calls in the window: of this workload's production
        calls to this provider, what share ended in a delivery failure
        (upstream error, exhausted chain, timeout, rate limit)? At or above
        ``chronic_failure_rate`` with at least ``min_calls`` calls, it is
        chronic. Until 0.17.2 this read ``provider_health.consecutive_failures``
        and indicted every workload that had ever used a provider with one
        bad entry, so a roster model dead since spring kept flagging workloads
        the provider was serving fine, and the flags never cleared.
        """
        out: list[Recommendation] = []
        since = _iso(self._window_start())
        failed = ",".join("?" for _ in DELIVERY_FAILURE_OUTCOMES)

        with self.repo._open() as conn:
            rows = conn.execute(
                f"""
                SELECT c.workload_id, w.name, c.provider,
                       COUNT(*) AS n,
                       SUM(CASE WHEN c.outcome IN ({failed}) THEN 1 ELSE 0 END) AS n_failed
                FROM calls c JOIN workloads w ON w.id = c.workload_id
                WHERE c.ts >= ? AND c.provider != ''
                  AND c.observation_role = 'production'
                  AND c.budget_eligible != 0
                GROUP BY c.workload_id, c.provider
                HAVING COUNT(*) >= ? AND n_failed * 1.0 / COUNT(*) >= ?
                """,
                (*DELIVERY_FAILURE_OUTCOMES, since, self.min_calls, self.chronic_failure_rate),
            ).fetchall()
            cooling = conn.execute(
                "SELECT provider, model, consecutive_failures FROM provider_health "
                "WHERE cooldown_until > ?",
                (_iso(datetime.now(UTC)),),
            ).fetchall()

        cooling_by_provider: dict[str, list[dict]] = {}
        for provider, model, failures in cooling:
            cooling_by_provider.setdefault(provider, []).append(
                {"model": model or None, "consecutive_failures": failures}
            )

        for wl_id, name, provider, n, n_failed in rows:
            rate = n_failed / n
            out.append(
                Recommendation(
                    workload_id=wl_id,
                    action="chronic_cooldown",
                    evidence={
                        "workload": name,
                        "provider": provider,
                        "n_calls": n,
                        "n_failed": n_failed,
                        "failure_rate": round(rate, 3),
                        "window_days": self.window_days,
                        "cooling": cooling_by_provider.get(provider, []),
                        "note": (
                            f"{n_failed} of {n} calls to {provider!r} in the last "
                            f"{self.window_days} days failed to deliver — add a fallback "
                            "provider, reorder the chain, or drop dead roster models"
                        ),
                    },
                    expected_impact=(
                        f"recover the ~{100 * rate:.0f}% of this workload's calls to "
                        f"{provider!r} that currently fail"
                    ),
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

    def _already_proposed(self, workload_id: str, action: str, candidate_model: str) -> bool:
        """Open, dismissed, or applied: a landing is proposed once per workload."""
        with self.repo._open() as conn:
            row = conn.execute(
                "SELECT 1 FROM recommendations WHERE workload_id = ? AND action = ? "
                "AND json_extract(evidence_json, '$.candidate.model') = ? LIMIT 1",
                (workload_id, action, candidate_model),
            ).fetchone()
        return row is not None

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

"""Shadow-eval worker — re-runs sampled production calls through a gold model
to build a real-world quality signal.

Design:
- OFF by default. Per-workload opt-in via `repo.set_shadow_config(...)`.
- privacy_class=private workloads are BANNED from shadow-eval — the gate is
  enforced by the `shadow_candidates` view (schema v2) AND defense-in-depth
  here in Python.
- Budget ceiling: per-workload `budget_usd_daily`. Worker sums the cost_usd
  of today's eval_results rows and stops sampling that workload when it
  hits the cap.
- Graders (v0.3b): structural (JSON field overlap) + text similarity
  (word-bigram Jaccard). Judge model is opt-in + deferred to v0.3b+1.
- Crash-safe: each eval_results row has `grading_started_at` lease.
  Worker picks rows where it's NULL or expired >10 min.

Shadow config shape (stored as JSON on workloads.shadow_config_json):

    {
        "gold_provider": "openrouter",
        "gold_model":    "google/gemma-3-27b-it:free",
        "sample_rate":   0.02,
        "budget_usd_daily": 2.0,
        "max_grades_per_run": 20,
        "min_days_between_regrades": 14
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from somm_core.graders import (
    build_binary_judge_prompt,
    grade_response_pair,
    normalize_binary_criteria,
    parse_binary_judge_response,
)

if TYPE_CHECKING:
    from somm.providers.base import SommProvider
    from somm_core.repository import Repository


_log = logging.getLogger("somm.workers.shadow_eval")


@dataclass(slots=True)
class ShadowConfig:
    gold_provider: str
    gold_model: str
    sample_rate: float = 0.02
    budget_usd_daily: float = 1.0
    max_grades_per_run: int = 20
    min_days_between_regrades: int = 14
    judge: dict | None = None  # {"provider": ..., "model": ...} — v0.3b+1

    @classmethod
    def from_dict(cls, data: dict) -> ShadowConfig:
        return cls(
            gold_provider=data["gold_provider"],
            gold_model=data["gold_model"],
            sample_rate=float(data.get("sample_rate", 0.02)),
            budget_usd_daily=float(data.get("budget_usd_daily", 1.0)),
            max_grades_per_run=int(data.get("max_grades_per_run", 20)),
            min_days_between_regrades=int(data.get("min_days_between_regrades", 14)),
            judge=data.get("judge"),
        )


@dataclass(slots=True)
class EvalOutcome:
    call_id: str
    structural_score: float | None
    text_similarity_score: float | None
    judge_score: float | None
    gold_response_hash: str | None
    judge_reason: dict | None = None
    extra_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


class ShadowEvalWorker:
    """Sampled shadow-eval grader. Writes eval_results rows."""

    name = "shadow_eval"

    def __init__(
        self,
        repo: Repository,
        providers: list[SommProvider],
        lease_window_s: int = 600,
    ) -> None:
        self.repo = repo
        self.providers = {p.name: p for p in providers}
        self.lease_window_s = lease_window_s

    def run_once(self) -> dict:
        """Grade a batch of un-graded sampled calls. Returns a summary dict."""
        summary: dict = {
            "workloads_considered": 0,
            "calls_graded": 0,
            "budget_skipped": 0,
            "private_skipped": 0,
            "errors": [],
        }

        swept = self._sweep_stale_leases()
        if swept:
            _log.info("shadow: swept %d stale grading lease(s)", swept)

        candidates = self._fetch_candidates()
        by_workload: dict[str, list[dict]] = {}
        for c in candidates:
            by_workload.setdefault(c["workload_id"], []).append(c)

        for workload_id, calls in by_workload.items():
            summary["workloads_considered"] += 1
            try:
                cfg_raw = self.repo.get_shadow_config(workload_id)
                if not cfg_raw:
                    continue
                cfg = ShadowConfig.from_dict(cfg_raw)

                # Privacy defense-in-depth (view already filters, but double check).
                # Skipped here silently — SOMM_PRIVACY_VIOLATION is raised in the
                # router when a private workload is called upstream.

                # Budget check
                spent = self._shadow_spent_today(workload_id)
                remaining = cfg.budget_usd_daily - spent
                if remaining <= 0:
                    summary["budget_skipped"] += len(calls)
                    _log.info("shadow: workload %s over budget ($%.4f)", workload_id, spent)
                    continue

                # Candidates already passed the capture-time sample gate
                # (rate applied once, in the library) — here we only cap
                # the per-run batch, deterministically ordered.
                sampled = _deterministic_sample(calls, 1.0, cfg.max_grades_per_run)
                for call_row in sampled:
                    if not self._claim_lease(call_row["call_id"]):
                        continue
                    try:
                        outcome = self._grade_call(call_row, cfg)
                    except Exception as e:
                        _log.warning("shadow: grade failed for %s: %s", call_row["call_id"], e)
                        summary["errors"].append(f"{call_row['call_id']}: {e}")
                        self._release_lease(call_row["call_id"])
                        continue
                    self._write_result(call_row, cfg, outcome)
                    summary["calls_graded"] += 1
                    spent = self._shadow_spent_today(workload_id)
                    if spent >= cfg.budget_usd_daily:
                        break
            except Exception as e:
                _log.warning("shadow: workload %s failed: %s", workload_id, e)
                summary["errors"].append(f"{workload_id}: {e}")

        return summary

    # ------------------------------------------------------------------
    # Candidate sourcing

    def _fetch_candidates(self) -> list[dict]:
        # Only calls with captured bodies are gradable: sample_rate is
        # applied ONCE, at capture time in the library (0.7.0+). Historical
        # candidates that predate body capture are simply never considered
        # rather than churned into "samples not captured" results.
        with self.repo._open() as conn:
            rows = conn.execute(
                "SELECT call_id, ts, project, workload_id, provider, model, "
                "prompt_hash, response_hash, workload_name, privacy_class, "
                "shadow_config_json FROM shadow_candidates sc "
                "WHERE EXISTS (SELECT 1 FROM samples s WHERE s.call_id = sc.call_id) "
                "ORDER BY ts DESC LIMIT 500"
            ).fetchall()
        return [
            {
                "call_id": r[0],
                "ts": r[1],
                "project": r[2],
                "workload_id": r[3],
                "provider": r[4],
                "model": r[5],
                "prompt_hash": r[6],
                "response_hash": r[7],
                "workload_name": r[8],
                "privacy_class": r[9],
                "shadow_config_json": r[10],
            }
            for r in rows
        ]

    def _shadow_spent_today(self, workload_id: str) -> float:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self.repo._open() as conn:
            # Shadow costs are recorded as the gold-model call's cost on the
            # source call's cost_usd? No — shadow calls land as new rows with
            # workload_id = <some shadow tag>. Simpler: track cost in the
            # eval_results row directly via a column? For v0.3b, approximate:
            # cost of gold call = (tokens_in + tokens_out) * price / 1M —
            # but we store it inline on eval_results as judge_score's sibling
            # would be awkward. Cleanest: add a shadow_calls tag to the cost
            # column in a D3b+1 patch. For now, compute from tokens we stored
            # on the eval_result itself via the notes field.
            row = conn.execute(
                """
                SELECT COALESCE(SUM(
                    json_extract(notes.value, '$.cost_usd')
                ), 0)
                FROM eval_results er, json_each(COALESCE(er.judge_reason, '[]')) notes
                WHERE er.call_id IN (
                    SELECT id FROM calls WHERE workload_id = ? AND date(ts) = ?
                )
                """,
                (workload_id, today),
            ).fetchone()
        try:
            return float(row[0] or 0.0)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Lease + write

    def _claim_lease(self, call_id: str) -> bool:
        """Atomic lease acquisition. Returns True if acquired."""
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=self.lease_window_s)
        with self.repo._open() as conn:
            # First check: is there an existing completed eval_result?
            existing = conn.execute(
                "SELECT id, grading_started_at, judge_score FROM eval_results WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            if existing:
                eval_id, started_at, judge = existing
                if judge is not None:
                    return False  # already graded
                # existing in-flight — check lease
                started = _parse_ts(started_at) if started_at else None
                if started and started > stale_before:
                    return False
                conn.execute(
                    "UPDATE eval_results SET grading_started_at = ? WHERE id = ?",
                    (now.isoformat(), eval_id),
                )
                return True
            # No existing row — insert a lease placeholder
            conn.execute(
                "INSERT INTO eval_results (call_id, gold_model, grading_started_at) "
                "VALUES (?, ?, ?)",
                (call_id, "pending", now.isoformat()),
            )
        return True

    def _release_lease(self, call_id: str) -> None:
        """Delete the lease placeholder outright.

        The shadow_candidates view excludes calls with ANY eval_results
        row, so a kept-but-released lease would permanently orphan the
        captured sample — a transient gold failure (429, timeout) must
        put the candidate back in the pool, not bury it."""
        with self.repo._open() as conn:
            conn.execute(
                "DELETE FROM eval_results WHERE call_id = ? "
                "AND gold_model = 'pending' AND structural_score IS NULL "
                "AND judge_score IS NULL",
                (call_id,),
            )

    def _sweep_stale_leases(self) -> int:
        """Self-heal leases from workers that died mid-grade: a 'pending'
        placeholder past the lease window is deleted so its candidate
        resurfaces. Returns rows swept."""
        stale_before = datetime.now(UTC) - timedelta(seconds=self.lease_window_s)
        with self.repo._open() as conn:
            cur = conn.execute(
                "DELETE FROM eval_results WHERE gold_model = 'pending' "
                "AND structural_score IS NULL AND judge_score IS NULL "
                "AND (grading_started_at IS NULL OR grading_started_at < ?)",
                (stale_before.isoformat(),),
            )
            return cur.rowcount

    def _write_result(self, call_row: dict, cfg: ShadowConfig, outcome: EvalOutcome) -> None:
        # Repurpose judge_reason as a JSON array with per-signal metadata; the
        # `cost_usd` used for budget accounting lives inside it. Not beautiful
        # but avoids another migration for v0.3b.
        notes = [
            {
                "cost_usd": _latest_cost(self.repo, call_row["call_id"], cfg)
                + outcome.extra_cost_usd
            }
        ]
        notes.extend({"note": n} for n in outcome.notes)
        if outcome.judge_reason is not None:
            notes.append({"judge": outcome.judge_reason})
        eval_result_id = None
        with self.repo._open() as conn:
            conn.execute(
                """
                UPDATE eval_results SET
                    gold_model = ?,
                    gold_response_hash = ?,
                    structural_score = ?,
                    embedding_score = ?,
                    judge_score = ?,
                    judge_reason = ?,
                    grading_started_at = NULL,
                    ts = CURRENT_TIMESTAMP
                WHERE call_id = ?
                """,
                (
                    cfg.gold_model,
                    outcome.gold_response_hash,
                    outcome.structural_score,
                    outcome.text_similarity_score,  # embedding_score column reused
                    outcome.judge_score,
                    json.dumps(notes),
                    call_row["call_id"],
                ),
            )
            row = conn.execute(
                "SELECT id FROM eval_results WHERE call_id = ?",
                (call_row["call_id"],),
            ).fetchone()
            eval_result_id = int(row[0]) if row else None
        if eval_result_id is not None:
            score = (
                outcome.judge_score
                if outcome.judge_score is not None
                else outcome.structural_score
                if outcome.structural_score is not None
                else outcome.text_similarity_score
            )
            self.repo.record_eval_receipt(
                receipt_type="shadow_eval",
                eval_result_id=eval_result_id,
                call_id=call_row["call_id"],
                score=score,
                payload={
                    "source": "shadow_eval",
                    "call_id": call_row["call_id"],
                    "gold_provider": cfg.gold_provider,
                    "gold_model": cfg.gold_model,
                    "gold_response_hash": outcome.gold_response_hash,
                    "structural_score": outcome.structural_score,
                    "text_similarity_score": outcome.text_similarity_score,
                    "judge_score": outcome.judge_score,
                    "judge": outcome.judge_reason,
                    "notes": notes,
                },
            )

    # ------------------------------------------------------------------
    # Grading

    def _grade_call(self, call_row: dict, cfg: ShadowConfig) -> EvalOutcome:
        from somm.providers.base import SommRequest

        # Fetch the original prompt body + response text — we stored hashes,
        # so we need prompt/response from the samples table IF it was captured.
        prompt, response = self._fetch_bodies(call_row["call_id"])
        if prompt is None or response is None:
            return EvalOutcome(
                call_id=call_row["call_id"],
                structural_score=None,
                text_similarity_score=None,
                judge_score=None,
                gold_response_hash=None,
                notes=["samples not captured — shadow needs per-workload sample opt-in"],
            )

        provider = self.providers.get(cfg.gold_provider)
        if provider is None:
            return EvalOutcome(
                call_id=call_row["call_id"],
                structural_score=None,
                text_similarity_score=None,
                judge_score=None,
                gold_response_hash=None,
                notes=[f"gold_provider {cfg.gold_provider!r} not configured"],
            )
        gold = provider.generate(
            SommRequest(prompt=prompt, model=cfg.gold_model, temperature=0.0, max_tokens=1024)
        )
        gold_text = gold.text

        scores = grade_response_pair(response, gold_text, judge=cfg.judge)
        judge_score = scores.judge_score
        judge_reason = None
        judge_cost_usd = 0.0
        judge_notes: list[str] = []
        if cfg.judge:
            judge_score, judge_reason, judge_cost_usd, judge_notes = self._run_judge(
                original_prompt=prompt,
                production_text=response,
                gold_text=gold_text,
                judge_cfg=cfg.judge,
            )

        from somm_core.parse import stable_hash

        return EvalOutcome(
            call_id=call_row["call_id"],
            structural_score=scores.structural_score,
            text_similarity_score=scores.text_similarity_score,
            judge_score=judge_score,
            gold_response_hash=stable_hash(gold_text),
            judge_reason=judge_reason,
            extra_cost_usd=judge_cost_usd,
            notes=[
                f"gold_tokens_in={gold.tokens_in}",
                f"gold_tokens_out={gold.tokens_out}",
                *scores.notes,
                *judge_notes,
            ],
        )

    def _fetch_bodies(self, call_id: str) -> tuple[str | None, str | None]:
        with self.repo._open() as conn:
            row = conn.execute(
                "SELECT prompt_body, response_body FROM samples WHERE call_id = ?",
                (call_id,),
            ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def _run_judge(
        self,
        *,
        original_prompt: str,
        production_text: str,
        gold_text: str,
        judge_cfg: dict,
    ) -> tuple[float | None, dict | None, float, list[str]]:
        from somm.providers.base import SommRequest
        from somm_core.pricing import cost_for_call

        try:
            criteria = normalize_binary_criteria(judge_cfg.get("criteria"))
            specs = _judge_specs(judge_cfg)
        except ValueError as exc:
            return None, {"error": str(exc)}, 0.0, [f"judge_config_error={exc}"]

        prompt = build_binary_judge_prompt(
            original_prompt=original_prompt,
            production_text=production_text,
            gold_text=gold_text,
            criteria=criteria,
        )
        receipts: list[dict[str, Any]] = []
        cost_usd = 0.0
        notes: list[str] = []
        for spec in specs:
            provider_name = spec.get("provider")
            model = spec.get("model")
            if not provider_name or not model:
                receipts.append({"error": "judge provider and model are required", "spec": spec})
                continue
            provider = self.providers.get(str(provider_name))
            if provider is None:
                receipts.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": f"judge_provider {provider_name!r} not configured",
                    }
                )
                continue
            try:
                response = provider.generate(
                    SommRequest(
                        prompt=prompt,
                        model=str(model),
                        temperature=0.0,
                        max_tokens=int(spec.get("max_tokens") or judge_cfg.get("max_tokens") or 512),
                    )
                )
                parsed = parse_binary_judge_response(response.text, criteria)
                cost_usd += cost_for_call(
                    self.repo,
                    str(provider_name),
                    str(model),
                    response.tokens_in,
                    response.tokens_out,
                )
                receipts.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                        "result": parsed,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — a judge failure should not drop base grades
                notes.append(f"judge_error={type(exc).__name__}: {exc}")
                receipts.append(
                    {
                        "provider": provider_name,
                        "model": model,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        usable = [r for r in receipts if isinstance(r.get("result"), dict)]
        if not usable:
            return None, {"criteria": [c.name for c in criteria], "judges": receipts}, cost_usd, notes

        aggregate = _aggregate_judge_votes(criteria, usable)
        aggregate["mode"] = "panel" if len(usable) > 1 else "single"
        aggregate["judges"] = receipts
        return float(aggregate["score"]), aggregate, cost_usd, notes


# ---------------------------------------------------------------------------
# Helpers


def _deterministic_sample(calls: list[dict], rate: float, cap: int) -> list[dict]:
    """Pick the first N calls where hash(call_id) < rate * MAX_HASH. Stable."""
    import hashlib

    if rate >= 1.0:
        return calls[:cap]
    if rate <= 0:
        return []
    threshold = int(rate * (2**32))
    picked = []
    for c in calls:
        h = int.from_bytes(hashlib.sha256(c["call_id"].encode()).digest()[:4], "big")
        if h < threshold:
            picked.append(c)
            if len(picked) >= cap:
                break
    return picked


def _judge_specs(judge_cfg: dict) -> list[dict]:
    """Return one or more judge model specs.

    Supported shapes:
      {"provider": "openrouter", "model": "..."}
      {"panel": [{"provider": "ollama", "model": "a"}, ...]}

    A panel lets cheap judges vote before the user reaches for a frontier
    judge. Each panel item inherits top-level max_tokens when unset.
    """

    panel = judge_cfg.get("panel") or judge_cfg.get("ensemble")
    if panel is None:
        return [
            {
                "provider": judge_cfg.get("provider"),
                "model": judge_cfg.get("model"),
                "max_tokens": judge_cfg.get("max_tokens"),
            }
        ]
    if not isinstance(panel, list) or not panel:
        raise ValueError("judge panel must be a non-empty list")
    out: list[dict] = []
    for idx, item in enumerate(panel):
        if not isinstance(item, dict):
            raise ValueError(f"judge panel[{idx}] must be an object")
        spec = dict(item)
        if "max_tokens" not in spec and "max_tokens" in judge_cfg:
            spec["max_tokens"] = judge_cfg["max_tokens"]
        out.append(spec)
    return out


def _aggregate_judge_votes(criteria, receipts: list[dict[str, Any]]) -> dict:
    rows = []
    for criterion in criteria:
        votes = []
        reasons = []
        for receipt in receipts:
            result = receipt["result"]
            by_name = {
                row["name"]: row
                for row in result.get("criteria", [])
                if isinstance(row, dict) and "name" in row
            }
            row = by_name.get(criterion.name)
            if row is None:
                continue
            passed = bool(row.get("pass"))
            votes.append(passed)
            if row.get("reason"):
                reasons.append(
                    {
                        "provider": receipt.get("provider"),
                        "model": receipt.get("model"),
                        "pass": passed,
                        "reason": row.get("reason"),
                    }
                )
        votes_for = sum(1 for vote in votes if vote)
        votes_total = len(votes)
        rows.append(
            {
                "name": criterion.name,
                "pass": votes_total > 0 and votes_for > votes_total / 2,
                "votes_for": votes_for,
                "votes_total": votes_total,
                "reasons": reasons,
            }
        )
    score = sum(1 for row in rows if row["pass"]) / len(rows)
    return {"criteria": rows, "score": score}


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


def _latest_cost(repo: Repository, call_id: str, cfg: ShadowConfig) -> float:
    """Estimate cost of this shadow call from model_intel + the original call's tokens."""
    from somm_core.pricing import cost_for_call

    with repo._open() as conn:
        row = conn.execute(
            "SELECT tokens_in, tokens_out FROM calls WHERE id = ?",
            (call_id,),
        ).fetchone()
    if not row:
        return 0.0
    tokens_in, tokens_out = row[0] or 0, row[1] or 0
    return cost_for_call(repo, cfg.gold_provider, cfg.gold_model, tokens_in, tokens_out)

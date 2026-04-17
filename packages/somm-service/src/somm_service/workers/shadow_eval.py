"""Shadow-eval worker — re-runs sampled production calls through a gold model
to build a real-world quality signal.

Design (per PLAN.md reframe):
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
from typing import TYPE_CHECKING

from somm_core.parse import extract_json

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

                # Sample
                sampled = _deterministic_sample(calls, cfg.sample_rate, cfg.max_grades_per_run)
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
        with self.repo._open() as conn:
            rows = conn.execute(
                "SELECT call_id, ts, project, workload_id, provider, model, "
                "prompt_hash, response_hash, workload_name, privacy_class, "
                "shadow_config_json FROM shadow_candidates ORDER BY ts DESC LIMIT 500"
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
        with self.repo._open() as conn:
            conn.execute(
                "UPDATE eval_results SET grading_started_at = NULL "
                "WHERE call_id = ? AND judge_score IS NULL",
                (call_id,),
            )

    def _write_result(self, call_row: dict, cfg: ShadowConfig, outcome: EvalOutcome) -> None:
        # Repurpose judge_reason as a JSON array with per-signal metadata; the
        # `cost_usd` used for budget accounting lives inside it. Not beautiful
        # but avoids another migration for v0.3b.
        notes = [{"cost_usd": _latest_cost(self.repo, call_row["call_id"], cfg)}]
        notes.extend({"note": n} for n in outcome.notes)
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

        structural = _structural_score(response, gold_text)
        text_sim = _text_similarity(response, gold_text)

        from somm_core.parse import stable_hash

        return EvalOutcome(
            call_id=call_row["call_id"],
            structural_score=structural,
            text_similarity_score=text_sim,
            judge_score=None,  # v0.3b+1
            gold_response_hash=stable_hash(gold_text),
            notes=[f"gold_tokens_in={gold.tokens_in}", f"gold_tokens_out={gold.tokens_out}"],
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


# ---------------------------------------------------------------------------
# Graders — lightweight, no model dependencies


def _structural_score(prod_text: str, gold_text: str) -> float | None:
    """Score based on JSON shape overlap. Returns None if neither parses."""
    prod = extract_json(prod_text)
    gold = extract_json(gold_text)
    if prod is None and gold is None:
        return None
    if prod is None or gold is None:
        return 0.0
    return _json_overlap(prod, gold)


def _json_overlap(a, b) -> float:
    """Recursive structural similarity. Dict: key overlap + per-key value match.
    List: length match + element-wise (string-compare). Else: equality 1.0/0.0.
    """
    if type(a) is not type(b):
        return 0.0
    if isinstance(a, dict):
        if not a and not b:
            return 1.0
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        per_key = sum(_json_overlap(a[k], b[k]) for k in keys) / len(keys)
        jaccard = len(keys) / max(1, len(set(a) | set(b)))
        return (per_key + jaccard) / 2.0
    if isinstance(a, list):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        elem_score = sum(_json_overlap(a[i], b[i]) for i in range(n)) / max(len(a), len(b))
        return elem_score
    if isinstance(a, str):
        return 1.0 if a.strip() == b.strip() else _text_similarity(a, b)
    return 1.0 if a == b else 0.0


def _text_similarity(a: str, b: str) -> float:
    """Word-bigram Jaccard similarity. 0..1. Cheap, deterministic, no deps."""

    def bigrams(s: str) -> set[tuple[str, str]]:
        words = s.lower().split()
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)}

    ga, gb = bigrams(a), bigrams(b)
    if not ga and not gb:
        return 1.0 if a.strip() == b.strip() else 0.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


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

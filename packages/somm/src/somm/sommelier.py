"""Sommelier — the model-advisor layer.

Reads `model_intel` + (optionally) shadow-eval scores + prior decisions
and produces a ranked, reasoned candidate list for a free-form question.

This is the cold-start counterpart to the agent worker: the agent emits
recommendations *after* a workload has real data; the sommelier gives
advice when you have none, grounded in pricing + capability coverage +
institutional memory from prior decisions.

Calls into this module are agent-facing (via MCP) but the logic is
deterministic and testable without any LLM in the loop.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from somm_core.models import Decision
from somm_core.parse import stable_hash

from somm.capabilities import model_has_capability

if TYPE_CHECKING:
    from somm_core.repository import Repository


@dataclass(slots=True)
class AdviseConstraints:
    """Query constraints for sommelier.advise(). All optional — missing means
    'don't filter on this dimension'."""

    capabilities: list[str] = field(default_factory=list)
    providers: list[str] | None = None   # whitelist; None = any
    max_price_in_per_1m: float | None = None
    max_price_out_per_1m: float | None = None
    min_context_window: int | None = None
    free_only: bool = False               # shortcut: both prices == 0
    workload: str | None = None           # for scoping shadow-eval lookup
    limit: int = 8


@dataclass(slots=True)
class Candidate:
    """A ranked model candidate. `reasons` is a short list of human-readable
    factors the sommelier weighed — intended to be shown verbatim to users.
    """

    provider: str
    model: str
    price_in_per_1m: float | None
    price_out_per_1m: float | None
    context_window: int | None
    capabilities_json: dict | None
    last_seen: str | None
    score: float
    reasons: list[str]
    shadow_score: float | None = None   # set when shadow-eval data exists

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "price_in_per_1m": self.price_in_per_1m,
            "price_out_per_1m": self.price_out_per_1m,
            "context_window": self.context_window,
            "last_seen": self.last_seen,
            "score": round(self.score, 3),
            "shadow_score": self.shadow_score,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class ConsultResult:
    """Full sommelier output — ranked candidates plus any prior decisions
    that match the question or workload. This is what non-MCP callers want:
    one object with both the fresh ranking and the institutional memory.
    """

    question: str
    project: str
    constraints: dict
    candidates: list[Candidate]
    prior_decisions: list[dict]
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "project": self.project,
            "constraints": self.constraints,
            "candidates": [c.as_dict() for c in self.candidates],
            "prior_decisions": self.prior_decisions,
            "note": self.note,
        }


# ---------------------------------------------------------------------------


def advise(
    repo: Repository,
    constraints: AdviseConstraints,
) -> list[Candidate]:
    """Rank candidates from `model_intel` against the given constraints.

    Ranking priorities (in order):
      1. Shadow-eval score for the named workload (if any).
      2. Price (cheaper wins — free tier prefered when free_only=True).
      3. Context window (larger wins, as a tiebreaker for similar-priced).
      4. Recency of `last_seen` (fresher wins — dead models age out).

    Capability filtering is *hard*: a candidate that's known to lack a
    required capability is excluded. Unknown capability status is kept
    (same "let the provider try" semantics as the router).
    """
    rows = _fetch_intel(repo, constraints)
    shadow_map = _shadow_map_for_workload(repo, constraints.workload)

    candidates: list[Candidate] = []
    for r in rows:
        provider, model = r["provider"], r["model"]

        # Capability hard filter
        drop_for: str | None = None
        for cap in constraints.capabilities:
            verdict = model_has_capability(repo, provider, model, cap)
            if verdict is False:
                drop_for = cap
                break
        if drop_for:
            continue

        reasons: list[str] = []
        price_in = r["price_in_per_1m"] or 0.0
        price_out = r["price_out_per_1m"] or 0.0
        ctx = r["context_window"]

        if price_in == 0 and price_out == 0:
            reasons.append("free")
        else:
            reasons.append(f"${price_in:.2f}/${price_out:.2f} per 1M in/out")
        if ctx:
            reasons.append(f"{ctx:,} ctx")
        for cap in constraints.capabilities:
            verdict = model_has_capability(repo, provider, model, cap)
            if verdict is True:
                reasons.append(f"{cap}✓")
            elif verdict is None:
                reasons.append(f"{cap}?")

        shadow_score = shadow_map.get((provider, model))
        if shadow_score is not None:
            reasons.insert(0, f"shadow score {shadow_score:.2f}")

        score = _score(
            price_in, price_out, ctx, r.get("last_seen"), shadow_score
        )
        candidates.append(
            Candidate(
                provider=provider,
                model=model,
                price_in_per_1m=r["price_in_per_1m"],
                price_out_per_1m=r["price_out_per_1m"],
                context_window=ctx,
                capabilities_json=r.get("capabilities"),
                last_seen=r.get("last_seen"),
                score=score,
                reasons=reasons,
                shadow_score=shadow_score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[: constraints.limit]


# ---------------------------------------------------------------------------


def consult(
    repo: Repository,
    question: str,
    constraints: AdviseConstraints,
    *,
    project: str = "default",
    global_repo: Repository | None = None,
    priors_limit: int = 5,
) -> ConsultResult:
    """Full sommelier consultation: ranked candidates + prior decisions.

    Mirrors what `somm_advise` (MCP) returns but available directly to
    library callers. `global_repo`, when given, is searched for decisions
    before the project repo — keeps the "decisions mirror globally" model
    working for Python callers without requiring MCP.

    The returned `prior_decisions` have the same shape the MCP tool
    emits so code can switch between surfaces without reshaping data.
    """
    cands = advise(repo, constraints)
    priors = _search_prior_decisions(
        repo, question=question, workload=constraints.workload,
        limit=priors_limit, global_repo=global_repo,
    )
    return ConsultResult(
        question=question,
        project=project,
        constraints={
            "capabilities": constraints.capabilities,
            "providers": constraints.providers,
            "max_price_in_per_1m": constraints.max_price_in_per_1m,
            "max_price_out_per_1m": constraints.max_price_out_per_1m,
            "min_context_window": constraints.min_context_window,
            "free_only": constraints.free_only,
            "workload": constraints.workload,
        },
        candidates=cands,
        prior_decisions=priors,
        note=(
            "No candidates matched. Loosen constraints, "
            "run `somm-serve admin refresh-intel` to refresh the "
            "model intel cache, or add a provider with capable models."
        )
        if not cands
        else None,
    )


def _search_prior_decisions(
    repo: Repository,
    *,
    question: str | None,
    workload: str | None,
    limit: int,
    global_repo: Repository | None = None,
) -> list[dict]:
    """Search decisions across global → project, returning the first
    non-empty result set. Matches the MCP helper's fallback order.

    Falls back to keyword-level recall when the exact / substring query
    misses: `Repository.search_decisions` uses LIKE '%question%', which
    is empty on anything but near-identical phrasing. If the question
    is long enough to contain distinctive content words, we retry per
    keyword and dedupe — so "Which free vision model should malo use
    for captioning?" still recalls an earlier decision titled "Which
    local vision model should malo use for describing data-viz
    artifacts?".
    """
    for candidate in (global_repo, repo):
        if candidate is None:
            continue
        try:
            rows = candidate.search_decisions(
                question=question, workload=workload, limit=limit,
            )
        except Exception:  # noqa: BLE001
            rows = []
        if rows:
            return [_decision_as_dict(d) for d in rows]

    # Keyword fallback — union across every content word, dedupe by id,
    # newest-first. A single-keyword match is better than nothing; a
    # multi-keyword match surfaces decisions that overlap on more than
    # one theme.
    seen: set[str] = set()
    aggregated: list = []  # (ts, decision_dict) for final sort
    for keyword in _recall_keywords(question):
        for candidate in (global_repo, repo):
            if candidate is None:
                continue
            try:
                rows = candidate.search_decisions(
                    question=keyword, workload=workload, limit=limit,
                )
            except Exception:  # noqa: BLE001
                rows = []
            for d in rows:
                if d.id in seen:
                    continue
                seen.add(d.id)
                aggregated.append((d.ts, _decision_as_dict(d)))
    # Newest-first by ts
    aggregated.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in aggregated[:limit]]


_RECALL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "good", "has", "have", "how", "i", "in", "is",
    "it", "my", "of", "on", "or", "our", "should", "some", "that", "the",
    "these", "this", "to", "use", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you",
}


def _recall_keywords(question: str | None, *, min_len: int = 4, max_keywords: int = 4) -> list[str]:
    """Extract keyword candidates from a natural-language question.

    Longest content words first, under the assumption that rarer/longer
    words are better matches for prior decisions. Caps at `max_keywords`
    so we don't spam the repo with dozens of LIKE queries.
    """
    if not question:
        return []
    import re
    tokens = re.findall(r"[a-z][a-z0-9\-]+", question.lower())
    content = [t for t in tokens if len(t) >= min_len and t not in _RECALL_STOPWORDS]
    # Dedupe preserving order, then sort by length descending
    seen: set[str] = set()
    deduped: list[str] = []
    for t in content:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return sorted(deduped, key=lambda s: -len(s))[:max_keywords]


def _decision_as_dict(d: Decision) -> dict:
    return {
        "id": d.id,
        "ts": d.ts.isoformat() if hasattr(d.ts, "isoformat") else d.ts,
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


# ---------------------------------------------------------------------------


def build_decision(
    question: str,
    candidates: list[Candidate | dict],
    rationale: str,
    project: str,
    chosen_provider: str | None = None,
    chosen_model: str | None = None,
    workload: str | None = None,
    workload_id: str | None = None,
    constraints: AdviseConstraints | dict | None = None,
    agent: str | None = None,
) -> Decision:
    """Assemble a Decision ready for `Repository.record_decision`.

    `question` is stored verbatim plus hashed (normalised case/whitespace)
    so future queries can dedup even when the agent phrases things slightly
    differently.
    """
    cand_dicts = [c.as_dict() if isinstance(c, Candidate) else c for c in candidates]
    cons_dict: dict | None
    if isinstance(constraints, AdviseConstraints):
        cons_dict = {
            "capabilities": constraints.capabilities,
            "providers": constraints.providers,
            "max_price_in_per_1m": constraints.max_price_in_per_1m,
            "max_price_out_per_1m": constraints.max_price_out_per_1m,
            "min_context_window": constraints.min_context_window,
            "free_only": constraints.free_only,
        }
    else:
        cons_dict = constraints or None

    q_norm = _normalise_question(question)
    return Decision(
        id=str(uuid.uuid4()),
        ts=datetime.now(UTC),
        project=project,
        question=question,
        question_hash=stable_hash(q_norm),
        candidates=cand_dicts,
        rationale=rationale,
        chosen_provider=chosen_provider,
        chosen_model=chosen_model,
        workload_id=workload_id,
        workload_name=workload,
        constraints=cons_dict,
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Internals


def _fetch_intel(repo: Repository, c: AdviseConstraints) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if c.providers:
        placeholders = ",".join("?" for _ in c.providers)
        clauses.append(f"provider IN ({placeholders})")
        params.extend(c.providers)
    if c.free_only:
        clauses.append(
            "(COALESCE(price_in_per_1m, 0) = 0 AND COALESCE(price_out_per_1m, 0) = 0)"
        )
    else:
        if c.max_price_in_per_1m is not None:
            clauses.append("(price_in_per_1m IS NULL OR price_in_per_1m <= ?)")
            params.append(c.max_price_in_per_1m)
        if c.max_price_out_per_1m is not None:
            clauses.append("(price_out_per_1m IS NULL OR price_out_per_1m <= ?)")
            params.append(c.max_price_out_per_1m)
    if c.min_context_window is not None:
        clauses.append("(context_window IS NULL OR context_window >= ?)")
        params.append(c.min_context_window)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT provider, model, price_in_per_1m, price_out_per_1m, "
        "       context_window, capabilities_json, last_seen, source "
        f"FROM model_intel {where} ORDER BY provider, model"
    )
    with repo._open() as conn:
        rows = conn.execute(sql, params).fetchall()

    out: list[dict] = []
    for r in rows:
        caps = None
        if r[5]:
            try:
                caps = json.loads(r[5])
            except json.JSONDecodeError:
                caps = None
        out.append(
            {
                "provider": r[0],
                "model": r[1],
                "price_in_per_1m": r[2],
                "price_out_per_1m": r[3],
                "context_window": r[4],
                "capabilities": caps,
                "last_seen": r[6],
                "source": r[7],
            }
        )
    return out


def _shadow_map_for_workload(
    repo: Repository, workload: str | None
) -> dict[tuple[str, str], float]:
    if not workload:
        return {}
    with repo._open() as conn:
        wl_row = conn.execute(
            "SELECT id FROM workloads WHERE name = ? LIMIT 1",
            (workload,),
        ).fetchone()
        if not wl_row:
            return {}
        rows = conn.execute(
            """
            SELECT c.provider, c.model,
                   AVG(COALESCE(er.structural_score, er.embedding_score)) AS score
            FROM eval_results er
            JOIN calls c ON c.id = er.call_id
            WHERE c.workload_id = ?
              AND (er.structural_score IS NOT NULL OR er.embedding_score IS NOT NULL)
            GROUP BY c.provider, c.model
            """,
            (wl_row[0],),
        ).fetchall()
    return {(r[0], r[1]): r[2] for r in rows if r[2] is not None}


def _score(
    price_in: float,
    price_out: float,
    ctx: int | None,
    last_seen: str | None,
    shadow_score: float | None,
) -> float:
    """Composite score. Higher = better.

    Weights are tuned so shadow-eval evidence dominates pricing deltas when
    it exists — you shouldn't downgrade a model proven to work for your
    workload just because it's slightly pricier than an untested one.
    """
    score = 0.0
    if shadow_score is not None:
        score += 100.0 * shadow_score

    # Cheaper wins. Use log-like penalty so $0 vs $0.15 is a small nudge but
    # $0.15 vs $15 is significant.
    total = price_in + price_out
    if total == 0:
        score += 50.0
    else:
        # 50 at $0.01 total, ~35 at $0.30, ~20 at $1, ~10 at $5, 5 at $20
        score += max(0.0, 50.0 - 15.0 * (total**0.5))

    if ctx:
        score += min(20.0, ctx / 10_000)

    # Recency bonus — favour fresh entries; model_intel last_seen is ISO.
    if last_seen:
        try:
            seen = datetime.fromisoformat(last_seen.replace(" ", "T"))
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=UTC)
            days_old = (datetime.now(UTC) - seen).days
            if days_old < 7:
                score += 5.0
            elif days_old > 90:
                score -= 5.0
        except ValueError:
            pass

    return score


def _normalise_question(q: str) -> str:
    return " ".join(q.strip().lower().split())

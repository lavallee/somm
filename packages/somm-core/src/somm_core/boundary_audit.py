"""The data-boundary ledger audit: what tier actually carried each workload's
data, read back from ``calls`` (``somm boundary audit``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from somm_core.boundary import (
    REFUSAL_ERROR_KIND,
    BoundaryPolicy,
    DataTier,
    load_policy,
    parse_tier,
    resolve_boundary,
)


def audit(
    conn,
    project: str,
    *,
    since_days: float = 30.0,
    policy: BoundaryPolicy | None = None,
    base_urls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per workload: declared and effective boundary, and the worst tier that
    actually carried its data in the window, from the ledger.

    Rows written before ``calls.data_tier`` existed are classified from
    provider/model at read time (``base_urls`` maps a provider name to the
    endpoint to assume, e.g. the configured Ollama URL). Every row that
    reached a provider counts, whatever its outcome: a failed call still sent
    its prompt. Boundary refusals (no provider) are counted separately.
    ``violation`` is True when the worst tier is looser than the effective
    boundary.
    """
    policy = policy or load_policy()
    base_urls = base_urls or {}
    since = f"-{float(since_days)} days"
    workloads = conn.execute(
        "SELECT id, name, privacy_class, data_boundary FROM workloads WHERE project = ?",
        (project,),
    ).fetchall()
    by_id: dict[str, dict[str, Any]] = {}
    for wid, name, privacy, stored in workloads:
        stub = _AuditWorkload(name=name, privacy_class=privacy, data_boundary=stored)
        effective = resolve_boundary(stub, project=project, policy=policy)
        by_id[wid] = {
            "workload": name,
            "workload_id": wid,
            "privacy_class": privacy,
            "declared_boundary": stored,
            "effective_boundary": effective.tier.value,
            "boundary_source": effective.source,
            "calls": 0,
            "refusals": 0,
            "worst_tier": None,
            "violation": False,
            "assumed": False,
            "routes": [],
        }
    rows = conn.execute(
        "SELECT workload_id, provider, model, data_tier, error_kind, COUNT(*) "
        "FROM calls WHERE project = ? AND ts >= datetime('now', ?) "
        "GROUP BY workload_id, provider, model, data_tier, error_kind",
        (project, since),
    ).fetchall()
    for wid, provider, model, data_tier, error_kind, n in rows:
        entry = by_id.get(wid)
        if entry is None:
            continue
        if not provider:
            if error_kind == REFUSAL_ERROR_KIND:
                entry["refusals"] += n
            continue
        route = policy.classify(provider, model, base_url=base_urls.get(provider))
        if data_tier:
            try:
                tier = parse_tier(data_tier)
                basis, recorded = "recorded at call time", True
            except ValueError:
                tier, basis, recorded = DataTier.UNVETTED, f"unknown recorded tier {data_tier!r}", True
        else:
            tier, basis, recorded = route.tier, f"classified now: {route.basis}", False
        entry["calls"] += n
        entry["routes"].append(
            {
                "provider": provider,
                "model": model,
                "tier": tier.value,
                "calls": n,
                "recorded": recorded,
                "origin": route.origin,
                "basis": basis,
            }
        )
        worst = entry["worst_tier"]
        if worst is None or tier.rank > parse_tier(worst).rank:
            entry["worst_tier"] = tier.value
    out = []
    for entry in by_id.values():
        if entry["worst_tier"] is not None:
            entry["violation"] = (
                parse_tier(entry["worst_tier"]).rank > parse_tier(entry["effective_boundary"]).rank
            )
        entry["routes"].sort(key=lambda r: (-parse_tier(r["tier"]).rank, -r["calls"]))
        # "assumed": the worst route's tier rests on an account setting somm
        # cannot see (a seat), so the clean result is only as good as that.
        entry["assumed"] = any(
            r["origin"] == "assumed" and r["tier"] == entry["worst_tier"] for r in entry["routes"]
        )
        out.append(entry)
    out.sort(key=lambda e: (not e["violation"], -e["calls"], e["workload"]))
    return out


@dataclass(frozen=True, slots=True)
class _AuditWorkload:
    name: str
    privacy_class: str
    data_boundary: str | None


__all__ = ["audit"]

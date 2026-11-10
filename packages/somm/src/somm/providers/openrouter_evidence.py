"""OpenRouter's own published evidence for which upstream endpoints keep no data.

Two public lists, no key needed:

- ``/api/v1/endpoints/zdr``: every endpoint OpenRouter routes under zero data
  retention, one row per (model id, upstream endpoint ``tag``). A ``:free``
  model id is its own id with its own endpoints, so free and paid are judged
  separately with no special rule. ``safety_retention: true`` means the
  upstream may still keep prompts for abuse review: not strict zero retention.
- ``/api/v1/providers``: each upstream's ``slug`` and ``headquarters``.

The lists are fetched only when a call's boundary is strict enough to need
them, kept beside ``boundary.toml`` as ``openrouter-evidence.json``, and
refreshed at most daily. A failed fetch keeps the last good copy; with no
copy there is no evidence and nothing is promoted. Nothing here raises.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import httpx

ZDR_URL = "https://openrouter.ai/api/v1/endpoints/zdr"
PROVIDERS_URL = "https://openrouter.ai/api/v1/providers"
_REFRESH_S = 86400.0  # at most daily
_RETRY_S = 3600.0  # after a failed fetch, keep what we have for an hour

_lock = threading.Lock()
_memo: dict = {"evidence": None, "next_check": 0.0}
_log = logging.getLogger("somm.providers.openrouter")


def qualifying_upstreams(
    model_id: str, zdr: list[dict], providers: list[dict], headquarters: tuple[str, ...] | list[str]
) -> list[str]:
    """Endpoint tags that may carry ``model_id`` under strict zero retention.

    An endpoint qualifies when it is on the ZDR list for exactly this model
    id, and its upstream has no safety-retention endpoint for the model
    (OpenRouter matches a base slug against all of a provider's endpoints, so
    one retaining variant disqualifies the upstream), and the upstream's
    headquarters is known and in ``headquarters``.
    """
    allowed = {str(h).upper() for h in headquarters}
    hq = {p.get("slug"): p.get("headquarters") for p in providers if isinstance(p, dict)}
    mine = [e for e in zdr if isinstance(e, dict) and e.get("model_id") == model_id and e.get("tag")]

    def slug(e: dict) -> str:
        return str(e["tag"]).split("/")[0]

    retaining = {slug(e) for e in mine if e.get("safety_retention") is not False}
    return sorted(
        {
            str(e["tag"])
            for e in mine
            if slug(e) not in retaining
            and isinstance(hq.get(slug(e)), str)
            and hq[slug(e)].upper() in allowed
        }
    )


def cache_path() -> Path:
    from somm_core.boundary import boundary_path

    return boundary_path().parent / "openrouter-evidence.json"


def load_evidence() -> dict | None:
    """``{"fetched_at": epoch, "zdr": [...], "providers": [...]}``, or None
    when there has never been a good fetch. Never raises."""
    now = time.time()
    with _lock:
        if now < _memo["next_check"]:
            return _memo["evidence"]
        evidence = _memo["evidence"] or _read(cache_path())
        if evidence and now - evidence["fetched_at"] < _REFRESH_S:
            _memo.update(evidence=evidence, next_check=evidence["fetched_at"] + _REFRESH_S)
            return evidence
        try:
            evidence = _fetch()
            evidence["fetched_at"] = now
            _write(cache_path(), evidence)
            next_check = now + _REFRESH_S
        except Exception as exc:  # noqa: BLE001 — fail closed to the last good copy
            _log.warning("openrouter evidence fetch failed (%s); using %s", exc,
                         "the last good copy" if evidence else "no evidence")
            next_check = now + _RETRY_S
        _memo.update(evidence=evidence, next_check=next_check)
        return evidence


def _fetch() -> dict:
    with httpx.Client(timeout=10.0) as client:
        zdr = client.get(ZDR_URL)
        zdr.raise_for_status()
        providers = client.get(PROVIDERS_URL)
        providers.raise_for_status()
    rows = zdr.json()["data"]
    upstreams = providers.json()["data"]
    if not isinstance(rows, list) or not isinstance(upstreams, list):
        raise ValueError("unexpected payload shape")
    return {
        "zdr": [
            {k: e.get(k) for k in ("model_id", "tag", "provider_name", "safety_retention")}
            for e in rows
            if isinstance(e, dict)
        ],
        "providers": [
            {k: p.get(k) for k in ("slug", "name", "headquarters")}
            for p in upstreams
            if isinstance(p, dict)
        ],
    }


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("zdr"), list) and isinstance(data.get("providers"), list):
            data["fetched_at"] = float(data["fetched_at"])
            return data
    except Exception:  # noqa: BLE001 — a missing or damaged copy is no evidence
        pass
    return None


def _write(path: Path, evidence: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(evidence), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # the in-memory copy still serves this process
        _log.warning("could not save openrouter evidence to %s: %s", path, exc)

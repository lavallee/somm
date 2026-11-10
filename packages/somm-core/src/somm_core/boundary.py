"""Data boundary: how far a workload's data may travel, and where each route goes.

Four ordered tiers, strictest first (``DataTier``)::

    local < zdr < subscribed < unvetted

A *route* is a (provider, model, endpoint) somm could send a request to; the
classifier gives it a tier. A *workload boundary* is the loosest tier its data
may reach. A route is admissible when ``rank(route) <= rank(boundary)``, which
is all :func:`admit` checks. Every egress point in somm calls it.

Classification fails closed: a route somm cannot place is ``unvetted``. The
tiers are claims about the owner's contracts, so the owner's
``~/.somm/boundary.toml`` (``SOMM_BOUNDARY_PATH``) wins over the shipped
defaults in ``data/boundary_defaults.toml``. A tier is a declaration, not a
technical proof: somm enforces where it sends a request, not what the far end
does with it.

The owner file::

    [defaults]
    boundary = "subscribed"          # project default when a workload sets none

    [projects.fab]
    boundary = "local"               # per-project default (beats [defaults])

    [[route]]
    match = "openai"                 # provider, or provider/model-glob
    tier = "subscribed"
    note = "API account, org settings checked 2026-11"

    [[route]]
    host = "gpu1.example.ts.net"     # an endpoint host (glob), any provider
    tier = "local"
    note = "my rented box"
"""

from __future__ import annotations

import fnmatch
import ipaddress
import os
import sys
import threading
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class DataTier(StrEnum):
    """Ordered data tiers. Comparison follows strictness, not spelling:
    ``DataTier.LOCAL < DataTier.UNVETTED``."""

    LOCAL = "local"
    ZDR = "zdr"
    SUBSCRIBED = "subscribed"
    UNVETTED = "unvetted"

    @property
    def rank(self) -> int:
        return _RANK[self]

    # A StrEnum compares as text by default, where "zdr" sorts after
    # "subscribed"; these keep ``<`` meaning "stricter than".
    def __lt__(self, other: object) -> bool:
        return self.rank < _as_tier(other).rank

    def __le__(self, other: object) -> bool:
        return self.rank <= _as_tier(other).rank

    def __gt__(self, other: object) -> bool:
        return self.rank > _as_tier(other).rank

    def __ge__(self, other: object) -> bool:
        return self.rank >= _as_tier(other).rank

    __hash__ = StrEnum.__hash__


_RANK = {
    DataTier.LOCAL: 0,
    DataTier.ZDR: 1,
    DataTier.SUBSCRIBED: 2,
    DataTier.UNVETTED: 3,
}

TIER_NAMES: tuple[str, ...] = tuple(t.value for t in DataTier)


def _as_tier(value: object) -> DataTier:
    if isinstance(value, DataTier):
        return value
    if isinstance(value, str):
        return parse_tier(value)
    raise TypeError(f"cannot compare DataTier with {type(value).__name__}")


def parse_tier(value: DataTier | str) -> DataTier:
    """Parse a tier name. Raises ValueError naming the valid tiers."""
    if isinstance(value, DataTier):
        return value
    try:
        return DataTier(str(value).strip().lower())
    except ValueError:
        raise ValueError(
            f"unknown data tier {value!r}; use one of {', '.join(TIER_NAMES)}"
        ) from None


def admit(route: RouteTier | DataTier | str, boundary: DataTier | str) -> bool:
    """The one admission check: may ``route`` carry data bounded by ``boundary``?

    ``route`` is a classified :class:`RouteTier` or a bare tier.
    """
    tier = route.tier if isinstance(route, RouteTier) else parse_tier(route)
    return tier.rank <= parse_tier(boundary).rank


# ---------------------------------------------------------------------------
# Classification


@dataclass(frozen=True, slots=True)
class RouteTier:
    """A classified route: its tier and why."""

    provider: str
    model: str
    tier: DataTier
    basis: str
    # "default" (shipped/somm rule), "assumed" (a shipped tier that depends on
    # an account setting somm cannot see, e.g. a seat's training toggle),
    # "owner" (boundary.toml), or "evidence" (promoted by live evidence).
    origin: str
    rule: str = ""  # the pattern or rule that matched
    source: str = ""
    last_verified: str = ""
    note: str = ""
    confidence: str = ""

    @property
    def route(self) -> str:
        return f"{self.provider}/{self.model}" if self.model else self.provider

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier.value,
            "basis": self.basis,
            "origin": self.origin,
            "rule": self.rule,
            "source": self.source,
            "last_verified": self.last_verified,
            "note": self.note,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class _Rule:
    tier: DataTier
    origin: str  # "owner", "default", or "assumed" (a shipped `assumed = true` rule)
    match: str = ""  # "provider" or "provider/model-glob"
    host: str = ""  # endpoint host glob (owner only)
    hosts: tuple[str, ...] = ()  # hosts a shipped provider tier applies to
    basis: str = ""
    source: str = ""
    last_verified: str = ""
    note: str = ""
    confidence: str = ""
    quote: str = ""
    jurisdiction: str = ""

    @property
    def provider_pattern(self) -> str:
        return self.match.split("/", 1)[0]

    @property
    def model_pattern(self) -> str | None:
        return self.match.split("/", 1)[1] if "/" in self.match else None

    def matches(self, provider: str, model: str) -> bool:
        if not self.match:
            return False
        if not fnmatch.fnmatchcase(provider, self.provider_pattern):
            return False
        pattern = self.model_pattern
        return True if pattern is None else fnmatch.fnmatchcase(model or "", pattern)

    @property
    def specificity(self) -> tuple[int, int]:
        """Exact model > model glob > provider only; then longer patterns."""
        pattern = self.model_pattern
        if pattern is None:
            level = 0
        elif any(ch in pattern for ch in "*?["):
            level = 1
        else:
            level = 2
        return (level, len(self.match))


@dataclass(slots=True)
class BoundaryPolicy:
    """Shipped defaults plus the owner's file, parsed once per file change."""

    default_rules: list[_Rule] = field(default_factory=list)
    owner_rules: list[_Rule] = field(default_factory=list)
    owner_default: DataTier | None = None
    owner_projects: dict[str, DataTier] = field(default_factory=dict)
    owner_path: Path | None = None
    owner_error: str | None = None  # the owner file exists but could not be read
    # Where an OpenRouter upstream must be headquartered for live ZDR evidence
    # to promote it ([defaults] zdr_headquarters). Empty promotes nothing.
    zdr_headquarters: tuple[str, ...] = ("US",)

    # -- classification ------------------------------------------------

    def classify(
        self,
        provider: str,
        model: str | None = None,
        *,
        base_url: str | None = None,
        kind: str = "provider",
        declared_tier: DataTier | str | None = None,
    ) -> RouteTier:
        """Classify one route. Never raises; unknown is ``unvetted``.

        ``kind="handler"`` classifies a workload handler: its rule name is
        ``handler:<kind>``. When no rule matches, a handler that declares
        ``local`` is ``local``; any other declaration is ``unvetted``.
        """
        provider = str(provider or "")
        model = str(model or "")
        name = f"handler:{provider}" if kind == "handler" else provider
        host = _host(base_url)

        def result(rule: _Rule | None, tier: DataTier, basis: str, origin: str, label: str):
            return RouteTier(
                provider=name,
                model=model,
                tier=tier,
                basis=basis,
                origin=origin,
                rule=label,
                source=rule.source if rule else "",
                last_verified=rule.last_verified if rule else "",
                note=rule.note if rule else "",
                confidence=rule.confidence if rule else "",
            )

        # 1. Owner: endpoint host, then provider/model patterns.
        if host:
            for rule in self.owner_rules:
                if rule.host and fnmatch.fnmatchcase(host, rule.host.lower()):
                    return result(
                        rule, rule.tier, rule.note or f"owner declared host {rule.host}",
                        "owner", f"host={rule.host}",
                    )
        owner = _best(self.owner_rules, name, model)
        if owner is not None:
            return result(
                owner, owner.tier, owner.note or f"owner declared {owner.match}",
                "owner", owner.match,
            )
        # (An unreadable owner file contributes no rules, so its promotions
        # never apply; project_default() then fails closed to local.)
        # 2. Shipped model-level rules (a cloud model behind a local server).
        model_rules = [r for r in self.default_rules if r.model_pattern is not None]
        shipped_model = _best(model_rules, name, model)
        if shipped_model is not None:
            return result(shipped_model, shipped_model.tier, shipped_model.basis,
                          shipped_model.origin, shipped_model.match)
        # 3. Endpoint: loopback or private address never leaves your hardware.
        if host and _is_local_host(host):
            return RouteTier(
                provider=name, model=model, tier=DataTier.LOCAL,
                basis=f"endpoint {host} is a loopback or private address",
                origin="default", rule="endpoint",
            )
        # 4. Shipped provider-level rules, bound to the provider's own hosts.
        provider_rules = [r for r in self.default_rules if r.model_pattern is None]
        shipped = _best(provider_rules, name, model)
        if shipped is not None:
            if host and shipped.hosts and not any(
                fnmatch.fnmatchcase(host, h.lower()) for h in shipped.hosts
            ):
                return result(
                    None, DataTier.UNVETTED,
                    f"endpoint {host} is not {provider}'s own API host "
                    f"({', '.join(shipped.hosts)})",
                    "default", "endpoint",
                )
            return result(shipped, shipped.tier, shipped.basis, shipped.origin, shipped.match)
        # 5. A handler may claim only `local` (its data stays in this process
        # or on your hardware); a looser claim is no evidence of anything.
        if kind == "handler" and str(declared_tier or "").strip().lower() == "local":
            return RouteTier(
                provider=name, model=model, tier=DataTier.LOCAL,
                basis=f"declared local by the {provider!r} handler", origin="default",
                rule="handler",
            )
        # 6. Fail closed.
        what = (
            "handler does not declare local" if kind == "handler"
            else "no rule classifies this route"
        )
        return RouteTier(
            provider=name, model=model, tier=DataTier.UNVETTED,
            basis=f"{what}; unknown routes are unvetted", origin="default", rule="fallback",
        )

    # -- project default -----------------------------------------------

    def project_default(self, project: str | None, env: dict | None = None):
        """(tier, source) for the project default, or (None, "") when unset.

        ``SOMM_DATA_BOUNDARY`` wins, then ``[projects.<name>]``, then
        ``[defaults]``. An invalid value, or an owner file that cannot be
        read, fails closed to ``local`` with the reason in ``source``.
        """
        env = os.environ if env is None else env
        raw = (env.get("SOMM_DATA_BOUNDARY") or "").strip()
        if raw:
            try:
                return parse_tier(raw), "SOMM_DATA_BOUNDARY"
            except ValueError:
                return DataTier.LOCAL, f"invalid SOMM_DATA_BOUNDARY={raw!r}, treated as local"
        if self.owner_error:
            return DataTier.LOCAL, f"unreadable {self.owner_path} ({self.owner_error}), treated as local"
        if project and project in self.owner_projects:
            return self.owner_projects[project], f"{self.owner_path} [projects.{project}]"
        if self.owner_default is not None:
            return self.owner_default, f"{self.owner_path} [defaults]"
        return None, ""

    def routes(self) -> list[dict[str, Any]]:
        """Every declared rule, owner first, for ``somm boundary routes``."""
        out = []
        for rule in [*self.owner_rules, *self.default_rules]:
            out.append(
                {
                    "match": rule.match or f"host={rule.host}",
                    "tier": rule.tier.value,
                    "origin": rule.origin,
                    "basis": rule.note if rule.origin == "owner" else rule.basis,
                    "hosts": list(rule.hosts),
                    "source": rule.source,
                    "quote": rule.quote,
                    "jurisdiction": rule.jurisdiction,
                    "last_verified": rule.last_verified,
                    "confidence": rule.confidence,
                }
            )
        return out


def _best(rules: list[_Rule], provider: str, model: str) -> _Rule | None:
    best: _Rule | None = None
    for rule in rules:
        if rule.matches(provider, model) and (best is None or rule.specificity > best.specificity):
            best = rule
    return best


def _host(base_url: str | None) -> str:
    if not base_url:
        return ""
    raw = str(base_url).strip()
    if "://" not in raw:
        raw = "http://" + raw
    try:
        return (urlsplit(raw).hostname or "").lower()
    except ValueError:
        return ""


def _is_local_host(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a hostname somm would have to resolve: not assumed local
    if ip.is_loopback:
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in _PRIVATE_V4)
    return ip in _ULA_V6 or ip.is_link_local


_PRIVATE_V4 = tuple(
    ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_ULA_V6 = ipaddress.ip_network("fc00::/7")


# ---------------------------------------------------------------------------
# Loading


def boundary_path() -> Path:
    env = os.environ.get("SOMM_BOUNDARY_PATH")
    return Path(env).expanduser() if env else Path.home() / ".somm" / "boundary.toml"


def _parse_rules(data: dict, origin: str) -> list[_Rule]:
    rules: list[_Rule] = []
    for entry in data.get("route", []) or []:
        if not isinstance(entry, dict):
            continue
        match = str(entry.get("match", "") or "").strip()
        host = str(entry.get("host", "") or "").strip().lower()
        if not match and not host:
            continue
        try:
            tier = parse_tier(entry.get("tier", ""))
        except ValueError:
            # A rule with a bad tier fails closed rather than vanishing.
            tier = DataTier.UNVETTED
        hosts = entry.get("hosts") or ()
        # Only shipped rules can be "assumed"; an owner rule is a declaration.
        assumed = origin == "default" and entry.get("assumed") is True
        rules.append(
            _Rule(
                tier=tier,
                origin="assumed" if assumed else origin,
                match=match,
                host=host,
                hosts=tuple(str(h).lower() for h in hosts) if isinstance(hosts, list) else (),
                basis=str(entry.get("basis", "") or ""),
                source=str(entry.get("source", "") or ""),
                last_verified=str(entry.get("last_verified", "") or ""),
                note=str(entry.get("note", "") or ""),
                confidence=str(entry.get("confidence", "") or ""),
                quote=str(entry.get("quote", "") or ""),
                jurisdiction=str(entry.get("jurisdiction", "") or ""),
            )
        )
    return rules


def _load_defaults() -> list[_Rule]:
    from importlib import resources

    raw = (resources.files("somm_core") / "data" / "boundary_defaults.toml").read_bytes()
    return _parse_rules(tomllib.loads(raw.decode("utf-8")), "default")


_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"key": None, "policy": None, "defaults": None}


def load_policy(path: Path | None = None) -> BoundaryPolicy:
    """Shipped defaults + the owner file. Cached until the file changes."""
    path = path or boundary_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(path), None, None)
    with _cache_lock:
        if _cache["key"] == key and _cache["policy"] is not None:
            return _cache["policy"]
        if _cache["defaults"] is None:
            _cache["defaults"] = _load_defaults()
        defaults = _cache["defaults"]
    policy = BoundaryPolicy(default_rules=list(defaults), owner_path=path)
    if key[1] is not None:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — reported, and fails closed
            policy.owner_error = f"{type(exc).__name__}: {exc}"[:200]
            # Say so where the operator will see it: until the file parses,
            # every bounded workload is held to `local`.
            print(
                f"somm: {path} could not be read ({policy.owner_error}); "
                "its promotions are ignored and the default boundary is local "
                "until it is fixed",
                file=sys.stderr,
            )
            policy.zdr_headquarters = ()
        else:
            policy.owner_rules = _parse_rules(data, "owner")
            defaults_tbl = data.get("defaults") or {}
            if isinstance(defaults_tbl, dict) and defaults_tbl.get("boundary"):
                try:
                    policy.owner_default = parse_tier(defaults_tbl["boundary"])
                except ValueError:
                    policy.owner_default = DataTier.LOCAL
            if isinstance(defaults_tbl, dict) and "zdr_headquarters" in defaults_tbl:
                hq = defaults_tbl["zdr_headquarters"]
                ok = isinstance(hq, list) and all(isinstance(h, str) for h in hq)
                policy.zdr_headquarters = tuple(h.strip().upper() for h in hq) if ok else ()
            projects = data.get("projects") or {}
            if isinstance(projects, dict):
                for name, tbl in projects.items():
                    if isinstance(tbl, dict) and tbl.get("boundary"):
                        try:
                            policy.owner_projects[str(name)] = parse_tier(tbl["boundary"])
                        except ValueError:
                            policy.owner_projects[str(name)] = DataTier.LOCAL
    with _cache_lock:
        _cache["key"] = key
        _cache["policy"] = policy
    return policy


def classify(
    provider: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
    kind: str = "provider",
    declared_tier: DataTier | str | None = None,
    policy: BoundaryPolicy | None = None,
) -> RouteTier:
    """Classify a route against the current policy. See :meth:`BoundaryPolicy.classify`."""
    return (policy or load_policy()).classify(
        provider, model, base_url=base_url, kind=kind, declared_tier=declared_tier
    )


# Harness names → the provider entry whose terms the attempt runs under.
HARNESS_PROVIDERS: dict[str, str] = {
    "claude-cli": "claude-cli",
    "codex": "codex-cli",
    "codex-cli": "codex-cli",
    "opencode": "opencode",
    "pi": "pi",
}


# ---------------------------------------------------------------------------
# Resolution


@dataclass(frozen=True, slots=True)
class ResolvedBoundary:
    tier: DataTier
    source: str  # call | workload | privacy_class | project default source | fleet default

    @property
    def is_default(self) -> bool:
        """True when nothing set a boundary (fleet default, admits everything)."""
        return self.source == FLEET_DEFAULT_SOURCE


FLEET_DEFAULT_SOURCE = "fleet default"


def resolve_boundary(
    workload: Any = None,
    *,
    per_call: DataTier | str | None = None,
    project: str | None = None,
    policy: BoundaryPolicy | None = None,
    env: dict | None = None,
) -> ResolvedBoundary:
    """The boundary a call runs under. First hit wins:

    1. the workload's stored ``data_boundary``;
    2. ``privacy_class=private`` → ``local``;
    3. the project default (``SOMM_DATA_BOUNDARY``, then boundary.toml);
    4. ``unvetted`` — the fleet default, which admits every route.

    ``privacy_class=private`` also caps any stored value at ``local``: a
    private workload never leaves your hardware. A ``per_call`` boundary may
    only tighten the result; a looser one is ignored.
    """
    tier: DataTier | None = None
    source = ""
    stored = getattr(workload, "data_boundary", None) if workload is not None else None
    if stored:
        try:
            tier, source = parse_tier(stored), "workload"
        except ValueError:
            tier, source = DataTier.LOCAL, f"invalid stored data_boundary {stored!r}, treated as local"
    if _is_private(workload) and (tier is None or tier.rank > DataTier.LOCAL.rank):
        tier, source = DataTier.LOCAL, "privacy_class=private"
    if tier is None:
        if project is None and workload is not None:
            project = getattr(workload, "project", None)
        tier, source = (policy or load_policy()).project_default(project, env)
    if tier is None:
        tier, source = DataTier.UNVETTED, FLEET_DEFAULT_SOURCE
    if per_call is not None and per_call != "":
        try:
            call_tier = parse_tier(per_call)
        except ValueError:
            call_tier = DataTier.LOCAL
        if call_tier.rank < tier.rank:
            tier, source = call_tier, "call"
    return ResolvedBoundary(tier=tier, source=source)


def _is_private(workload: Any) -> bool:
    privacy = getattr(workload, "privacy_class", None)
    return str(getattr(privacy, "value", privacy or "")).lower() == "private"


# ---------------------------------------------------------------------------
# Refusal text (raised as somm.errors.SommPrivacyViolation by callers)


def refusal_message(
    *,
    workload: str | None,
    boundary: ResolvedBoundary,
    refused: list[RouteTier],
    context: str = "",
) -> str:
    """The canonical SOMM_PRIVACY_VIOLATION text for a boundary refusal."""
    routes = ", ".join(f"{r.route} ({r.tier.value})" for r in refused) or "none configured"
    where = f" for {context}" if context else ""
    name = f"workload {workload!r}" if workload else "this call"
    return (
        "SOMM_PRIVACY_VIOLATION\n\n"
        f"Problem: {name} has data boundary {boundary.tier.value!r} "
        f"(from {boundary.source}), and no admissible route remains{where}.\n"
        f"Cause: refused routes: {routes}. A boundary refusal is never retried "
        "and never falls through to a looser route.\n"
        "Fix: add a route within the boundary (e.g. a local Ollama), loosen the "
        "workload with `somm boundary set <workload> <tier>`, or declare the "
        "route's tier in ~/.somm/boundary.toml if your contract supports it. "
        "See `somm boundary routes`.\n"
        "Docs: docs/errors/SOMM_PRIVACY_VIOLATION.md"
    )


# Written as calls.error_kind on a boundary refusal row.
REFUSAL_ERROR_KIND = "SommPrivacyViolation"


__all__ = [
    "FLEET_DEFAULT_SOURCE",
    "audit",
    "HARNESS_PROVIDERS",
    "TIER_NAMES",
    "BoundaryPolicy",
    "DataTier",
    "ResolvedBoundary",
    "RouteTier",
    "admit",
    "boundary_path",
    "classify",
    "load_policy",
    "parse_tier",
    "refusal_message",
    "resolve_boundary",
]

# The ledger audit lives in its own module; re-exported here for old imports.
from somm_core.boundary_audit import audit  # noqa: E402

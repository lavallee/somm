"""Every ledger on this machine, who writes to it, and how far the mirror has it.

The read-only half (``ledger_facts``, ``ledgers_report``, ``fleet_status``)
opens each file with ``mode=ro`` and never migrates, creates, or locks
anything for writing: it is what to run instead of
``find ~ -name calls.sqlite``. The one writer here, ``mirror_sync``, copies
call rows a project ledger has and the mirror lacks; it is idempotent
(``INSERT OR IGNORE`` on the call id), commits in batches, and is safe to
interrupt and run again.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core import SCHEMA_VERSION, VERSION
from somm_core.cost import known_cost_sql
from somm_core.registry import registry_entries

LEDGER_NAME = "calls.sqlite"
RECENT_DAYS = 30
PRE_017 = "pre-0.17"
# Directories a scan never descends into: they hold code, not ledgers.
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "site-packages", ".tox"}


# ---------------------------------------------------------------------------
# small helpers


def _ro(path: Path, timeout: float = 5.0) -> sqlite3.Connection:
    """Open a ledger read-only; never creates, migrates, or writes."""
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=timeout)
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


def _columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")]


def _size(path: Path) -> int:
    total = 0
    for p in (path, path.with_name(path.name + "-wal")):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def is_temp_path(path: Path | str) -> bool:
    """True for a ledger under a temporary directory (it will vanish)."""
    raw = str(path)
    roots = {tempfile.gettempdir(), "/tmp", "/var/tmp", "/private/tmp"}
    return any(raw == r or raw.startswith(r.rstrip("/") + "/") for r in roots) or (
        "/pytest-" in raw
    )


def _version_key(v: str | None) -> tuple:
    if not v or v == PRE_017:
        return (0,)
    parts = []
    for piece in v.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_behind(version: str | None, installed: str = VERSION) -> bool:
    return _version_key(version) < _version_key(installed)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"  # pragma: no cover


def short_ts(ts: str | None) -> str:
    return ts[:16].replace("T", " ") if ts else "—"


def tilde(path: Path | str) -> str:
    raw, home = str(path), str(Path.home())
    return "~" + raw[len(home):] if raw == home or raw.startswith(home + "/") else raw


# ---------------------------------------------------------------------------
# facts about one file


def ledger_facts(path: Path, *, since_days: int = RECENT_DAYS) -> dict:
    """What one ledger holds. Never raises: an unreadable file says why."""
    path = Path(path)
    facts: dict = {
        "path": str(path),
        "exists": path.exists(),
        "temp": is_temp_path(path),
        "size_bytes": _size(path) if path.exists() else 0,
    }
    if not facts["exists"]:
        return facts
    since = (datetime.now(UTC) - timedelta(days=since_days)).isoformat()
    try:
        with _ro(path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            facts["schema_version"] = (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                if "schema_version" in tables
                else None
            )
            if "calls" not in tables:
                facts.update(rows=0, first_ts=None, last_ts=None, projects={})
                return facts
            cols = set(_columns(conn, "calls"))
            facts["rows"] = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            version_sql = "client_version" if "client_version" in cols else "NULL"
            unknown_sql = f"NOT {known_cost_sql()}" if "cost_source" in cols else "0"
            projects: dict[str, dict] = {}
            for name, n, first, last in conn.execute(
                "SELECT project, COUNT(*), MIN(ts), MAX(ts) FROM calls GROUP BY project"
            ):
                recent = conn.execute(
                    f"SELECT {version_sql}, COUNT(*), COALESCE(SUM({unknown_sql}), 0) "
                    f"FROM calls WHERE project = ? AND ts >= ? GROUP BY 1",
                    (name, since),
                ).fetchall()
                versions = {(v or PRE_017): c for v, c, _ in recent}
                projects[str(name)] = {
                    "calls": n,
                    "first_ts": first,
                    "last_ts": last,
                    "recent_calls": sum(versions.values()),
                    "recent_versions": dict(
                        sorted(versions.items(), key=lambda kv: _version_key(kv[0]), reverse=True)
                    ),
                    "recent_missing_client_version": versions.get(PRE_017, 0),
                    "recent_unknown_cost": sum(u for _, _, u in recent),
                }
            facts["projects"] = projects
            facts["first_ts"] = min((p["first_ts"] for p in projects.values()), default=None)
            facts["last_ts"] = max((p["last_ts"] for p in projects.values()), default=None)
            facts["last_worker_success"] = (
                conn.execute("SELECT MAX(last_success_at) FROM worker_heartbeat").fetchone()[0]
                if "worker_heartbeat" in tables
                else None
            )
    except sqlite3.Error as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def mirror_facts(mirror: Path, projects: Iterable[str]) -> dict:
    """``{"exists", "path", "size_bytes", "schema_version", "projects": {name:
    {"calls", "last_ts"}}}`` for the named projects, read-only."""
    mirror = Path(mirror)
    out: dict = {"path": str(mirror), "exists": mirror.exists(), "projects": {}}
    if not out["exists"]:
        return out
    out["size_bytes"] = _size(mirror)
    try:
        with _ro(mirror) as conn:
            out["schema_version"] = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]
            out["oldest_ts"] = conn.execute("SELECT MIN(ts) FROM calls").fetchone()[0]
            for name in sorted(set(projects)):
                n, last = conn.execute(
                    "SELECT COUNT(*), MAX(ts) FROM calls WHERE project = ?", (name,)
                ).fetchone()
                out["projects"][name] = {"calls": n, "last_ts": last}
    except sqlite3.Error as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# scanning a directory the user names


def scan_for_ledgers(root: Path) -> list[Path]:
    """Every ``calls.sqlite`` under ``root``, without following symlinks."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(Path(root).expanduser(), followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if LEDGER_NAME in filenames:
            found.append((Path(dirpath) / LEDGER_NAME).resolve())
    return sorted(found)


# ---------------------------------------------------------------------------
# the report


def ledgers_report(
    *, mirror: Path, scan: Iterable[Path] = (), since_days: int = RECENT_DAYS
) -> dict:
    """One entry per registered ledger file, plus the mirror and (with
    ``scan``) every unregistered ledger found under the named directories."""
    entries = registry_entries()
    by_path: dict[str, dict[str, str]] = {}
    for project, paths in entries.items():
        for raw, seen in paths.items():
            by_path.setdefault(str(Path(raw)), {})[project] = seen
    mirror_key = str(Path(mirror).resolve()) if Path(mirror).exists() else str(mirror)

    ledgers = []
    for raw in sorted(by_path):
        if raw == mirror_key:
            continue
        facts = ledger_facts(Path(raw), since_days=since_days)
        facts["registered_as"] = sorted(by_path[raw])
        facts["registered_last_seen"] = max(by_path[raw].values() or [""])
        ledgers.append(facts)

    names = {n for led in ledgers for n in led["registered_as"]}
    names |= {n for led in ledgers for n in led.get("projects", {})}
    mfacts = mirror_facts(Path(mirror), names)

    # Per project: rows across every ledger it is registered to, vs the mirror.
    ledger_calls: dict[str, int] = {}
    for led in ledgers:
        for name in led["registered_as"]:
            ledger_calls[name] = ledger_calls.get(name, 0) + led.get("projects", {}).get(
                name, {}
            ).get("calls", 0)
    for led in ledgers:
        led["flags"] = _flags(led, mfacts, ledger_calls)
        led["mirrored_through"] = _mirrored_through(led, mfacts)

    unregistered = []
    registered_paths = set(by_path) | {mirror_key}
    for root in scan:
        for found in scan_for_ledgers(Path(root)):
            if str(found) in registered_paths:
                continue
            facts = ledger_facts(found, since_days=since_days)
            mtime = datetime.fromtimestamp(found.stat().st_mtime, UTC)
            facts["modified"] = mtime.isoformat()
            facts["recent"] = mtime >= datetime.now(UTC) - timedelta(days=since_days)
            unregistered.append(facts)
            registered_paths.add(str(found))
    unregistered.sort(key=lambda f: (not f["recent"], f.get("last_ts") or "", f["path"]))

    return {
        "installed_version": VERSION,
        "installed_schema": SCHEMA_VERSION,
        "window_days": since_days,
        "ledgers": ledgers,
        "mirror": mfacts,
        "unregistered": unregistered,
        "scanned": [str(Path(r).expanduser()) for r in scan],
    }


def _mirrored_through(led: dict, mirror: dict) -> str | None:
    lasts = [
        mirror["projects"].get(n, {}).get("last_ts")
        for n in led.get("projects", {})
        if n in led["registered_as"]
    ]
    lasts = [t for t in lasts if t]
    return min(lasts) if lasts else None


def _flags(led: dict, mirror: dict, ledger_calls: dict[str, int]) -> list[str]:
    flags = []
    if not led["exists"]:
        return ["missing"]
    if led.get("error"):
        flags.append("unreadable")
    if len(led["registered_as"]) > 1:
        flags.append("shared")
    if led["temp"]:
        flags.append("temp")
    if mirror["exists"] and not mirror.get("error"):
        for name in led["registered_as"]:
            mine = led.get("projects", {}).get(name)
            if not mine:
                continue
            m = mirror["projects"].get(name, {})
            if not m.get("calls"):
                flags.append("not-mirrored")
                break
            if (m.get("last_ts") or "") < (mine.get("last_ts") or ""):
                flags.append("mirror-behind")
                break
        for name in led["registered_as"]:
            if mirror["projects"].get(name, {}).get("calls", 0) > ledger_calls.get(name, 0):
                flags.append("mirror-has-more")
                break
    return flags


# ---------------------------------------------------------------------------
# text rendering


def render_ledgers(report: dict) -> list[str]:
    lines = []
    header = (
        f"{'projects':<26} {'size':>8} {'schema':>6} {'rows':>9} {'oldest':<10} "
        f"{'last call':<16} {'clients (' + str(report['window_days']) + 'd)':<20} "
        f"{'mirrored through':<16}  flags / path"
    )
    lines.append(header)
    for led in sorted(report["ledgers"], key=lambda r: -r.get("size_bytes", 0)):
        names = ",".join(led["registered_as"])
        if not led["exists"]:
            lines.append(f"{names[:26]:<26} {'—':>8} {'—':>6} {'—':>9} {'':<10} {'':<16} "
                         f"{'':<20} {'':<16}  missing  {tilde(led['path'])}")
            continue
        versions: dict[str, int] = {}
        for p in led.get("projects", {}).values():
            for v, n in p["recent_versions"].items():
                versions[v] = versions.get(v, 0) + n
        clients = ", ".join(sorted(versions, key=_version_key, reverse=True)) or "—"
        flags = " ".join(led["flags"])
        lines.append(
            f"{names[:26]:<26} {human_size(led['size_bytes']):>8} "
            f"{str(led.get('schema_version') or '—'):>6} {led.get('rows', 0):>9} "
            f"{(led.get('first_ts') or '—')[:10]:<10} {short_ts(led.get('last_ts')):<16} "
            f"{clients[:20]:<20} {short_ts(led.get('mirrored_through')):<16}  "
            f"{(flags + '  ') if flags else ''}{tilde(led['path'])}"
        )
    m = report["mirror"]
    if m["exists"]:
        lines.append(
            f"{'(mirror)':<26} {human_size(m.get('size_bytes', 0)):>8} "
            f"{str(m.get('schema_version') or '—'):>6} {'':>9} "
            f"{(m.get('oldest_ts') or '—')[:10]:<10} {'':<16} {'':<20} {'':<16}  "
            f"{tilde(m['path'])}"
        )
    else:
        lines.append(f"(mirror)  none at {tilde(m['path'])}")

    if report["scanned"]:
        lines.append("")
        where = ", ".join(tilde(s) for s in report["scanned"])
        un = report["unregistered"]
        recent = [u for u in un if u["recent"]]
        lines.append(
            f"unregistered under {where}: {len(un)} ledger(s), {len(recent)} written in the "
            f"last {report['window_days']}d"
        )
        for u in un[:40]:
            names = ",".join(sorted(u.get("projects", {}))) or "—"
            lines.append(
                f"  {'recent' if u['recent'] else 'old   '} {names[:24]:<24} "
                f"{human_size(u['size_bytes']):>8} {u.get('rows', 0):>9} "
                f"{short_ts(u.get('last_ts')):<16}  {tilde(u['path'])}"
            )
        if len(un) > 40:
            lines.append(f"  ... and {len(un) - 40} more (--json for all)")
        if un:
            lines.append("  register one with `somm ledgers adopt PATH [--project NAME]`")

    lines.append("")
    lines.extend(_footer(report))
    return lines


def _footer(report: dict) -> list[str]:
    leds = report["ledgers"]
    count = lambda flag: sum(flag in led["flags"] for led in leds)  # noqa: E731
    total = sum(led.get("size_bytes", 0) for led in leds)
    out = [
        f"{len(leds)} registered ledger file(s), {human_size(total)}: {count('missing')} missing, "
        f"{count('shared')} shared by several names, {count('temp')} in a temp directory; "
        f"installed somm {report['installed_version']} (schema {report['installed_schema']})"
    ]
    behind = count("mirror-behind") + count("not-mirrored")
    if behind:
        out.append(
            f"mirror: {behind} ledger(s) hold calls the mirror does not; "
            "`somm mirror sync --dry-run` counts them, without --dry-run copies them"
        )
    if count("missing"):
        out.append("stale registrations: `somm ledgers forget PROJECT [--path PATH] --dry-run`")
    existing = [led for led in leds if led["exists"] and led.get("rows")]
    if existing:
        big = max(existing, key=lambda r: r["size_bytes"])
        out.append(
            "retention: nothing prunes on a schedule. Largest ledger since "
            f"{(big.get('first_ts') or '?')[:10]}:"
        )
        out.append(
            f"  SOMM_DB_DIR={tilde(Path(big['path']).parent)} somm prune --older-than 90 "
            "--dry-run   (then without --dry-run; --vacuum gives the space back)"
        )
        out.append("  somm prune --global --older-than 90 --dry-run   (the mirror)")
    return out


# ---------------------------------------------------------------------------
# fleet status: one line per project, problems first


def fleet_status(report: dict) -> list[dict]:
    """Per project, folded over every ledger it is registered to."""
    rows: dict[str, dict] = {}
    for led in report["ledgers"]:
        for name in led["registered_as"]:
            row = rows.setdefault(name, {
                "project": name, "ledgers": 0, "versions": {}, "recent_calls": 0,
                "missing_client_version": 0, "unknown_cost": 0, "schemas": set(),
                "last_worker_success": None, "last_ts": None, "flags": set(),
            })
            row["ledgers"] += 1
            if not led["exists"]:
                row["flags"].add("missing")
                continue
            if led.get("schema_version") is not None:
                row["schemas"].add(led["schema_version"])
            mine = led.get("projects", {}).get(name)
            if mine:
                for v, n in mine["recent_versions"].items():
                    row["versions"][v] = row["versions"].get(v, 0) + n
                row["recent_calls"] += mine["recent_calls"]
                row["missing_client_version"] += mine["recent_missing_client_version"]
                row["unknown_cost"] += mine["recent_unknown_cost"]
                row["last_ts"] = max(filter(None, [row["last_ts"], mine["last_ts"]]), default=None)
            lws = led.get("last_worker_success")
            row["last_worker_success"] = max(
                filter(None, [row["last_worker_success"], lws]), default=None
            )
            row["flags"] |= {f for f in led["flags"] if f in ("mirror-behind", "not-mirrored")}

    out = []
    for row in rows.values():
        problems = []
        behind = [v for v in row["versions"] if is_behind(v, report["installed_version"])]
        if behind:
            problems.append("behind")
        if row["schemas"] and min(row["schemas"]) < report["installed_schema"]:
            problems.append("schema")
        if row["missing_client_version"]:
            problems.append("no-client-version")
        if row["unknown_cost"]:
            problems.append("unknown-cost")
        problems += sorted(row["flags"])
        out.append({
            "project": row["project"],
            "ledgers": row["ledgers"],
            "recent_calls": row["recent_calls"],
            "versions": dict(
                sorted(row["versions"].items(), key=lambda kv: _version_key(kv[0]), reverse=True)
            ),
            "behind_installed": sorted(behind, key=_version_key),
            "schema_versions": sorted(row["schemas"]),
            "last_worker_success": row["last_worker_success"],
            "last_call": row["last_ts"],
            "unknown_cost_share": _share(row["unknown_cost"], row["recent_calls"]),
            "missing_client_version_share": _share(
                row["missing_client_version"], row["recent_calls"]
            ),
            "problems": problems,
        })
    out.sort(key=lambda r: (not r["problems"], r["recent_calls"] == 0, -r["recent_calls"],
                            r["project"]))
    return out


def _share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def render_fleet_status(rows: list[dict], *, window_days: int, installed: str,
                        schema: int) -> list[str]:
    def pct(x: float | None) -> str:
        if x is None:
            return "—"
        return "<1%" if 0 < x < 0.005 else f"{x:.0%}"

    bad = sum(1 for r in rows if r["problems"])
    lines = [
        f"fleet ({len(rows)} registered project(s), {window_days}d, installed {installed} / "
        f"schema {schema}): {bad} with problems"
    ]
    for r in rows:
        versions = ", ".join(r["versions"]) or "idle"
        schemas = "/".join(str(s) for s in r["schema_versions"]) or "—"
        loop = (r["last_worker_success"] or "never")[:10]
        lines.append(
            f"  {'PROBLEM' if r['problems'] else 'ok':<7} {r['project'][:24]:<24} "
            f"{versions[:24]:<24} schema {schemas:<6} loop ok {loop:<10} "
            f"unknown cost {pct(r['unknown_cost_share']):>4}  "
            f"no client_version {pct(r['missing_client_version_share']):>4}"
            + (f"  [{' '.join(r['problems'])}]" if r["problems"] else "")
        )
    return lines


# ---------------------------------------------------------------------------
# mirror sync — the one writer in this module


def mirror_sync(
    ledger: Path,
    mirror: Path,
    *,
    project: str | None = None,
    batch: int = 5000,
    dry_run: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Copy call rows (and their workload rows) the mirror lacks from one
    ledger. Returns ``{"missing": n}`` for a dry run, else ``{"copied": n,
    "workloads": n}``.

    The ledger is attached read-only. Each batch is its own transaction and
    ``INSERT OR IGNORE`` keys on the call id, so a rerun after an interrupt
    copies only what is still missing. Only columns both files have are
    copied, so a ledger on an older schema syncs into a newer mirror.
    """
    ledger, mirror = Path(ledger).resolve(), Path(mirror)
    where, params = ("AND s.project = ?", [project]) if project else ("", [])
    if dry_run:
        if not mirror.exists():
            with _ro(ledger) as conn:
                n = conn.execute(
                    f"SELECT COUNT(*) FROM calls s WHERE 1 {where}", params
                ).fetchone()[0]
            return {"missing": n}
        with _ro(mirror) as conn:
            conn.execute("ATTACH DATABASE ? AS src", (f"{ledger.as_uri()}?mode=ro",))
            n = conn.execute(
                f"SELECT COUNT(*) FROM src.calls s WHERE 1 {where} AND NOT EXISTS "
                "(SELECT 1 FROM main.calls m WHERE m.id = s.id)",
                params,
            ).fetchone()[0]
        return {"missing": n}

    from somm_core.repository import Repository

    Repository(mirror).close()  # creates or migrates the mirror, as any client does
    # uri=True so the ATTACH below may open the ledger with ?mode=ro.
    conn = sqlite3.connect(mirror.resolve().as_uri(), uri=True, isolation_level=None, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("ATTACH DATABASE ? AS src", (f"{ledger.as_uri()}?mode=ro",))
        wl_cols = [c for c in _columns(conn, "workloads", "src")
                   if c in set(_columns(conn, "workloads"))]
        call_cols = [c for c in _columns(conn, "calls", "src")
                     if c in set(_columns(conn, "calls"))]
        wl_list, call_list = ", ".join(wl_cols), ", ".join(call_cols)
        sel = ", ".join(f"s.{c}" for c in call_cols)
        wl_where = "WHERE s.project = ?" if project else ""
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"INSERT OR IGNORE INTO main.workloads ({wl_list}) "
            f"SELECT {', '.join('s.' + c for c in wl_cols)} FROM src.workloads s {wl_where}",
            params,
        )
        workloads = conn.execute("SELECT changes()").fetchone()[0]
        conn.execute("COMMIT")
        copied, last = 0, 0
        while True:
            hi = conn.execute(
                "SELECT MAX(rowid) FROM (SELECT rowid FROM src.calls WHERE rowid > ? "
                "ORDER BY rowid LIMIT ?)",
                (last, batch),
            ).fetchone()[0]
            if hi is None:
                break
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"INSERT OR IGNORE INTO main.calls ({call_list}) SELECT {sel} FROM src.calls s "
                f"WHERE s.rowid > ? AND s.rowid <= ? {where}",
                [last, hi, *params],
            )
            copied += conn.execute("SELECT changes()").fetchone()[0]
            conn.execute("COMMIT")
            last = hi
            if progress is not None:
                progress(copied, last)
    finally:
        conn.close()
    return {"copied": copied, "workloads": workloads}

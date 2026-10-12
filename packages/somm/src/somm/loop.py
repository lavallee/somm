"""The intelligence loop as something you can run.

Somm's loop — refresh model intel, grade sampled production calls against a
gold model, turn the grades into recommendations — has three workers and a
scheduler, and for most of 2026 nothing on this machine ran them: the loop
needed either the web service or a long-lived host process, and a project
that is a batch job or a CLI has neither. Three workloads opted into online
eval and were never graded.

``run_once`` is the loop as a one-shot, library-only command: it runs the
workers that are due for one project ledger and exits. ``install`` writes a
systemd user timer that calls it, so a project needs no resident process.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.registry import fleet_db_paths
from somm_core.repository import Repository

from somm.workers._heartbeat import beat
from somm.workers.inprocess import build_workers_factory

# How stale a worker's last success may be before `run_once` runs it again.
# Shadow grading is cheap to check (nothing to grade → nothing happens), so it
# runs every time; the other two only when they are actually stale.
DEFAULT_STALENESS: dict[str, timedelta | None] = {
    "shadow_eval": None,
    "model_intel": timedelta(hours=24),
    "agent": timedelta(days=7),
}
WORKER_ORDER = ("model_intel", "shadow_eval", "agent")


@dataclass(slots=True)
class LoopReport:
    project: str
    db_path: Path
    ran: dict[str, dict] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "project": self.project,
            "db_path": str(self.db_path),
            "ran": self.ran,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def last_success(repo: Repository, worker: str) -> datetime | None:
    with repo._open() as conn:
        row = conn.execute(
            "SELECT last_success_at FROM worker_heartbeat WHERE worker_name = ?", (worker,)
        ).fetchone()
    if not row or not row[0]:
        return None
    try:
        when = datetime.fromisoformat(str(row[0]).replace(" ", "T"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def run_once(
    cfg: Config,
    *,
    force: bool = False,
    only: tuple[str, ...] | None = None,
    staleness: dict[str, timedelta | None] | None = None,
    factory=None,
    now: datetime | None = None,
) -> LoopReport:
    """Run the due workers for one project ledger and record heartbeats.

    ``force`` runs every worker regardless of staleness; ``only`` restricts
    to named workers. ``factory`` and ``now`` are injection points for tests.
    """
    repo = Repository(cfg.db_path)
    report = LoopReport(project=cfg.project, db_path=cfg.db_path)
    staleness = staleness or DEFAULT_STALENESS
    now = now or datetime.now(UTC)
    make = factory or build_workers_factory(cfg, repo)
    for name in WORKER_ORDER:
        if only and name not in only:
            continue
        window = staleness.get(name)
        if not force and window is not None:
            last = last_success(repo, name)
            if last is not None and now - last < window:
                report.skipped[name] = f"ran {_ago(now - last)} ago"
                continue
        worker = make(name)
        if worker is None:
            report.skipped[name] = "no worker"
            continue
        try:
            summary = worker.run_once()
        except Exception as exc:  # noqa: BLE001 - one worker failing must not stop the others
            beat(repo, name, False)
            report.errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        ok = not (isinstance(summary, dict) and summary.get("errors"))
        beat(repo, name, ok)
        report.ran[name] = summary if isinstance(summary, dict) else {"result": summary}
    return report


def run_fleet(*, force: bool = False, only: tuple[str, ...] | None = None) -> list[LoopReport]:
    """``run_once`` for every registered project ledger that still exists."""
    reports = []
    for db_path in fleet_db_paths():
        cfg = load_config(cwd=db_path.parent.parent)
        cfg.db_dir = db_path.parent
        reports.append(run_once(cfg, force=force, only=only))
    return reports


def _ago(delta: timedelta) -> str:
    s = int(delta.total_seconds())
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


# ---------------------------------------------------------------------------
# systemd user units


def somm_executable() -> str:
    """The `somm` the timer should run: the one beside this interpreter when
    there is one (a project's own venv), else whatever PATH says."""
    import sys

    beside = Path(sys.executable).with_name("somm")
    if beside.exists():
        return str(beside)
    return shutil.which("somm") or "/usr/bin/env somm"


def unit_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def unit_names(project: str) -> tuple[str, str]:
    return f"somm-loop@{project}.service", f"somm-loop@{project}.timer"


def render_units(*, somm_bin: str, every: str = "15min") -> dict[str, str]:
    """The service/timer template pair for ``somm loop run --project %i``.

    The templates are shared by every project; anything per project (the
    ledger location) goes in a drop-in, see :func:`render_dropin`.
    """
    service = f"""[Unit]
Description=somm intelligence loop — one pass for project %i
# One-shot: refreshes model intel when stale, grades sampled calls, and
# emits recommendations when due, then exits. Written by `somm loop install`.
After=network-online.target

[Service]
Type=oneshot
ExecStart={somm_bin} loop run --project %i
StandardOutput=journal
StandardError=journal
"""
    timer = f"""[Unit]
Description=somm intelligence loop — periodic pass for project %i

[Timer]
OnBootSec=5min
OnUnitActiveSec={every}
Persistent=true

[Install]
WantedBy=timers.target
"""
    return {"somm-loop@.service": service, "somm-loop@.timer": timer}


def render_dropin(db_dir: Path) -> str:
    """Per-project drop-in: where this project's ledger lives."""
    return f"""# Written by `somm loop install`. The template unit is shared; this
# names the ledger for this one project.
[Service]
Environment=SOMM_DB_DIR={db_dir}
"""


def install(
    cfg: Config,
    *,
    every: str = "15min",
    target_dir: Path | None = None,
    enable: bool = True,
    systemctl=None,
) -> list[Path]:
    """Write the template units and a per-project drop-in, and enable the timer.

    Returns the files written. ``systemctl`` is injectable; when it is
    ``None`` and no ``systemctl`` is on PATH, the units are written and the
    operator is told how to enable them.
    """
    target = target_dir or unit_dir()
    target.mkdir(parents=True, exist_ok=True)
    somm_bin = somm_executable()
    written = []
    for name, body in render_units(somm_bin=somm_bin, every=every).items():
        path = target / name
        path.write_text(body)
        written.append(path)
    service, timer = unit_names(cfg.project)
    dropin_dir = target / f"{service}.d"
    dropin_dir.mkdir(parents=True, exist_ok=True)
    dropin = dropin_dir / "10-somm.conf"
    dropin.write_text(render_dropin(cfg.db_dir.resolve()))
    written.append(dropin)
    if enable:
        run = systemctl or _systemctl
        run(["daemon-reload"])
        run(["enable", "--now", timer])
    return written


def uninstall(cfg: Config, *, systemctl=None) -> None:
    _, timer = unit_names(cfg.project)
    run = systemctl or _systemctl
    run(["disable", "--now", timer])


def status(cfg: Config, *, systemctl=None) -> dict:
    repo = Repository(cfg.db_path) if cfg.db_path.exists() else None
    heartbeats: dict[str, str | None] = {}
    if repo is not None:
        for name in WORKER_ORDER:
            last = last_success(repo, name)
            heartbeats[name] = last.isoformat() if last else None
    _, timer = unit_names(cfg.project)
    timer_state = None
    try:
        run = systemctl or _systemctl
        out = run(["is-active", timer], capture=True)
        timer_state = (out or "").strip() or None
    except Exception:  # noqa: BLE001
        timer_state = None
    return {"project": cfg.project, "timer": timer, "timer_state": timer_state, "heartbeats": heartbeats}


def _systemctl(args: list[str], *, capture: bool = False) -> str | None:
    binary = shutil.which("systemctl")
    if binary is None:
        raise RuntimeError("systemctl not found; enable the timer with your init system")
    proc = subprocess.run(
        [binary, "--user", *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0 and args[0] != "is-active":
        raise RuntimeError(f"systemctl --user {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout if capture else None


__all__ = [
    "DEFAULT_STALENESS",
    "LoopReport",
    "install",
    "last_success",
    "render_dropin",
    "render_units",
    "somm_executable",
    "run_fleet",
    "run_once",
    "status",
    "uninstall",
]

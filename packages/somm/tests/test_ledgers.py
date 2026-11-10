"""`somm ledgers`, `somm fleet status`, and `somm mirror sync`, on temp files only."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from somm import ledgers
from somm.cli import main
from somm.providers.base import ProviderHealth
from somm_core import registry
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


def _calls(repo: Repository, project: str, n: int, *, days_ago: float = 0,
           client_version: str | None = "0.1.0", cost_source: str | None = None) -> list[str]:
    wl = repo.register_workload(name="w", project=project)
    ts = datetime.now(UTC) - timedelta(days=days_ago)
    batch = [
        Call(
            id=str(uuid.uuid4()), ts=ts, project=project, workload_id=wl.id, prompt_id=None,
            provider="fake", model="m", tokens_in=1, tokens_out=1, latency_ms=1, cost_usd=0.0,
            outcome=Outcome.OK, error_kind=None, prompt_hash="p", response_hash="r",
            client_version=client_version, cost_source=cost_source,
        )
        for _ in range(n)
    ]
    repo.write_calls_batch(batch)
    return [c.id for c in batch]


@pytest.fixture
def fleet(tmp_path):
    """Three ledgers and a mirror:

    - ``alpha``: 5 calls, mirrored up to the first 3 (the mirror fell behind);
    - ``beta`` and ``beta-scratch`` share one file, never mirrored;
    - ``gone``: registered to a path that no longer exists.
    """
    mirror = Repository(Path(os.environ["SOMM_GLOBAL_PATH"]))
    alpha = Repository(tmp_path / "alpha" / ".somm" / "calls.sqlite")
    old = _calls(alpha, "alpha", 3, days_ago=40)
    _calls(alpha, "alpha", 2, client_version=None, cost_source="unknown:fake")
    with mirror._open() as m, alpha._open() as a:
        rows = a.execute(
            "SELECT id, ts, project, workload_id, provider, model, prompt_hash, response_hash "
            f"FROM calls WHERE id IN ({','.join('?' * len(old))})", old,
        ).fetchall()
        m.executemany(
            "INSERT INTO calls (id, ts, project, workload_id, provider, model, prompt_hash, "
            "response_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows,
        )
    beta = Repository(tmp_path / "beta" / ".somm" / "calls.sqlite")
    _calls(beta, "beta", 4)
    registry.adopt_ledger("alpha", alpha.db_path)
    registry.adopt_ledger("beta", beta.db_path)
    registry.adopt_ledger("beta-scratch", beta.db_path)
    reg = registry.registry_path()
    data = json.loads(reg.read_text())
    data["projects"]["gone"] = {"db_path": str(tmp_path / "gone" / "calls.sqlite"),
                                "last_seen": "2026-01-01T00:00:00+00:00"}
    reg.write_text(json.dumps(data))
    return {"alpha": alpha, "beta": beta, "mirror": mirror, "tmp": tmp_path}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ledgers_report_flags_and_reads_without_writing(fleet, capsys):
    alpha = fleet["alpha"].db_path
    fleet["alpha"].close()
    before = _digest(alpha)

    assert main(["ledgers", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    by_name = {",".join(r["registered_as"]): r for r in report["ledgers"]}

    a = by_name["alpha"]
    assert a["rows"] == 5 and a["schema_version"] == report["installed_schema"]
    # The three 0.1.0 calls are 40 days old: outside the 30-day window.
    assert a["projects"]["alpha"]["recent_versions"] == {"pre-0.17": 2}
    assert a["first_ts"] < a["last_ts"]
    assert "mirror-behind" in a["flags"]
    assert a["mirrored_through"] < a["last_ts"]
    assert set(by_name["beta,beta-scratch"]["flags"]) >= {"shared", "not-mirrored"}
    assert by_name["gone"]["flags"] == ["missing"]
    assert all(r["temp"] for r in report["ledgers"] if r["exists"])  # pytest tmp
    assert _digest(alpha) == before  # read-only

    assert main(["ledgers"]) == 0
    out = capsys.readouterr().out
    assert "(mirror)" in out
    assert "somm mirror sync --dry-run" in out
    assert "somm prune --older-than 90 --dry-run" in out
    assert "somm ledgers forget" in out


def test_scan_lists_only_unregistered_ledgers(fleet, capsys):
    stray = Repository(fleet["tmp"] / "scratch" / "x" / ".somm" / "calls.sqlite")
    _calls(stray, "stray", 1)
    (fleet["tmp"] / "scratch" / "node_modules" / ".somm").mkdir(parents=True)
    (fleet["tmp"] / "scratch" / "node_modules" / ".somm" / "calls.sqlite").write_bytes(b"")

    assert main(["ledgers", "--scan", str(fleet["tmp"]), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [u["path"] for u in report["unregistered"]] == [str(stray.db_path.resolve())]
    assert report["unregistered"][0]["recent"] is True
    assert list(report["unregistered"][0]["projects"]) == ["stray"]


def test_forget_and_adopt_from_the_cli(fleet, capsys):
    assert main(["ledgers", "forget", "gone", "--dry-run"]) == 0
    assert "would forget gone" in capsys.readouterr().out
    assert "gone" in registry.registry_entries()
    assert main(["ledgers", "forget", "gone"]) == 0
    assert "gone" not in registry.registry_entries()
    assert main(["ledgers", "forget", "gone"]) == 2

    stray = Repository(fleet["tmp"] / "stray" / "calls.sqlite")
    _calls(stray, "s1", 1)
    _calls(stray, "s2", 1)
    assert main(["ledgers", "adopt", str(stray.db_path), "--dry-run"]) == 0
    assert capsys.readouterr().out.count("would register") == 2
    assert "s1" not in registry.registry_entries()
    assert main(["ledgers", "adopt", str(stray.db_path), "--project", "s1"]) == 0
    assert set(registry.registry_entries()) >= {"s1"} and "s2" not in registry.registry_entries()
    assert main(["ledgers", "adopt", str(fleet["tmp"] / "nope.sqlite")]) == 2


def test_fleet_status_puts_problems_first_and_doctor_global_prints_it(fleet, capsys, monkeypatch):
    rows = ledgers.fleet_status(ledgers.ledgers_report(mirror=fleet["mirror"].db_path))
    assert rows[0]["problems"]  # problems first
    alpha = next(r for r in rows if r["project"] == "alpha")
    assert set(alpha["problems"]) >= {"behind", "no-client-version", "unknown-cost",
                                      "mirror-behind"}
    assert alpha["unknown_cost_share"] == 1.0  # both recent calls are unpriced
    assert alpha["missing_client_version_share"] == 1.0

    assert main(["fleet", "status"]) == 0
    status = capsys.readouterr().out
    assert status.startswith("fleet (")
    assert main(["fleet"]) == 0
    assert capsys.readouterr().out == status

    monkeypatch.setattr(
        "somm.cli.OllamaProvider.health", lambda self: ProviderHealth(available=True, detail="ok")
    )
    main(["doctor", "--global"])
    assert status.splitlines()[1] in capsys.readouterr().out


def test_mirror_sync_copies_what_is_missing_and_is_safe_to_rerun(fleet, capsys):
    mirror = fleet["mirror"]

    def count(project):
        with mirror._open() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM calls WHERE project = ?", (project,)
            ).fetchone()[0]

    assert main(["mirror", "sync", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would copy 6 call row(s)" in out  # 2 alpha + 4 beta
    assert count("alpha") == 3

    assert main(["mirror", "sync", "--project", "alpha"]) == 0
    assert "copied 2 call row(s)" in capsys.readouterr().out
    assert (count("alpha"), count("beta")) == (5, 0)

    # An interrupt mid-run keeps the batches already committed; a rerun
    # copies only the rest.
    def stop_after_first(copied, _last):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ledgers.mirror_sync(fleet["beta"].db_path, mirror.db_path, batch=1,
                            progress=stop_after_first)
    assert count("beta") == 1
    assert main(["mirror", "sync"]) == 0
    assert "copied 3 call row(s)" in capsys.readouterr().out
    assert main(["mirror", "sync"]) == 0
    assert "copied 0 call row(s)" in capsys.readouterr().out
    assert count("beta") == 4
    with mirror._open() as conn:
        # Every copied call's workload resolves to a name in the mirror (both
        # fixtures declare `w` with no schemas, so they share one id and row).
        assert conn.execute(
            "SELECT COUNT(*) FROM calls c LEFT JOIN workloads w ON w.id = c.workload_id "
            "WHERE w.id IS NULL"
        ).fetchone()[0] == 0

    assert main(["mirror", "sync", "--project", "nobody"]) == 2

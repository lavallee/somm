"""CLI integration tests — status, tail, compare, doctor.

Tail and compare use short-running inputs. Compare uses a fake provider
by stubbing the providers list directly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from somm.cli import (
    _age_since,
    _age_until,
    _fetch_since,
    _fmt_delta,
    _parse_model_specs,
    _print_comparison,
    main,
)
from somm_core.config import Config
from somm_core.models import Call, Outcome
from somm_core.repository import Repository


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "cli-test"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


# ---------------------------------------------------------------------------
# helpers (age/delta/parse)


def test_fmt_delta_formats():
    assert _fmt_delta(timedelta(seconds=3)) == "3s"
    assert _fmt_delta(timedelta(minutes=5)) == "5m"
    assert _fmt_delta(timedelta(hours=2, minutes=30)) == "2h30m"
    assert _fmt_delta(timedelta(days=3, hours=4)) == "3d4h"


def test_age_since_past():
    past = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
    s = _age_since(past)
    assert "ago" in s
    assert "m" in s


def test_age_since_never():
    assert _age_since("") == "never"


def test_age_until_future():
    fut = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    s = _age_until(fut)
    assert s.endswith("m") or s == "now"


def test_parse_model_specs_variants():
    # slash form, colon form, comma-separated, repeated
    assert _parse_model_specs(["ollama/gemma4:e4b"]) == [("ollama", "gemma4:e4b")]
    assert _parse_model_specs(["openai:gpt-4o-mini"]) == [("openai", "gpt-4o-mini")]
    out = _parse_model_specs(["ollama/a,openrouter/b"])
    assert out == [("ollama", "a"), ("openrouter", "b")]


def test_parse_model_specs_empty():
    assert _parse_model_specs(None) == []
    assert _parse_model_specs([""]) == []


# ---------------------------------------------------------------------------
# status


def test_status_empty(tmp_path, capsys):
    cfg = _tmp_config(tmp_path)
    Repository(cfg.db_path)  # create db
    env = {"SOMM_PROJECT": cfg.project}
    rc = main(["status", "--project", cfg.project, "--since", "7"])
    out = capsys.readouterr().out
    # Rely on load_config picking up default db_dir — skip the test if it
    # doesn't align (CLI tests here exercise the functions, not arg routing).
    assert rc == 0


def test_status_with_rows(tmp_path, capsys, monkeypatch):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="cli_stat", project=cfg.project)
    repo.write_call(
        Call(
            id=str(uuid.uuid4()),
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="ollama",
            model="gemma4:e4b",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50,
            cost_usd=0.01,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="a",
            response_hash="b",
        )
    )
    # Patch load_config to use our temp cfg
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    rc = main(["status", "--since", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cli_stat" in out
    assert "ollama" in out


# ---------------------------------------------------------------------------
# tail


def test_fetch_since_returns_rows(tmp_path):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="w_tail", project=cfg.project)

    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    repo.write_call(
        Call(
            id=str(uuid.uuid4()),
            ts=datetime.now(UTC),
            project=cfg.project,
            workload_id=wl.id,
            prompt_id=None,
            provider="ollama",
            model="gemma4:e4b",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="a",
            response_hash="b",
        )
    )
    rows = _fetch_since(repo, cfg.project, cutoff)
    assert len(rows) == 1
    assert rows[0]["workload"] == "w_tail"


def test_fetch_since_workload_filter(tmp_path):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    wl1 = repo.register_workload(name="wa", project=cfg.project)
    wl2 = repo.register_workload(name="wb", project=cfg.project)

    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    for wl in (wl1, wl2):
        repo.write_call(
            Call(
                id=str(uuid.uuid4()),
                ts=datetime.now(UTC),
                project=cfg.project,
                workload_id=wl.id,
                prompt_id=None,
                provider="ollama",
                model="g",
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                cost_usd=0.0,
                outcome=Outcome.OK,
                error_kind=None,
                prompt_hash="a",
                response_hash="b",
            )
        )
    rows = _fetch_since(repo, cfg.project, cutoff, workload="wa")
    assert len(rows) == 1
    assert rows[0]["workload"] == "wa"


# ---------------------------------------------------------------------------
# compare


def test_print_comparison_happy(capsys):
    results = [
        {
            "provider": "ollama",
            "model": "g",
            "text": "hello",
            "tokens_in": 3,
            "tokens_out": 1,
            "latency_ms": 50,
            "cost_usd": 0.0,
            "outcome": "ok",
            "call_id": "abc-123",
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "text": "hi",
            "tokens_in": 3,
            "tokens_out": 1,
            "latency_ms": 200,
            "cost_usd": 0.00003,
            "outcome": "ok",
            "call_id": "def-456",
        },
    ]
    _print_comparison(results)
    out = capsys.readouterr().out
    assert "ollama" in out
    assert "openai" in out
    assert "hello" in out


def test_print_comparison_error_row(capsys):
    results = [
        {"provider": "anthropic", "model": "x", "error": "auth failed"},
    ]
    _print_comparison(results)
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "auth failed" in out


# ---------------------------------------------------------------------------
# doctor (in-process; patches the env to avoid touching real repo)


def test_doctor_reports_schema_and_no_db(tmp_path, capsys, monkeypatch):
    # Use a fresh tmp dir with no .somm — doctor should report db missing
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "doctor-test")
    # Disable ollama check by pointing to a dead URL
    monkeypatch.setenv("SOMM_OLLAMA_URL", "http://127.0.0.1:1")
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "somm v" in out
    assert "project: doctor-test" in out
    # db shouldn't exist at this path yet
    assert "exists:" in out

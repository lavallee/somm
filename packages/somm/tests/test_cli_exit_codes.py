"""Exit-code contract: 0 success, 1 ran and failed, 2 bad usage / named thing missing."""

from __future__ import annotations

import json

import somm.client as client_mod
from somm.cli import main
from somm_core.models import Outcome, SommResult


class _FailingLLM:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, **kwargs):
        return SommResult(
            text="",
            provider="fake",
            model="m",
            tokens_in=0,
            tokens_out=0,
            latency_ms=1,
            cost_usd=0.0,
            call_id="c1",
            outcome=Outcome.UPSTREAM_ERROR,
            error_kind="upstream_5xx",
        )

    def close(self):
        pass


def test_bench_exits_1_when_no_call_succeeded(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(client_mod, "SommLLM", _FailingLLM)
    rc = main(["bench", "latency", "hi", "--iterations", "2", "--provider", "fake", "--json"])
    cap = capsys.readouterr()
    assert rc == 1, cap.err
    assert json.loads(cap.out)["runs"][0]["outcome"] == "upstream_error", cap.err


def test_compare_exits_1_when_every_model_failed(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["compare", "hi", "--models", "nonesuch/m"]) == 1
    assert "not configured" in capsys.readouterr().out


def test_alias_write_without_a_ledger_exits_1_but_listing_is_fine(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["alias", "seed", "--project", "ghost"]) == 1
    assert main(["alias", "set", "p", "m", "canon", "--project", "ghost"]) == 1
    assert main(["alias", "list", "--project", "ghost"]) == 0
    capsys.readouterr()


def test_missing_workload_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["workload", "show", "nope", "--project", "ghost"]) == 2
    capsys.readouterr()

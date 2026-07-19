"""CLI integration tests — status, tail, compare, doctor.

Tail and compare use short-running inputs. Compare uses a fake provider
by stubbing the providers list directly.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from somm.cli import (
    WORKLOAD_EXAMPLES,
    _age_since,
    _age_until,
    _dataset_judge,
    _fetch_since,
    _fmt_delta,
    _load_eval_judge_config,
    _parse_model_specs,
    _print_comparison,
    build_parser,
    main,
)
from somm.providers.base import ProviderHealth
from somm_core.config import Config
from somm_core.models import Call, Outcome, SommResult
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


def test_eval_judge_config_requires_explicit_panel_and_quorum(tmp_path):
    path = tmp_path / "judge.json"
    path.write_text(json.dumps({
        "min_judges": 1,
        "criteria": ["correctness"],
        "panel": [{"provider": "minimax", "model": "MiniMax-M3"}],
    }))

    config = _load_eval_judge_config(str(path))

    assert config["min_judges"] == 1
    assert config["panel"][0]["provider"] == "minimax"


def test_dataset_judge_pins_every_panel_member_without_fallback():
    class FakeLLM:
        def __init__(self):
            self.requests = []

        def generate(self, **kwargs):
            self.requests.append(kwargs)
            return SommResult(
                text=json.dumps({
                    "criteria": [{
                        "name": "correctness", "pass": True, "reason": "grounded",
                    }]
                }),
                provider=kwargs["provider"],
                model=kwargs["model"],
                tokens_in=10,
                tokens_out=5,
                latency_ms=10,
                cost_usd=0.0,
                call_id=str(uuid.uuid4()),
            )

    llm = FakeLLM()
    judge = _dataset_judge(llm, workload="qa", config={
        "min_judges": 2,
        "criteria": ["correctness"],
        "panel": [
            {"provider": "minimax", "model": "MiniMax-M2.7-highspeed"},
            {"provider": "openrouter", "model": "nvidia/model:free"},
        ],
    })

    grade = judge(
        SimpleNamespace(prompt_body="question", expected_response_body="answer"),
        SimpleNamespace(text="answer"),
    )

    assert grade.score == 1.0
    assert grade.reason["quorum"] is True
    assert len(grade.call_ids) == 2
    assert all(request["no_fallback"] is True for request in llm.requests)
    assert [(r["provider"], r["model"]) for r in llm.requests] == [
        ("minimax", "MiniMax-M2.7-highspeed"),
        ("openrouter", "nvidia/model:free"),
    ]


# ---------------------------------------------------------------------------
# status


def test_status_empty(tmp_path, capsys):
    cfg = _tmp_config(tmp_path)
    Repository(cfg.db_path)  # create db
    rc = main(["status", "--project", cfg.project, "--since", "7"])
    capsys.readouterr()  # drain captured output
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
            ttft_ms=10,
            cache_tokens_in=4,
            cache_tokens_out=1,
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


def test_status_json_with_rows(tmp_path, capsys, monkeypatch):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="cli_stat_json", project=cfg.project)
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
            ttft_ms=10,
            cache_tokens_in=4,
            cache_tokens_out=1,
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    rc = main(["status", "--since", "1", "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert data["scope"] == "project"
    assert data["count"] == 1
    row = data["rows"][0]
    assert row["workload"] == "cli_stat_json"
    assert row["p95_latency_ms"] == 50
    assert row["p95_ttft_ms"] == 10
    assert row["tpot_ms"] == 10
    assert row["input_tokens_per_second"] == 200
    assert row["output_tokens_per_second"] == 100
    assert row["total_tokens_per_second"] == 300
    assert row["requests_per_second"] == 20
    assert row["cache_tokens_in"] == 4
    assert row["cache_tokens_out"] == 1
    assert row["cache_read_ratio"] == 0.4
    assert row["goodput_under_slo"] is None


def test_generate_json_uses_somm_llm(tmp_path, capsys, monkeypatch):
    cfg = _tmp_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    class FakeLLM:
        def __init__(self, config):
            self.config = config

        def generate(self, prompt, **kwargs):
            assert prompt == "hello"
            assert kwargs["workload"] == "cli_generate"
            assert kwargs["provider"] == "fake"
            return SommResult(
                text="world",
                provider="fake",
                model="fake-model",
                tokens_in=1,
                tokens_out=1,
                latency_ms=5,
                cost_usd=0.0,
                call_id="call-1",
            )

        def close(self):
            pass

    import somm.client as client_mod

    monkeypatch.setattr(client_mod, "SommLLM", FakeLLM)

    rc = main(
        [
            "generate",
            "hello",
            "--workload",
            "cli_generate",
            "--provider",
            "fake",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert data["ok"] is True
    assert data["text"] == "world"
    assert data["call_id"] == "call-1"


def test_cache_advice_json_flags_low_cache_reuse(tmp_path, capsys, monkeypatch):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    wl = repo.register_workload(name="cache_heavy", project=cfg.project)
    for idx in range(2):
        repo.write_call(
            Call(
                id=str(uuid.uuid4()),
                ts=datetime.now(UTC),
                project=cfg.project,
                workload_id=wl.id,
                prompt_id=None,
                provider="fake",
                model="m",
                tokens_in=1_000,
                tokens_out=10,
                latency_ms=100,
                cost_usd=0.0,
                outcome=Outcome.OK,
                error_kind=None,
                prompt_hash=f"p-{idx}",
                response_hash=f"r-{idx}",
                cache_tokens_in=0,
                cache_tokens_out=0,
            )
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    rc = main(["cache-advice", "--project", cfg.project, "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert data["count"] == 1
    row = data["rows"][0]
    assert row["workload"] == "cache_heavy"
    assert row["tokens_in"] == 2_000
    assert row["cache_read_ratio"] == 0.0
    assert row["issue"] == "no_cache_reads"


def test_bench_latency_json_uses_somm_llm(tmp_path, capsys, monkeypatch):
    cfg = _tmp_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", cfg.project)

    class FakeLLM:
        instances = []

        def __init__(self, config):
            self.config = config
            self.calls = 0
            self.seen = []
            FakeLLM.instances.append(self)

        def generate(self, prompt, **kwargs):
            self.calls += 1
            self.seen.append((prompt, kwargs))
            latency = 100 * self.calls
            ttft = 20 if self.calls == 1 else 50
            tokens_out = 11 if self.calls == 1 else 21
            return SommResult(
                text="ok",
                provider="fake",
                model="fake-model",
                tokens_in=5,
                tokens_out=tokens_out,
                latency_ms=latency,
                cost_usd=0.0,
                call_id=f"call-{self.calls}",
                ttft_ms=ttft,
            )

        def close(self):
            pass

    import somm.client as client_mod

    monkeypatch.setattr(client_mod, "SommLLM", FakeLLM)

    rc = main(["bench", "latency", "hello", "--iterations", "2", "--provider", "fake", "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert data["mode"] == "latency"
    assert data["workload"] == "bench_latency"
    assert data["summary"]["latency_ms"]["p50"] == 100
    assert data["summary"]["latency_ms"]["p95"] == 200
    assert data["summary"]["ttft_ms"]["p95"] == 50
    assert data["summary"]["tpot_ms"]["mean"] == 7.75
    assert [row["call_id"] for row in data["runs"]] == ["call-1", "call-2"]
    assert FakeLLM.instances[0].seen[0][1]["workload"] == "bench_latency"
    assert FakeLLM.instances[0].seen[0][1]["provider"] == "fake"


def test_generate_json_error_envelope(capsys):
    rc = main(["generate", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.err)

    assert rc == 2
    assert data["ok"] is False
    assert data["error"]["type"] == "ValueError"
    assert "prompt is required" in data["error"]["message"]


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
# workload


def test_workload_add_applies_structured_extraction_template(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(
        [
            "workload",
            "add",
            "extract_entities",
            "--project",
            "cli-workload",
            "--description",
            "Extract entities",
            "--privacy-class",
            "private",
            "--from-example",
            "structured-extraction",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "registered workload 'extract_entities'" in out
    assert "privacy_class: private" in out
    assert "input_schema: yes" in out
    assert "output_schema: yes" in out

    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    wl = repo.workload_by_name("extract_entities", "cli-workload")
    assert wl is not None
    assert wl.description == "Extract entities"
    assert wl.privacy_class.value == "private"
    assert wl.input_schema == WORKLOAD_EXAMPLES["structured-extraction"]["input_schema"]
    assert wl.output_schema == WORKLOAD_EXAMPLES["structured-extraction"]["output_schema"]


def test_workload_add_freeform_list_and_show(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "workload",
                "add",
                "drafting",
                "--project",
                "cli-workload",
                "--from-example",
                "freeform",
            ]
        )
        == 0
    )
    capsys.readouterr()

    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    wl = repo.workload_by_name("drafting", "cli-workload")
    assert wl is not None
    repo.write_call(
        Call(
            id=str(uuid.uuid4()),
            ts=datetime.now(UTC),
            project="cli-workload",
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

    assert main(["workload", "list", "--project", "cli-workload"]) == 0
    list_out = capsys.readouterr().out
    assert "drafting" in list_out
    assert "internal" in list_out
    assert "       1" in list_out

    assert main(["workload", "show", "drafting", "--project", "cli-workload"]) == 0
    show_out = capsys.readouterr().out
    assert "name: drafting" in show_out
    assert "privacy_class: internal" in show_out
    assert "call_count: 1" in show_out
    assert "max_p95_latency_ms: —" in show_out
    assert "max_p95_ttft_ms: —" in show_out
    assert "max_tpot_ms: —" in show_out
    assert "input_schema:\n  —" in show_out
    assert "output_schema:\n  —" in show_out

    assert (
        main(
            [
                "workload",
                "set-constraints",
                "drafting",
                "--project",
                "cli-workload",
                "--max-p95-latency-ms",
                "400",
                "--max-p95-ttft-ms",
                "120",
                "--max-tpot-ms",
                "18.5",
                "--max-capability-failure-rate",
                "0.05",
                "--max-cost-per-call-usd",
                "0.01",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["workload", "show", "drafting", "--project", "cli-workload"]) == 0
    show_out = capsys.readouterr().out
    assert "max_p95_latency_ms: 400" in show_out
    assert "max_p95_ttft_ms: 120" in show_out
    assert "max_tpot_ms: 18.5" in show_out
    assert "max_capability_failure_rate: 0.05" in show_out
    assert "max_cost_per_call_usd: 0.01" in show_out


def test_eval_promote_call_creates_dataset(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    wl = repo.register_workload(name="eval_w", project="cli-eval")
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project="cli-eval",
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
    repo.write_sample(call_id, "prompt body", "expected response")

    rc = main(
        [
            "eval",
            "promote-call",
            call_id,
            "--dataset",
            "golden",
            "--project",
            "cli-eval",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "promoted call" in out
    assert "dataset_id:" in out
    dataset = repo.get_dataset(project="cli-eval", workload_id=wl.id, name="golden")
    assert dataset is not None
    items = repo.dataset_items(dataset.id)
    assert len(items) == 1
    assert items[0].prompt_body == "prompt body"
    assert items[0].expected_response_body == "expected response"


def test_eval_run_uses_dataset_and_returns_gate_status(tmp_path, capsys, monkeypatch):
    import somm as somm_pkg

    monkeypatch.chdir(tmp_path)
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    wl = repo.register_workload(name="eval_w", project="cli-eval")
    source_call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=source_call_id,
            ts=datetime.now(UTC),
            project="cli-eval",
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
    repo.write_sample(source_call_id, "prompt body", "expected response")
    repo.promote_call_to_dataset(source_call_id, "golden", project="cli-eval")

    class _FakeEvalLLM:
        def __init__(self, config):
            self.config = config
            self.repo = Repository(config.db_path)

        def generate(self, prompt, workload, **_kwargs):
            call_id = str(uuid.uuid4())
            wl_row = self.repo.workload_by_name(workload, self.config.project)
            self.repo.write_call(
                Call(
                    id=call_id,
                    ts=datetime.now(UTC),
                    project=self.config.project,
                    workload_id=wl_row.id,
                    prompt_id=None,
                    provider="fake",
                    model="eval",
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
            return SommResult(
                text="expected response",
                provider="fake",
                model="eval",
                tokens_in=1,
                tokens_out=1,
                latency_ms=1,
                cost_usd=0.0,
                call_id=call_id,
            )

        def close(self):
            pass

    monkeypatch.setattr(somm_pkg, "SommLLM", _FakeEvalLLM)

    rc = main(
        [
            "eval",
            "run",
            "--workload",
            "eval_w",
            "--dataset",
            "golden",
            "--project",
            "cli-eval",
            "--threshold",
            "0.9",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "PASS eval cli-eval/eval_w/golden" in out
    with repo._open() as conn:
        n = conn.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0]
    assert n == 1


def test_corrected_command_hints_parse_against_real_parsers():
    parser = build_parser()
    for argv in (
        ["workload", "add", "orders", "--from-example", "structured-extraction"],
        ["workload", "add", "orders", "--from-example", "freeform"],
        ["workload", "list"],
        ["workload", "show", "orders"],
        ["workload", "set-constraints", "orders", "--max-p95-ttft-ms", "100"],
        ["bench", "latency", "hello", "--iterations", "1"],
        ["prompt", "list", "--workload", "orders"],
        ["prompt", "show", "--workload", "orders", "--version", "v1"],
        ["prompt", "register", "--workload", "orders", "--body", "hello"],
        ["prompt", "fork", "--workload", "orders", "--from", "v1", "--body-file", "p.txt"],
        ["prompt", "diff", "--workload", "orders", "v1", "v1.1"],
        ["prompt", "label", "--workload", "orders", "--label", "production", "--version", "v1"],
        [
            "prompt",
            "label",
            "--workload",
            "orders",
            "--label",
            "production",
            "--weights",
            "v1=90,v1.1=10",
        ],
        ["prompt", "promote", "--workload", "orders", "--version", "v1", "--to", "production"],
        ["prompt", "score", "--workload", "orders", "--label", "production"],
        ["eval", "promote-call", "abc", "--dataset", "golden"],
        ["eval", "run", "--workload", "orders", "--dataset", "golden"],
        ["optimize", "--workload", "orders"],
        ["campaign", "run", "--workload", "orders", "--dataset", "golden"],
        ["inbox", "list"],
        ["inbox", "apply", "1"],
        ["inbox", "dismiss", "1"],
        ["plugin", "list"],
        ["plugin", "info", "cache"],
        ["frontier", "--workload", "orders"],
        ["cache-advice"],
        ["compare", "hello", "--models", "ollama/g"],
        ["spend"],
        ["plans"],
        ["plans", "--learn", "--dry-run"],
        ["doctor", "--project", "my_project"],
        ["serve", "--project", "my_project"],
        ["drain-spool"],
    ):
        parser.parse_args(argv)

    from somm_service.cli import build_parser as build_serve_parser

    serve_parser = build_serve_parser()
    serve_parser.parse_args(["admin", "refresh-intel"])


# ---------------------------------------------------------------------------
# doctor (in-process; patches the env to avoid touching real repo)


def test_doctor_reports_schema_and_no_db(tmp_path, capsys, monkeypatch):
    # Use a fresh tmp dir with no .somm — doctor should report db missing
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "doctor-test")
    # Disable ollama check by pointing to a dead URL
    monkeypatch.setenv("SOMM_OLLAMA_URL", "http://127.0.0.1:1")
    main(["doctor"])
    out = capsys.readouterr().out
    assert "somm v" in out
    assert "project: doctor-test" in out
    # db shouldn't exist at this path yet
    assert "exists:" in out


def test_doctor_reports_worker_heartbeats(tmp_path, capsys, monkeypatch):
    cfg = _tmp_config(tmp_path)
    repo = Repository(cfg.db_path)
    with repo._open() as conn:
        conn.execute(
            "INSERT INTO worker_heartbeat "
            "(worker_name, last_run_at, last_success_at, consecutive_failures) "
            "VALUES ('shadow_eval', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"
        )

    monkeypatch.setattr("somm.cli.load_config", lambda project=None: cfg)
    monkeypatch.setattr(
        "somm.cli.OllamaProvider.health",
        lambda self: ProviderHealth(available=True, detail="ok"),
    )

    rc = main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "worker_heartbeat:" in out
    assert "shadow_eval" in out
    assert "last_run_at" in out
    assert "last_success_at" in out

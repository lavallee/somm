from __future__ import annotations

import uuid
from datetime import UTC, datetime

from somm.campaigns import MetricContract, run_eval_campaign
from somm.cli import main
from somm_core.models import Call, Outcome, SommResult
from somm_core.repository import Repository


def _seed_dataset(repo: Repository, project: str = "campaign"):
    wl = repo.register_workload(name="claims", project=project)
    source_call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=source_call_id,
            ts=datetime.now(UTC),
            project=project,
            workload_id=wl.id,
            prompt_id=None,
            provider="fake",
            model="source",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=0.0,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )
    repo.write_sample(source_call_id, "Answer yes.", "yes")
    dataset, item = repo.promote_call_to_dataset(
        source_call_id,
        "golden",
        project=project,
        created_by="test",
    )
    return wl, dataset, item


def _write_generated_call(
    repo: Repository,
    *,
    project: str,
    workload_id: str,
    text: str,
    tokens_in: int = 2,
    tokens_out: int = 3,
) -> SommResult:
    call_id = str(uuid.uuid4())
    repo.write_call(
        Call(
            id=call_id,
            ts=datetime.now(UTC),
            project=project,
            workload_id=workload_id,
            prompt_id=None,
            provider="fake",
            model="candidate",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=1,
            cost_usd=0.001,
            outcome=Outcome.OK,
            error_kind=None,
            prompt_hash="p",
            response_hash="r",
        )
    )
    return SommResult(
        text=text,
        provider="fake",
        model="candidate",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=1,
        cost_usd=0.001,
        call_id=call_id,
    )


def test_campaign_plateau_records_keep_revert_log(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, dataset, _item = _seed_dataset(repo)

    def generate(_item):
        return _write_generated_call(
            repo,
            project="campaign",
            workload_id=wl.id,
            text="yes",
        )

    result = run_eval_campaign(
        repo,
        project="campaign",
        workload="claims",
        dataset="golden",
        generate=generate,
        contract=MetricContract(metric="mean_score", threshold=0.8),
        max_rounds=4,
        token_budget=100,
        plateau_window=2,
        min_delta=0.01,
    )

    assert result.stop_reason == "plateau"
    assert result.passed is True
    assert result.campaign.status == "plateau"
    assert result.campaign.dataset_id == dataset.id
    round_events = [event for event in result.events if event.event_type == "round_completed"]
    assert [event.action for event in round_events] == ["keep", "revert", "revert"]
    assert [event.metric_score for event in round_events] == [1.0, 1.0, 1.0]
    assert result.total_tokens == 15
    assert "campaign_stopped" in result.jsonl()

    stored = repo.campaign_events(result.campaign.id)
    assert len(stored) == len(result.events) == 5
    assert stored[-1].payload["stop_reason"] == "plateau"


def test_campaign_stops_when_token_budget_is_spent(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    wl, _dataset, _item = _seed_dataset(repo, project="budget-campaign")

    def generate(_item):
        return _write_generated_call(
            repo,
            project="budget-campaign",
            workload_id=wl.id,
            text="yes",
            tokens_in=2,
            tokens_out=2,
        )

    result = run_eval_campaign(
        repo,
        project="budget-campaign",
        workload="claims",
        dataset="golden",
        generate=generate,
        max_rounds=5,
        token_budget=4,
        plateau_window=5,
    )

    assert result.stop_reason == "token_budget"
    assert result.campaign.status == "budget_exhausted"
    round_events = [event for event in result.events if event.event_type == "round_completed"]
    assert len(round_events) == 1
    assert round_events[0].total_tokens == 4


def test_campaign_cli_runs_and_writes_jsonl(tmp_path, capsys, monkeypatch):
    import somm as somm_pkg

    monkeypatch.chdir(tmp_path)
    repo = Repository(tmp_path / ".somm" / "calls.sqlite")
    wl, _dataset, _item = _seed_dataset(repo, project="campaign-cli")

    class _FakeCampaignLLM:
        def __init__(self, config):
            self.config = config
            self.repo = Repository(config.db_path)

        def generate(self, **kwargs):
            workload = self.repo.workload_by_name(kwargs["workload"], self.config.project)
            assert workload is not None
            assert workload.id == wl.id
            return _write_generated_call(
                self.repo,
                project=self.config.project,
                workload_id=workload.id,
                text="yes",
            )

        def close(self):
            pass

    monkeypatch.setattr(somm_pkg, "SommLLM", _FakeCampaignLLM)

    rc = main(
        [
            "campaign",
            "run",
            "--workload",
            "claims",
            "--dataset",
            "golden",
            "--project",
            "campaign-cli",
            "--max-rounds",
            "1",
            "--log",
            "campaign.jsonl",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "PASS campaign" in out
    log = tmp_path / "campaign.jsonl"
    assert log.exists()
    assert "campaign_started" in log.read_text()
    assert "round_completed" in log.read_text()

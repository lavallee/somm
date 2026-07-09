from __future__ import annotations

import pytest
from somm.client import SommLLM
from somm.prompts import (
    PromptNotFound,
    fork_prompt,
    get_label,
    get_prompt,
    label_history,
    list_labels,
    register_prompt,
    set_label,
)
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.repository import Repository


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="ok",
            model=request.model or "fake-m",
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _repo_with_workload(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    workload = repo.register_workload(name="claims", project="prompt-labels")
    return repo, workload


def _tmp_config(tmp_path) -> Config:
    cfg = Config()
    cfg.project = "prompt-labels"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def test_set_label_moves_pointer_and_records_history(tmp_path):
    repo, workload = _repo_with_workload(tmp_path)
    try:
        p1 = register_prompt(repo, workload.id, "body one")
        p2 = register_prompt(repo, workload.id, "body two")

        set_label(repo, workload.id, "production", p1.id, updated_by="alice")
        assert get_label(repo, workload.id, "production").id == p1.id

        set_label(repo, workload.id, "production", p2.id, updated_by="bob")
        assert get_label(repo, workload.id, "production").id == p2.id

        # Rollback is the same primitive: move the label back to an older id.
        set_label(repo, workload.id, "production", p1.id, updated_by="alice")
        assert get_label(repo, workload.id, "production").id == p1.id

        history = label_history(repo, workload.id, "production")
        assert [row["prompt_id"] for row in history] == [p1.id, p2.id, p1.id]
        assert [row["moved_by"] for row in history] == ["alice", "bob", "alice"]
    finally:
        repo.close()


def test_set_label_rejects_prompt_from_another_workload(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    try:
        first = repo.register_workload(name="first", project="prompt-labels")
        second = repo.register_workload(name="second", project="prompt-labels")
        prompt = register_prompt(repo, first.id, "first body")

        with pytest.raises(PromptNotFound, match="does not exist for workload"):
            set_label(repo, second.id, "production", prompt.id)
    finally:
        repo.close()


def test_get_prompt_resolves_label_and_label_wins_over_version(tmp_path):
    repo, workload = _repo_with_workload(tmp_path)
    try:
        p1 = register_prompt(repo, workload.id, "body one")
        p2 = register_prompt(repo, workload.id, "body two")
        set_label(repo, workload.id, "production", p1.id)

        assert get_prompt(repo, workload.id, label="production").id == p1.id
        assert (
            get_prompt(repo, workload.id, version=p2.version, label="production").id
            == p1.id
        )
        assert get_label(repo, workload.id, "missing") is None
        with pytest.raises(PromptNotFound, match="no prompt label"):
            get_prompt(repo, workload.id, label="missing")
    finally:
        repo.close()


def test_list_labels_returns_prompts_by_label(tmp_path):
    repo, workload = _repo_with_workload(tmp_path)
    try:
        p1 = register_prompt(repo, workload.id, "body one")
        p2 = register_prompt(repo, workload.id, "body two")
        set_label(repo, workload.id, "production", p1.id)
        set_label(repo, workload.id, "staging", p2.id)

        labels = list_labels(repo, workload.id)
        assert labels["production"].id == p1.id
        assert labels["staging"].id == p2.id
        assert labels["latest"].id == p2.id
    finally:
        repo.close()


def test_fork_prompt_sets_parent_and_is_retrievable(tmp_path):
    repo, workload = _repo_with_workload(tmp_path)
    try:
        source = register_prompt(repo, workload.id, "source body")
        set_label(repo, workload.id, "production", source.id)

        fork = fork_prompt(repo, workload.id, "production", "fork body", updated_by="alice")

        assert fork.parent_prompt_id == source.id
        assert get_prompt(repo, workload.id, version=fork.version).parent_prompt_id == source.id
        assert get_prompt(repo, workload.id, label="latest").id == fork.id
    finally:
        repo.close()


def test_register_prompt_moves_latest_but_not_production(tmp_path):
    repo, workload = _repo_with_workload(tmp_path)
    try:
        p1 = register_prompt(repo, workload.id, "body one")
        set_label(repo, workload.id, "production", p1.id)

        p2 = register_prompt(repo, workload.id, "body two")
        idempotent = register_prompt(repo, workload.id, "body two")

        assert idempotent.id == p2.id
        assert get_label(repo, workload.id, "latest").id == p2.id
        assert get_label(repo, workload.id, "production").id == p1.id
        assert [row["prompt_id"] for row in label_history(repo, workload.id, "latest")] == [
            p1.id,
            p2.id,
        ]
    finally:
        repo.close()


def test_sommllm_prompt_label_and_fork_wrappers(tmp_path):
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="claims", body="body one")
        llm.set_prompt_label("claims", "production", p1.version)

        p2 = llm.register_prompt(workload="claims", body="body two")
        fork = llm.fork_prompt("claims", "production", "fork body")

        assert llm.prompt("claims", label="production").id == p1.id
        assert llm.prompt("claims", version=p2.version).id == p2.id
        assert fork.parent_prompt_id == p1.id
    finally:
        llm.close()

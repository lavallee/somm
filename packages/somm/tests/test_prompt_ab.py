from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from somm import hooks
from somm.client import SommLLM
from somm.prompts import (
    get_prompt,
    register_prompt,
    resolve_label,
    set_label_weights,
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


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "prompt-ab"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def test_set_label_weights_stores_normalized_distribution(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    try:
        workload = repo.register_workload(name="claims", project="prompt-ab")
        p1 = register_prompt(repo, workload.id, "body one")
        p2 = register_prompt(repo, workload.id, "body two")

        set_label_weights(repo, workload.id, "production", {p1.version: 90, p2.id: 10})

        with repo._open() as conn:
            prompt_id, weights_json = conn.execute(
                "SELECT prompt_id, weights_json FROM prompt_labels "
                "WHERE workload_id = ? AND label = 'production'",
                (workload.id,),
            ).fetchone()

        assert prompt_id == p1.id
        assert f'"{p1.id}":0.9' in weights_json
        assert f'"{p2.id}":0.1' in weights_json
        assert get_prompt(repo, workload.id, label="production").id == p1.id
    finally:
        repo.close()


def test_resolve_label_weighted_is_deterministic_and_roughly_weighted(tmp_path):
    repo = Repository(tmp_path / "calls.sqlite")
    try:
        workload = repo.register_workload(name="claims", project="prompt-ab")
        p1 = register_prompt(repo, workload.id, "body one")
        p2 = register_prompt(repo, workload.id, "body two")
        set_label_weights(repo, workload.id, "production", {p1.version: 90, p2.version: 10})

        assert (
            resolve_label(repo, workload.id, "production", bucket_key="user-42").id
            == resolve_label(repo, workload.id, "production", bucket_key="user-42").id
        )

        counts = Counter(
            resolve_label(repo, workload.id, "production", bucket_key=str(i)).id
            for i in range(1000)
        )
        assert 850 <= counts[p1.id] <= 950
        assert 50 <= counts[p2.id] <= 150
    finally:
        repo.close()


def test_sommllm_prompt_weighted_label_stable_by_bucket_key(tmp_path):
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="claims", body="body one")
        p2 = llm.register_prompt(workload="claims", body="body two")
        workload = llm._require_workload("claims")
        set_label_weights(
            llm.repo,
            workload.id,
            "production",
            {p1.version: 90, p2.version: 10},
        )

        first = llm.prompt("claims", label="production", bucket_key="user-42")
        second = llm.prompt("claims", label="production", bucket_key="user-42")
        assert first.id == second.id

        seen = {
            llm.prompt("claims", label="production", bucket_key=f"user-{i}").id
            for i in range(200)
        }
        assert {p1.id, p2.id}.issubset(seen)
    finally:
        llm.close()


def test_sommllm_prompt_uses_correlation_id_when_bucket_key_omitted(tmp_path):
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[FakeProvider()])
    hooks.set_correlation_provider(lambda: "corr-42")
    try:
        p1 = llm.register_prompt(workload="claims", body="body one")
        p2 = llm.register_prompt(workload="claims", body="body two")
        workload = llm._require_workload("claims")
        set_label_weights(
            llm.repo,
            workload.id,
            "production",
            {p1.version: 90, p2.version: 10},
        )

        assert llm.prompt("claims", label="production").id == llm.prompt(
            "claims", label="production"
        ).id
    finally:
        hooks.set_correlation_provider(None)
        llm.close()


def test_end_to_end_weighted_prompts_bind_selected_prompt_ids(tmp_path):
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[FakeProvider()])
    try:
        p1 = llm.register_prompt(workload="claims", body="body one")
        p2 = llm.register_prompt(workload="claims", body="body two")
        workload = llm._require_workload("claims")
        set_label_weights(
            llm.repo,
            workload.id,
            "production",
            {p1.version: 90, p2.version: 10},
        )

        for i in range(300):
            prompt = llm.prompt("claims", label="production", bucket_key=str(i))
            llm.generate(prompt, workload="claims")
    finally:
        llm.close()

    with sqlite3.connect(cfg.db_path) as conn:
        rows = conn.execute(
            "SELECT prompt_id, COUNT(*) FROM calls GROUP BY prompt_id"
        ).fetchall()
    counts = dict(rows)
    assert set(counts) == {p1.id, p2.id}
    assert 240 <= counts[p1.id] <= 285
    assert 15 <= counts[p2.id] <= 60

"""Sommelier — candidate ranking + decision memory + cross-project mirror."""

from __future__ import annotations

from pathlib import Path

from somm.sommelier import AdviseConstraints, advise, build_decision
from somm_core.config import Config
from somm_core.parse import stable_hash
from somm_core.pricing import write_intel
from somm_core.repository import Repository


def _tmp_repo(tmp_path: Path, name: str = "calls.sqlite") -> Repository:
    cfg = Config()
    cfg.project = "sm"
    cfg.db_dir = tmp_path / name
    return Repository(cfg.db_dir / "calls.sqlite")


def _seed_intel(repo: Repository):
    """A small but realistic model_intel population."""
    # Anthropic — paid, no caps dict (name-hint vision coverage).
    write_intel(
        repo, "anthropic", "claude-haiku-4-5-20251001",
        price_in_per_1m=1.0, price_out_per_1m=5.0,
        context_window=200_000, capabilities=None, source="static",
    )
    write_intel(
        repo, "anthropic", "claude-opus-4-7",
        price_in_per_1m=15.0, price_out_per_1m=75.0,
        context_window=200_000, capabilities=None, source="static",
    )
    # OpenRouter free vision models.
    write_intel(
        repo, "openrouter", "google/gemma-3-27b-it:free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=128_000,
        capabilities={"modality": "text+image->text"},
        source="openrouter",
    )
    write_intel(
        repo, "openrouter", "meta-llama/llama-3.3-70b-instruct:free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=128_000,
        capabilities={"modality": "text->text"},   # NOT vision
        source="openrouter",
    )
    # Ollama — local.
    write_intel(
        repo, "ollama", "llava:13b",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=None,
        capabilities={"family": "llava"},
        source="ollama-local",
    )
    # An openrouter paid model.
    write_intel(
        repo, "openrouter", "anthropic/claude-sonnet-4-6",
        price_in_per_1m=3.0, price_out_per_1m=15.0,
        context_window=200_000,
        capabilities={"modality": "text+image->text"},
        source="openrouter",
    )


# ---------------------------------------------------------------------------
# Candidate ranking


def test_advise_hard_filters_non_vision_when_required(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(capabilities=["vision"]))
    names = {(c.provider, c.model) for c in cands}
    assert ("openrouter", "meta-llama/llama-3.3-70b-instruct:free") not in names
    # Vision-capable ones remain.
    assert ("openrouter", "google/gemma-3-27b-it:free") in names


def test_advise_free_only_excludes_paid(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(free_only=True))
    for c in cands:
        assert (c.price_in_per_1m or 0) == 0
        assert (c.price_out_per_1m or 0) == 0


def test_advise_provider_whitelist(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(providers=["openrouter"]))
    assert cands
    for c in cands:
        assert c.provider == "openrouter"


def test_advise_price_ceiling(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(max_price_in_per_1m=2.0))
    # claude-haiku ($1 in) passes, claude-opus ($15 in) must not.
    names = {(c.provider, c.model) for c in cands}
    assert ("anthropic", "claude-haiku-4-5-20251001") in names
    assert ("anthropic", "claude-opus-4-7") not in names


def test_advise_min_context_window(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(min_context_window=150_000))
    # Gemma (128k) must be excluded; claude-* (200k) included.
    for c in cands:
        assert (c.context_window or 0) >= 150_000 or c.context_window is None


def test_advise_reasons_surface_price_and_capability(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(capabilities=["vision"]))
    # Find the free gemma candidate
    gem = next(c for c in cands if "gemma-3-27b" in c.model)
    assert "free" in gem.reasons
    assert any("ctx" in r for r in gem.reasons)
    assert any("vision" in r for r in gem.reasons)


def test_advise_cheaper_wins_all_else_equal(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel(repo)
    cands = advise(repo, AdviseConstraints(capabilities=["vision"], limit=10))
    # Across vision-capable: free ones should outrank paid sonnet.
    assert cands[0].provider in ("openrouter", "ollama")
    assert (cands[0].price_in_per_1m or 0) == 0


# ---------------------------------------------------------------------------
# Decisions


def test_build_decision_hashes_question(tmp_path):
    d = build_decision(
        question="  Good free vision models on openrouter  ",
        candidates=[{"provider": "openrouter", "model": "x"}],
        rationale="because it's the only one",
        project="p",
    )
    expected = stable_hash("good free vision models on openrouter")
    assert d.question_hash == expected
    assert d.question == "  Good free vision models on openrouter  "


def test_record_and_get_decision_roundtrip(tmp_path):
    repo = _tmp_repo(tmp_path)
    d = build_decision(
        question="vision models",
        candidates=[{"provider": "openrouter", "model": "gemma-3", "score": 3.2}],
        rationale="largest free vision ctx",
        project="malo",
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
        workload="critique_visual",
        agent="claude-code",
    )
    repo.record_decision(d)
    got = repo.get_decision(d.id)
    assert got is not None
    assert got.chosen_model.startswith("google/gemma-3-27b")
    assert got.agent == "claude-code"
    assert got.candidates[0]["provider"] == "openrouter"


def test_search_decisions_by_question_hash_and_project(tmp_path):
    repo = _tmp_repo(tmp_path)
    repo.record_decision(
        build_decision(
            question="Good free vision models?",
            candidates=[],
            rationale="picked gemma",
            project="malo",
            chosen_provider="openrouter",
            chosen_model="google/gemma-3-27b-it:free",
        )
    )
    repo.record_decision(
        build_decision(
            question="Cheap long context model?",
            candidates=[],
            rationale="picked haiku",
            project="malo",
            chosen_provider="anthropic",
            chosen_model="claude-haiku-4-5-20251001",
        )
    )
    # Case/whitespace-insensitive via hash match
    found = repo.search_decisions(question="good  Free vision models?")
    assert len(found) == 1
    assert found[0].chosen_model.startswith("google/gemma-3-27b")

    # LIKE fallback
    partial = repo.search_decisions(question="long context")
    assert len(partial) == 1
    assert partial[0].chosen_model.startswith("claude-haiku")

    # Filter by chosen_provider
    anth = repo.search_decisions(chosen_provider="anthropic")
    assert len(anth) == 1


def test_decisions_survive_mirror_to_second_repo(tmp_path):
    """Simulate the global-mirror pattern: recording to both primary + global
    succeeds and both can be searched independently."""
    primary = _tmp_repo(tmp_path / "primary")
    global_repo = _tmp_repo(tmp_path / "global")
    d = build_decision(
        question="vision pick",
        candidates=[],
        rationale="x",
        project="malo",
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
    )
    primary.record_decision(d)
    global_repo.record_decision(d)

    assert primary.get_decision(d.id) is not None
    assert global_repo.get_decision(d.id) is not None

    # Can search in global for any project.
    results = global_repo.search_decisions(project="malo")
    assert results and results[0].id == d.id


def test_mark_decision_outcome(tmp_path):
    repo = _tmp_repo(tmp_path)
    d = build_decision(
        question="q",
        candidates=[],
        rationale="r",
        project="p",
    )
    repo.record_decision(d)
    repo.mark_decision_outcome(d.id, "struggled with colour contrasts")
    got = repo.get_decision(d.id)
    assert got.outcome_note == "struggled with colour contrasts"

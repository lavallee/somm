"""Sommelier — candidate ranking + decision memory + cross-project mirror."""

from __future__ import annotations

from pathlib import Path

from datetime import UTC, datetime, timedelta

from somm.sommelier import (
    AdviseConstraints,
    advise,
    build_decision,
    consult,
)
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


# ---------------------------------------------------------------------------
# 0.2.2 — meta-router exclusion, output modality, exclude_models, unknown-cap
# penalty, prior-decision signals, deterministic tiebreak, empty-note
# ---------------------------------------------------------------------------


def _seed_intel_022(repo: Repository):
    """Fixture reflecting the real-world mess from malo's report:
    meta-routers, a Lyria-shape audio-out-vision-in model, an unknown-cap
    MiniMax entry, and real vision models.
    """
    # Meta-routers (should always be excluded unless include_meta_routers).
    write_intel(
        repo, "openrouter", "openrouter/auto",
        price_in_per_1m=None, price_out_per_1m=None,
        context_window=2_000_000,
        capabilities={"modality": "text+image->text"},
        source="openrouter",
    )
    write_intel(
        repo, "openrouter", "openrouter/free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=200_000,
        capabilities={"modality": "text+image->text"},
        source="openrouter",
    )
    # Lyria-shape: image in, AUDIO out. Should be filtered when
    # required_output_modalities=["text"].
    write_intel(
        repo, "openrouter", "google/lyria-3-pro-preview",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=1_048_576,
        capabilities={
            "modality": "text+image->audio",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["audio"],
            },
        },
        source="openrouter",
    )
    # Real free vision model with text output.
    write_intel(
        repo, "openrouter", "google/gemma-3-27b-it:free",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=262_144,
        capabilities={
            "modality": "text+image->text",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
        },
        source="openrouter",
    )
    # Unknown-capability model (minimax doesn't publish caps).
    write_intel(
        repo, "minimax", "MiniMax-M2",
        price_in_per_1m=0.0, price_out_per_1m=0.0,
        context_window=200_000,
        capabilities=None,
        source="static",
    )


def test_advise_excludes_meta_routers_by_default(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    cands = advise(repo, AdviseConstraints(capabilities=["vision"], free_only=True))
    names = {c.model for c in cands}
    assert "openrouter/auto" not in names
    assert "openrouter/free" not in names


def test_advise_include_meta_routers_opt_in(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    cands = advise(
        repo,
        AdviseConstraints(
            capabilities=["vision"], free_only=True, include_meta_routers=True,
        ),
    )
    names = {c.model for c in cands}
    assert "openrouter/free" in names


def test_advise_output_modality_filters_lyria(tmp_path):
    """Audio-out model with image-in must drop for a text-output workload."""
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    cands = advise(
        repo,
        AdviseConstraints(
            capabilities=["vision"],
            required_output_modalities=["text"],
            free_only=True,
            limit=20,
        ),
    )
    names = {c.model for c in cands}
    assert "google/lyria-3-pro-preview" not in names
    assert "google/gemma-3-27b-it:free" in names


def test_advise_exclude_models_glob(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    cands = advise(
        repo,
        AdviseConstraints(
            capabilities=["vision"],
            exclude_models=["openrouter/google/lyria-*"],
            free_only=True,
        ),
    )
    names = {c.model for c in cands}
    assert "google/lyria-3-pro-preview" not in names


def test_advise_unknown_capability_penalty_applied(tmp_path):
    """MiniMax-M2 (caps=None) must score lower than gemma (vision✓) once a
    vision capability is requested."""
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    cands = advise(
        repo,
        AdviseConstraints(
            capabilities=["vision"], free_only=True,
            unknown_capability_penalty=0.9, limit=20,
        ),
    )
    by_model = {c.model: c.score for c in cands}
    # Both remain in the list but gemma outscores minimax.
    assert "google/gemma-3-27b-it:free" in by_model
    assert "MiniMax-M2" in by_model
    assert by_model["google/gemma-3-27b-it:free"] > by_model["MiniMax-M2"]


def test_advise_unknown_capability_penalty_disabled(tmp_path):
    """At penalty=1.0 we preserve pre-0.2.2 behaviour (tie)."""
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    cands = advise(
        repo,
        AdviseConstraints(
            capabilities=["vision"], free_only=True,
            unknown_capability_penalty=1.0, limit=20,
        ),
    )
    by_model = {c.model: c.score for c in cands}
    assert by_model["google/gemma-3-27b-it:free"] == by_model["MiniMax-M2"]


def test_advise_ranking_is_deterministic(tmp_path):
    """Two runs against the same intel must produce the same order."""
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    a = advise(repo, AdviseConstraints(free_only=True, limit=20))
    b = advise(repo, AdviseConstraints(free_only=True, limit=20))
    assert [(c.provider, c.model) for c in a] == [(c.provider, c.model) for c in b]


def test_consult_empty_result_reports_filter_reasons(tmp_path):
    """Seed a world where every vision-capable model has a known wrong
    output modality, and verify the note surfaces which filter ate them.

    We scope to providers=['openrouter'] to exclude name-hint-only
    capability providers (Anthropic/etc) whose unknown output modality
    would leak through the filter by design.
    """
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    # Gemma outputs text; Lyria outputs audio; force a wrong-modality query.
    result = consult(
        repo,
        question="vision model that writes music",
        constraints=AdviseConstraints(
            capabilities=["vision"],
            required_output_modalities=["audio"],
            providers=["openrouter"],
            free_only=True,
            exclude_models=["openrouter/google/lyria-*"],
        ),
        project="test",
    )
    assert not result.candidates
    assert result.note is not None
    assert "exclude_models match" in result.note or "output modality" in result.note


def test_consult_prior_decision_positive_annotates_and_boosts(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    # Record a prior decision picking gemma, fresh (no decay).
    d = build_decision(
        question="free vision model for captioning",
        candidates=[],
        rationale="largest free vision ctx",
        project="malo",
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
    )
    repo.record_decision(d)
    baseline = next(
        c.score for c in advise(
            repo, AdviseConstraints(capabilities=["vision"], free_only=True, limit=20)
        )
        if c.model == "google/gemma-3-27b-it:free"
    )
    result = consult(
        repo,
        question="free vision model for captioning",
        constraints=AdviseConstraints(capabilities=["vision"], free_only=True, limit=20),
        project="malo",
    )
    gem = next(c for c in result.candidates if c.model == "google/gemma-3-27b-it:free")
    assert gem.score > baseline   # positive nudge
    assert any("prior" in r and "chose" in r for r in gem.reasons)


def test_consult_prior_decision_negative_penalises(tmp_path):
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)
    d = build_decision(
        question="free vision model",
        candidates=[],
        rationale="picked gemma initially",
        project="malo",
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
    )
    repo.record_decision(d)
    repo.mark_decision_outcome(d.id, "unreliable from Google AI Studio upstream")
    baseline = next(
        c.score for c in advise(
            repo, AdviseConstraints(capabilities=["vision"], free_only=True, limit=20)
        )
        if c.model == "google/gemma-3-27b-it:free"
    )
    result = consult(
        repo,
        question="free vision model",
        constraints=AdviseConstraints(capabilities=["vision"], free_only=True, limit=20),
        project="malo",
    )
    gem = next(c for c in result.candidates if c.model == "google/gemma-3-27b-it:free")
    assert gem.score < baseline   # soft penalty
    assert any("prior" in r and "flagged" in r for r in gem.reasons)


def test_consult_prior_decision_decays_with_age(tmp_path):
    """A year-old prior should carry far less weight than a fresh one."""
    repo = _tmp_repo(tmp_path)
    _seed_intel_022(repo)

    # Fresh prior (no outcome note → positive).
    fresh = build_decision(
        question="free vision model",
        candidates=[], rationale="r", project="malo",
        chosen_provider="openrouter",
        chosen_model="google/gemma-3-27b-it:free",
    )
    repo.record_decision(fresh)
    fresh_gem = next(
        c for c in consult(
            repo, question="free vision model",
            constraints=AdviseConstraints(capabilities=["vision"], free_only=True, limit=20),
            project="malo",
        ).candidates
        if c.model == "google/gemma-3-27b-it:free"
    )

    # Backdate the decision directly in SQL to ~1 year ago and re-consult.
    old_ts = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    with repo._open() as conn:
        conn.execute("UPDATE decisions SET ts = ? WHERE id = ?", (old_ts, fresh.id))
    aged_gem = next(
        c for c in consult(
            repo, question="free vision model",
            constraints=AdviseConstraints(capabilities=["vision"], free_only=True, limit=20),
            project="malo",
        ).candidates
        if c.model == "google/gemma-3-27b-it:free"
    )
    # Decayed nudge is smaller than the fresh one.
    assert aged_gem.score < fresh_gem.score
    # But still a positive nudge compared to the no-prior baseline.
    baseline = next(
        c.score for c in advise(
            repo, AdviseConstraints(capabilities=["vision"], free_only=True, limit=20)
        )
        if c.model == "google/gemma-3-27b-it:free"
    )
    assert aged_gem.score > baseline


def test_advise_output_modality_via_openrouter_architecture(tmp_path):
    """Exercise the architecture.output_modalities path (not just scalar modality)."""
    repo = _tmp_repo(tmp_path)
    write_intel(
        repo, "openrouter", "some-vision-text-model",
        price_in_per_1m=0.0, price_out_per_1m=0.0, context_window=100_000,
        capabilities={
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            # Intentionally no scalar `modality` — force fallback to arch.
        },
        source="openrouter",
    )
    write_intel(
        repo, "openrouter", "some-audio-gen-model",
        price_in_per_1m=0.0, price_out_per_1m=0.0, context_window=100_000,
        capabilities={
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["audio"],
            },
        },
        source="openrouter",
    )
    cands = advise(
        repo,
        AdviseConstraints(
            required_output_modalities=["text"], free_only=True, limit=20,
        ),
    )
    names = {c.model for c in cands}
    assert "some-vision-text-model" in names
    assert "some-audio-gen-model" not in names

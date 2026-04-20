"""Tests for cost-tracking safety features.

Feature 1: seed_known_pricing populates model_intel on first use.
Feature 2: cost_for_call warns on missing pricing for paid providers.
Feature 3: generate() warns when daily budget cap is exceeded.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from somm_core.pricing import (
    _KNOWN_PRICING,
    _warned_missing_pricing,
    cost_for_call,
    list_intel,
    seed_known_pricing,
    write_intel,
)
from somm_core.repository import Repository


def _fresh_repo() -> Repository:
    """Create a repo with a fresh temp DB."""
    tmpdir = TemporaryDirectory()
    db_path = Path(tmpdir.name) / "test.db"
    repo = Repository(db_path)
    repo._tmpdir = tmpdir  # prevent GC
    return repo


# ---- Feature 1: seed_known_pricing -----------------------------------------


def test_seed_populates_empty_table():
    repo = _fresh_repo()
    # Table should be empty initially
    assert list_intel(repo) == []

    seed_known_pricing(repo)

    rows = list_intel(repo)
    assert len(rows) == len(_KNOWN_PRICING)

    # Check Sonnet entries — multiple snapshots are seeded (current + prior).
    anthropic_sonnet = [
        r for r in rows if r["provider"] == "anthropic" and "sonnet" in r["model"]
    ]
    assert len(anthropic_sonnet) >= 1
    for row in anthropic_sonnet:
        assert row["price_in_per_1m"] == 3.00
        assert row["price_out_per_1m"] == 15.00
        assert row["source"] == "somm_seed"


def test_seed_does_not_overwrite_existing():
    repo = _fresh_repo()
    # Manually set a price
    write_intel(repo, "anthropic", "custom-model", 99.0, 99.0, None, None, "manual")

    seed_known_pricing(repo)

    rows = list_intel(repo)
    # Should still have just the one manually-set row, not seeded
    assert len(rows) == 1
    assert rows[0]["model"] == "custom-model"
    assert rows[0]["price_in_per_1m"] == 99.0


def test_seed_includes_free_providers():
    repo = _fresh_repo()
    seed_known_pricing(repo)

    rows = list_intel(repo)
    ollama_rows = [r for r in rows if r["provider"] == "ollama"]
    assert len(ollama_rows) >= 1
    assert ollama_rows[0]["price_in_per_1m"] == 0.0
    assert ollama_rows[0]["price_out_per_1m"] == 0.0


# ---- Feature 2: warn on missing pricing for paid providers ------------------


def test_warn_missing_pricing_for_paid_provider():
    repo = _fresh_repo()
    _warned_missing_pricing.clear()

    buf = io.StringIO()
    with redirect_stderr(buf):
        cost = cost_for_call(repo, "anthropic", "claude-unknown-99", 1000, 500)

    assert cost == 0.0
    warning = buf.getvalue()
    assert "WARNING" in warning
    assert "anthropic/claude-unknown-99" in warning


def test_warn_only_once_per_pair():
    repo = _fresh_repo()
    _warned_missing_pricing.clear()

    buf = io.StringIO()
    with redirect_stderr(buf):
        cost_for_call(repo, "anthropic", "claude-once-test", 1000, 500)
        cost_for_call(repo, "anthropic", "claude-once-test", 2000, 1000)

    # Should only have one warning line
    lines = [l for l in buf.getvalue().strip().splitlines() if "WARNING" in l]
    assert len(lines) == 1


def test_no_warn_for_free_providers():
    repo = _fresh_repo()
    _warned_missing_pricing.clear()

    buf = io.StringIO()
    with redirect_stderr(buf):
        cost = cost_for_call(repo, "ollama", "llama3", 1000, 500)

    assert cost == 0.0
    assert buf.getvalue() == ""


def test_no_warn_when_pricing_exists():
    repo = _fresh_repo()
    _warned_missing_pricing.clear()
    seed_known_pricing(repo)

    buf = io.StringIO()
    with redirect_stderr(buf):
        cost = cost_for_call(
            repo, "anthropic", "claude-sonnet-4-20250514", 1_000_000, 0
        )

    assert cost == 3.00
    assert buf.getvalue() == ""


# ---- Feature 1+2 integration: seeded pricing produces correct costs ---------


def test_seeded_pricing_computes_cost():
    repo = _fresh_repo()
    seed_known_pricing(repo)

    # 1M input tokens of claude-opus-4 = $15.00
    cost = cost_for_call(
        repo, "anthropic", "claude-opus-4-20250514", 1_000_000, 0
    )
    assert cost == 15.0

    # 1M output tokens of gpt-4o-mini = $0.60
    cost = cost_for_call(repo, "openai", "gpt-4o-mini", 0, 1_000_000)
    assert cost == 0.60

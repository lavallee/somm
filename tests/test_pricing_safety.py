"""Tests for cost-tracking safety features.

Feature 1: the bundled snapshot is the one price table.
Feature 2: cost_for_call warns on missing pricing for paid providers.
Feature 3: generate() warns when daily budget cap is exceeded.
Feature 4: local unpriced calls are not represented as zero economic cost.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from somm.client import _cost_provenance
from somm_core.pricing import (
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


# ---- Feature 1: one price table ----------------------------------------------


def test_seed_is_the_bundle():
    """``seed_known_pricing`` no longer carries its own table: it syncs the bundle."""
    repo = _fresh_repo()
    assert list_intel(repo) == []

    seed_known_pricing(repo)

    rows = {(r["provider"], r["model"]): r for r in list_intel(repo)}
    assert len(rows) > 200
    assert {r["source"] for r in rows.values()} == {"litellm_bundle"}
    assert rows[("anthropic", "claude-sonnet-5-5")]["price_in_per_1m"] == 2.00


def test_seed_does_not_overwrite_manual_prices():
    repo = _fresh_repo()
    write_intel(repo, "anthropic", "claude-sonnet-5-5", 99.0, 99.0, None, None, "manual")

    seed_known_pricing(repo)

    row = next(
        r for r in list_intel(repo, provider="anthropic") if r["model"] == "claude-sonnet-5-5"
    )
    assert (row["price_in_per_1m"], row["source"]) == (99.0, "manual")


def test_bundle_replaces_rows_from_the_removed_tables():
    """Old databases hold rows from the hand-written seed and static tables,
    which disagreed with the bundle (claude-opus-4-7 at 15/75 against 5/25).
    The bundle owns those rows now, so the stale prices go."""
    repo = _fresh_repo()
    write_intel(repo, "anthropic", "claude-opus-4-7", 15.0, 75.0, None, None, "static")
    write_intel(repo, "deepseek", "deepseek-flash", 0.15, 0.60, None, None, "somm_seed")

    seed_known_pricing(repo)

    prices = {
        r["model"]: (r["price_in_per_1m"], r["price_out_per_1m"], r["source"])
        for r in list_intel(repo)
    }
    assert prices["claude-opus-4-7"] == (5.0, 25.0, "litellm_bundle")
    assert prices["deepseek-flash"] == (0.3, 1.2, "litellm_bundle")


def test_local_price_provenance_is_included_and_unpriced():
    repo = _fresh_repo()
    assert _cost_provenance(
        repo,
        "ollama",
        "qwen2.5:7b",
        reported=False,
        source=None,
    ) == ("unknown", "included", "unknown", "local-included-unpriced", None)


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
    lines = [ln for ln in buf.getvalue().strip().splitlines() if "WARNING" in ln]
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
            repo, "anthropic", "claude-sonnet-4-6", 1_000_000, 0
        )

    assert cost == 3.00
    assert buf.getvalue() == ""


# ---- Feature 1+2 integration: seeded pricing produces correct costs ---------


def test_seeded_pricing_computes_cost():
    repo = _fresh_repo()
    seed_known_pricing(repo)

    # 1M input tokens of claude-opus-4-7 = $5.00 (the bundle's list price)
    cost = cost_for_call(repo, "anthropic", "claude-opus-4-7", 1_000_000, 0)
    assert cost == 5.0

    # 1M output tokens of gpt-4o-mini = $0.60
    cost = cost_for_call(repo, "openai", "gpt-4o-mini", 0, 1_000_000)
    assert cost == 0.60

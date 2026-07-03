## Summary

<!-- What does this PR do, and why? -->

## Checklist

- [ ] Tests pass: `uv run pytest packages/ tests/`
- [ ] Blocklist guard passes: `uv run pytest tests/test_blocklist.py`
- [ ] `uv run ruff check packages/ tests/` is clean (or pre-existing
      violations are unrelated to this change)
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` (or this change
      doesn't affect behavior — docs/CI/tooling only)
- [ ] If this touches versioned files, I've read `RELEASING.md` and
      am not bumping versions independently

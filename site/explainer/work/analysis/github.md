## Purpose

This area defines how changes are proposed, validated, reported, and released. It standardizes contributor input while automating compatibility, quality, security, performance, and PyPI publishing checks.

## How it works

Pull requests run CI on Python 3.12 and 3.13 using the frozen `uv` workspace. Each matrix job checks release-version consistency, linting, tests, and performance budgets (`.github/workflows/ci.yml:9`). A separate security job audits locked third-party dependencies with `pip-audit` and GitHub Actions with `zizmor` (`.github/workflows/ci.yml:35`).

Publishing begins when a GitHub release is published or the workflow is manually dispatched (`.github/workflows/publish.yml:23`). A matrix builds and publishes each of the six workspace packages independently, using a package-specific GitHub environment and PyPI trusted publishing via OIDC (`.github/workflows/publish.yml:28`). Bug and feature templates collect reproducible diagnostics and package scope; the PR template requires contributors to confirm tests, the blocklist guard, lint status, changelog handling, and coordinated versioning (`.github/PULL_REQUEST_TEMPLATE.md:5`).

## Key surfaces

- `uv run pytest packages/ tests/` — contributor-facing test command (`.github/PULL_REQUEST_TEMPLATE.md:7`).
- `uv run pytest tests/test_blocklist.py` — focused blocklist guard (`.github/PULL_REQUEST_TEMPLATE.md:8`).
- `uv run python scripts/check_release_gate.py` — version-consistency gate used by CI and publishing (`.github/workflows/ci.yml:23`).
- `uv run python scripts/check_perf_budget.py` — automated performance-budget check (`.github/workflows/ci.yml:32`).
- `release: published` / `workflow_dispatch` — automatic and manual publishing entry points (`.github/workflows/publish.yml:23`).
- `somm doctor` — diagnostic output requested with bug reports (`.github/ISSUE_TEMPLATE/bug_report.md:22`).
- `somm calls --status error` — CLI route for surfacing failed-call telemetry (`.github/ISSUE_TEMPLATE/bug_report.md:28`).

## Design decisions

- Third-party actions are pinned to commit SHAs, reducing exposure to mutable action tags (`.github/workflows/ci.yml:15`).
- Security checks are isolated from the Python-version test matrix and operate on exported, locked production dependencies without workspace packages or development dependencies (`.github/workflows/ci.yml:35`).
- Each package has its own PyPI environment because pending trusted publishers must be unique per repository, workflow, and environment (`.github/workflows/publish.yml:4`).
- Publishing uses separate artifact directories because workspace-member builds otherwise write into the root `dist/` directory (`.github/workflows/publish.yml:54`).
- `skip-existing` makes a partially completed multi-package release safely rerunnable (`.github/workflows/publish.yml:63`).
- Bug reports request local telemetry details but explicitly tell reporters to redact API keys, reflecting the project’s privacy-sensitive diagnostic model (`.github/ISSUE_TEMPLATE/bug_report.md:28`, `.github/ISSUE_TEMPLATE/bug_report.md:46`).

## One-liner

This area is somm’s contribution and release control plane, turning structured reports and pull requests into tested, audited, independently published workspace packages.
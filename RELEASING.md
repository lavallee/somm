# Releasing somm

This checklist is the canonical release path. If anything here is wrong
or missing, update this file *first*, then run the release.

## Versioning

somm uses a single unified version across all workspace packages. A
release touches every `pyproject.toml` + `packages/somm-core/src/somm_core/version.py`
simultaneously.

Version bump rules:

- **Patch (0.1.0 → 0.1.1)** — bug fixes, doc updates, wire-format
  alignment. No new schema, no API additions.
- **Minor (0.1.1 → 0.2.0)** — new feature surface (MCP tools, library
  APIs, schema migrations) that stays backward-compatible.
- **Major (0.x → 1.0)** — a deliberate stability declaration, not a routine
  feature release. It requires a checked-in readiness decision before CI or
  the PyPI publish workflow will build artifacts.

CI and publish both run:

```bash
uv run python scripts/check_release_gate.py
```

That gate fails any `1.x` version unless
`docs/release/ONE_DOT_ZERO_GO_DECISION.md` exists and contains
`somm-1.0-go: true`. Add that file only after the project has explicitly
decided that the public API, MCP contracts, migration semantics, release
cadence, and operational docs are ready to be called stable.

Schema migrations bump `SCHEMA_VERSION` in lockstep with a new
`packages/somm-core/src/somm_core/migrations/NNNN_<name>.sql` file. One
migration per release is the norm; multiple is fine if they're
independent.

## The checklist

1. **All tests pass locally.**
   ```bash
   uv run pytest -q
   ```
   One skip is expected (`opentelemetry` not installed). Any other
   skip or failure blocks release. This line said "two skips (no local
   ollama)" through 0.15.0 and had been wrong for some time — a stale
   expectation in a checklist trains people to wave past the real thing.

2. **Bump every version string in lockstep.**
   ```bash
   # Update these to the new version:
   pyproject.toml
   packages/somm/pyproject.toml
   packages/somm-core/pyproject.toml
   packages/somm-service/pyproject.toml
   packages/somm-mcp/pyproject.toml
   packages/somm-skill/pyproject.toml
   packages/somm-core/src/somm_core/version.py   # VERSION + SCHEMA_VERSION
   ```
   One-liner (also rewrites the exact inter-package dependency pins —
   `somm==X`, `somm-core==X` — which MUST move in lockstep or PyPI
   installs can mix incompatible package versions):
   ```bash
   OLD=0.1.1; NEW=0.2.0
   for f in pyproject.toml packages/*/pyproject.toml roster.toml; do
     sed -i "s/version = \"$OLD\"/version = \"$NEW\"/; s/==$OLD\"/==$NEW\"/" "$f"
   done
   sed -i "s/VERSION = \"$OLD\"/VERSION = \"$NEW\"/" packages/somm-core/src/somm_core/version.py
   ```

3. **Update `CHANGELOG.md`.** Add a new dated heading at the top:
   ```markdown
   ## [X.Y.Z] — YYYY-MM-DD

   ### Added
   - …

   ### Fixed
   - …
   ```
   Be specific. Future-you and porters both read this.

4. **Update `docs/index.html`.** The GitHub Pages landing page is a
   marketing surface, not auto-generated — it lags if we forget.
   Minimum edits on every release:
   - `<span class="mark-sub">/ vX.Y.Z</span>` in the header
   - Any new headline features added to the feature grid or hero code
   - The `<section class="section section-alt">` "Status" block near
     the bottom (version + test count)

   If the release adds a user-visible concept (multimodal, sommelier,
   streaming), add a short sub-section to the landing page linking out
   to the corresponding `docs/*.md`.

5. **Skill parity check.** `somm-skill` is how coding agents (and any
   skill-distribution system that brokers it, e.g. ivy's lyra-forge)
   learn to use somm — it ships stale unless the release keeps it
   honest. If this release adds or changes a user-facing surface
   (API, CLI command, env var, MCP tool, billing/plan concept), reflect
   it in `packages/somm-skill/src/somm_skill/SKILL.md` and, for
   model-choice-related changes, `SOMMELIER.md`. An agent following an
   outdated skill writes outdated integrations at fleet scale.

6. **Refresh the bundled data snapshots.** Two files ship stale unless
   a human refreshes them:
   - **Pricing bundle** — regenerate from LiteLLM's price file:
     ```bash
     uv run python scripts/update_pricing_bundle.py
     ```
   - **Plan catalog** (`packages/somm-core/src/somm_core/data/plan_catalog.toml`) —
     plan limits are vendor marketing copy, not an API. Re-verify any
     entry whose `last_verified` is older than ~90 days against its
     `source` URL and bump the date (`somm plans --catalog` lists them
     with ages). Vendors change subscription limits far more often than
     token prices — treat unverified entries as suspect.

7. **Run the test suite one more time.** Version bumps occasionally
   touch version-format tests.
   ```bash
   uv run pytest -q
   ```

8. **Commit the release.** Two commits is the convention when the
   release bundles work that pre-dates the version bump: one
   `chore(release): X.Y.Z` that is *only* version + changelog + index,
   then the feature commits (or vice versa — whatever keeps the tag
   pointing at a clean state).

9. **Tag and push.**
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z — one-line summary"
   git push origin main
   git push origin vX.Y.Z
   ```

10. **Create the GitHub release.** Use `gh release create` with
   `--notes` via a heredoc so Markdown renders cleanly. Keep the
   release notes focused on what users care about: new features,
   breaking changes, migration notes. Link to the full diff:
   ```bash
   gh release create vX.Y.Z \
     --title "vX.Y.Z — short tagline" \
     --notes "$(cat <<'EOF'
   ## Highlights
   …

   ## Upgrade notes
   …

   **Full diff:** https://github.com/lavallee/somm/compare/vX.Y.Z-1...vX.Y.Z
   EOF
   )"
   ```

## Post-release

- Confirm the release at https://github.com/lavallee/somm/releases.
- If `docs/index.html` changed, wait ~1 minute for GitHub Pages to
  deploy, then verify the version badge updated.
- PyPI publishing is automatic: creating the GitHub release triggers
  `.github/workflows/publish.yml` (trusted publishing, per-package
  environments, `skip-existing`). Verify the run went green and the new
  version shows on https://pypi.org/project/somm/.

## Release memory

Over time, recurring gotchas accrete. Append them here so the next
release doesn't re-discover them.

- **Stale `VERSION` import.** Don't use the literal string `0.1.0` in
  tests. Import `somm_core.VERSION`.
- **uv.lock churn.** `uv sync --all-packages` after the bump often
  rewrites the lockfile; commit it as part of the release commit
  (otherwise the next PR opens with an unrelated diff).
- **Mirror of older DBs.** Schema migrations run on first open. A
  release with a new schema must NOT be mixed with older libraries
  pointing at the same global `.sqlite` — bump + migrate first.
- **The publish action ages out from under the build.** 0.15.0's first
  publish failed on every package with `InvalidDistribution: '2.5' is
  not a valid metadata version`. Nothing was wrong with the release:
  `uv build` had moved on to `Metadata-Version: 2.5` and the SHA-pinned
  `gh-action-pypi-publish` (Feb 2026) shipped a twine that could not
  read it. Pinning by SHA is right for supply-chain reasons and means
  the pin silently rots — when a publish fails at the upload step while
  the build step passes, suspect the pin before the artifacts.
- **`main` is protected: PR + 3 status checks.** A release cannot be
  pushed straight to `main`, and merge commits are refused (linear
  history) — use `gh pr merge --rebase` so the tag can point at the
  release commit. Do NOT push the tag before the PR merges: it will
  point at a local commit that the rebase then rewrites, leaving the
  tag dangling.

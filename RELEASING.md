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
- **Major (0.x → 1.0)** — breaking changes to the library surface, MCP
  tool signatures, or schema semantics.

Schema migrations bump `SCHEMA_VERSION` in lockstep with a new
`packages/somm-core/src/somm_core/migrations/NNNN_<name>.sql` file. One
migration per release is the norm; multiple is fine if they're
independent.

## The checklist

1. **All tests pass locally.**
   ```bash
   uv run pytest -q
   ```
   Two skips are expected (`no local ollama`). Any other skip or
   failure blocks release.

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
   One-liner:
   ```bash
   OLD=0.1.1; NEW=0.2.0
   for f in pyproject.toml packages/*/pyproject.toml; do
     sed -i '' "s/version = \"$OLD\"/version = \"$NEW\"/" "$f"
   done
   sed -i '' "s/VERSION = \"$OLD\"/VERSION = \"$NEW\"/" packages/somm-core/src/somm_core/version.py
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

5. **Run the test suite one more time.** Version bumps occasionally
   touch version-format tests.
   ```bash
   uv run pytest -q
   ```

6. **Commit the release.** Two commits is the convention when the
   release bundles work that pre-dates the version bump: one
   `chore(release): X.Y.Z` that is *only* version + changelog + index,
   then the feature commits (or vice versa — whatever keeps the tag
   pointing at a clean state).

7. **Tag and push.**
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z — one-line summary"
   git push origin main
   git push origin vX.Y.Z
   ```

8. **Create the GitHub release.** Use `gh release create` with
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
- If we ever publish to PyPI (not yet as of 0.2.0), the publish step
  goes between 7 and 8 and requires a trusted-publisher workflow.

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

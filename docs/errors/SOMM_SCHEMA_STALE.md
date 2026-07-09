# SOMM_SCHEMA_STALE

**Problem.** The SQLite database on disk is on an older schema version
than the installed somm library expects.

**Why.** You upgraded `somm` (or one of the `somm-*` packages) while an
older process was still running, or the installed package set is missing
the migration files required by its schema version.

**Fix.**

1. Upgrade all `somm-*` packages used by this project to the same release.

2. Stop any running processes that still import the old version:
   ```bash
   # if somm serve is running, Ctrl-C it or stop it the way you started it.
   pkill -f "somm serve"
   pkill -f "somm-serve"
   ```

3. Re-run any somm command. Repository initialization applies pending
   migrations automatically:
   ```bash
   somm doctor --project my_project
   ```

4. Restart your own daemons after the command succeeds:
   ```bash
   somm serve --project my_project &
   # + whatever uses the library
   ```

**What's a schema version?** Every `somm-core` release pins a
`SCHEMA_VERSION` constant; on first DB access the library compares the
value to the highest-applied migration and applies pending files from
`packages/somm-core/src/somm_core/migrations/`. Migrations are
append-only and idempotent.

**Related.**
- [`SOMM_PORT_BUSY`](./SOMM_PORT_BUSY.md) — happens when a prior somm
  service is still holding the port during restart.

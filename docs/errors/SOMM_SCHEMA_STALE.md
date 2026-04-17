# SOMM_SCHEMA_STALE

**Problem.** The SQLite database on disk is on an older schema version
than the installed somm library expects.

**Why.** You upgraded `somm` (or one of the `somm-*` packages) and the
schema migrations in the new version haven't been applied yet.

**Fix.**

1. Stop any running somm services:
   ```bash
   # if somm serve is running, Ctrl-C it or:
   pkill -f "somm serve"
   pkill -f "somm-serve"
   ```

2. Apply the migrations:
   ```bash
   # future: `somm migrate` — until then, re-running any somm command
   # triggers `ensure_schema()` on library init, which applies
   # pending migrations inside a transaction per file.
   somm doctor --project my_project
   ```

3. Restart your processes:
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

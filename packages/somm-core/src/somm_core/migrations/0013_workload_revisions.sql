-- Migration 0013 — append-only workload config revision history.
--
-- Workloads keep their mutable config columns as the live fast-read path for
-- routing. This table records post-change snapshots for audit, diff, and
-- forward-only rollback.

BEGIN;

CREATE TABLE workload_revisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workload_id TEXT NOT NULL REFERENCES workloads(id),
    revision    INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by  TEXT
);

CREATE UNIQUE INDEX idx_workload_revisions_wl_rev
    ON workload_revisions(workload_id, revision);
CREATE INDEX idx_workload_revisions_wl
    ON workload_revisions(workload_id, created_at);

COMMIT;

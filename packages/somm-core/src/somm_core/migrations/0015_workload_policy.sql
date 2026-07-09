-- Migration 0015 — per-workload routing policy fast-read path.
--
-- The live workloads row carries the current policy for generate() routing;
-- workload_revisions stores versioned snapshots of the same config.

BEGIN;
ALTER TABLE workloads ADD COLUMN policy_json TEXT;
COMMIT;

-- Migration 0010 — drop calls.commission_id (contract phase).
--
-- Completes the 0009 expand-contract: correlation_id (populated via
-- somm.hooks) is the only external-correlation column. commission_id
-- was kept through 0.3.0 so telemetry writers still running pre-0.3
-- code survived a migrated database; with those writers restarted, the
-- dead column (all NULL, never read) can go.
--
-- A writer that somehow still runs pre-0.3 code after this migration
-- fails its INSERT and spills to the JSONL spool — recover with
-- `somm drain-spool` after restarting it.

ALTER TABLE calls DROP COLUMN commission_id;

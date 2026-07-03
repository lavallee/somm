-- Migration 0009 — add calls.correlation_id (expand phase).
--
-- Generalizes the 0007 commission_id column into a neutral extension
-- point: any external system can stamp its own id (request id, trace
-- id, job id) on somm calls via somm.hooks and join somm telemetry to
-- its own records.
--
-- Expand-contract: commission_id is left in place (dead, all NULL —
-- no reader ever shipped) so telemetry writers still running pre-0.3
-- code keep inserting successfully against a migrated database.
-- A future migration drops commission_id once no pre-0.3 writers
-- remain (tracked in ROADMAP.md).

ALTER TABLE calls ADD COLUMN correlation_id TEXT;
DROP INDEX IF EXISTS idx_calls_commission;
CREATE INDEX IF NOT EXISTS idx_calls_correlation ON calls(correlation_id, ts);

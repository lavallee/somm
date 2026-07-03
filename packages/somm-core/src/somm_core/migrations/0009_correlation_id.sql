-- Migration 0009 — rename calls.commission_id → calls.correlation_id.
--
-- The column generalizes from one specific upstream integration to a
-- neutral extension point: any external system can stamp its own id
-- (request id, trace id, job id) on somm calls via somm.hooks and join
-- somm telemetry to its own records. The column was write-only until
-- now (no reader shipped), so the rename is data-safe.

DROP INDEX IF EXISTS idx_calls_commission;
ALTER TABLE calls RENAME COLUMN commission_id TO correlation_id;
CREATE INDEX IF NOT EXISTS idx_calls_correlation ON calls(correlation_id, ts);

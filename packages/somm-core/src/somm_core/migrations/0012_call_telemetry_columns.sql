-- Migration 0012 — add call telemetry columns for streaming, trace grouping,
-- prompt-cache usage, and grounded-search citations.
--
-- Additive only: all columns are nullable so v11 databases migrate cleanly.

ALTER TABLE calls ADD COLUMN ttft_ms INTEGER;
ALTER TABLE calls ADD COLUMN session_id TEXT;
ALTER TABLE calls ADD COLUMN parent_call_id TEXT;
ALTER TABLE calls ADD COLUMN cache_tokens_in INTEGER;
ALTER TABLE calls ADD COLUMN cache_tokens_out INTEGER;
ALTER TABLE calls ADD COLUMN citations_json TEXT;

CREATE INDEX IF NOT EXISTS idx_calls_session_ts ON calls(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_calls_parent_call ON calls(parent_call_id);


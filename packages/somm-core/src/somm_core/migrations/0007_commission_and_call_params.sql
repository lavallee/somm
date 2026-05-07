-- Migration 0007 — commission_id linkage to scribe + per-call params.
--
-- commission_id ties an LLM call to its parent commission (scribe.commissions.id).
-- Auto-populated from scribe's contextvar when present; NULL otherwise. This
-- is the spine that lets cross-tool audit views answer "which model calls
-- happened inside this commission?" without joining across two databases.
--
-- temperature / max_tokens / top_p / stop_sequences_json record the exact
-- params the caller asked for. Providers may inject their own defaults, but
-- this is what the application requested — the "intent" not the "translation".

ALTER TABLE calls ADD COLUMN commission_id TEXT;
ALTER TABLE calls ADD COLUMN temperature REAL;
ALTER TABLE calls ADD COLUMN max_tokens INTEGER;
ALTER TABLE calls ADD COLUMN top_p REAL;
ALTER TABLE calls ADD COLUMN stop_sequences_json TEXT;

CREATE INDEX IF NOT EXISTS idx_calls_commission ON calls(commission_id, ts);

-- Migration 0007 — external-correlation column + per-call params.
--
-- commission_id ties an LLM call to a record in an external system
-- (renamed to correlation_id in 0009, where the column generalized into a
-- neutral hook — see somm.hooks). Auto-populated from a registered
-- correlation provider when present; NULL otherwise. This is the spine
-- that lets cross-tool audit views answer "which model calls happened
-- inside this parent task?" without joining across two databases.
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

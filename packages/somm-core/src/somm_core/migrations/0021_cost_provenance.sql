-- Migration 0021 — make monetary provenance explicit on every call.
-- Legacy rows remain unknown: their exact write path cannot be reconstructed.

ALTER TABLE calls ADD COLUMN cost_basis TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE calls ADD COLUMN cost_kind TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE calls ADD COLUMN cost_accuracy TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE calls ADD COLUMN cost_source TEXT;
ALTER TABLE calls ADD COLUMN pricing_version TEXT;

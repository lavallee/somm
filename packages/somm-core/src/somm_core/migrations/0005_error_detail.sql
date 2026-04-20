-- v5 — add error_detail column to calls.
-- Captures the provider's error body or the exception message (truncated to
-- 512 chars by the library) so operators can triage failures without
-- reconstructing from logs.

ALTER TABLE calls ADD COLUMN error_detail TEXT;

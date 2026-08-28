-- Migration 0023 — where a call came from.
--
-- A workload name says what kind of work a call is; it does not say which code
-- asked for it. Without that, telemetry and static analysis describe the same
-- fleet in two vocabularies that cannot be joined: static analysis has the file
-- and line, telemetry has the workload and the outcome, and nothing links them.
--
-- Nullable with no default on purpose. Existing rows predate capture and are
-- honestly unknown; an empty string would read as an answer.

ALTER TABLE calls ADD COLUMN call_site TEXT;

CREATE INDEX idx_calls_call_site ON calls(project, call_site);

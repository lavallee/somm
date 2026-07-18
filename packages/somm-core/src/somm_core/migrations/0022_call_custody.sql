-- Migration 0022 — first-class request custody and economic eligibility.
-- Production and auxiliary provider requests remain ordinary call rows.
-- Imported OTLP rows are explicitly foreign and excluded from hot-path policy.

ALTER TABLE calls ADD COLUMN observation_role TEXT NOT NULL DEFAULT 'production';
ALTER TABLE calls ADD COLUMN source_call_id TEXT;
ALTER TABLE calls ADD COLUMN eval_result_id INTEGER;
ALTER TABLE calls ADD COLUMN provider_request_id TEXT;
ALTER TABLE calls ADD COLUMN billing_id TEXT;
ALTER TABLE calls ADD COLUMN origin TEXT NOT NULL DEFAULT 'native';
ALTER TABLE calls ADD COLUMN budget_eligible INTEGER NOT NULL DEFAULT 1;

CREATE INDEX idx_calls_source_call ON calls(source_call_id, ts);
CREATE INDEX idx_calls_eval_result ON calls(eval_result_id, ts);
CREATE INDEX idx_calls_provider_request ON calls(provider, provider_request_id);
CREATE INDEX idx_calls_billing_id ON calls(provider, billing_id);
CREATE INDEX idx_calls_role_ts ON calls(observation_role, ts);

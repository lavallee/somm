-- Migration 0008 — learned per-(workload, provider, model) parameter overrides.
--
-- Self-healing substrate. The agent worker detects a recurring capability
-- failure signature for a (workload, provider, model) — e.g. a reasoning model
-- that exhausts its output-token budget on thinking and returns empty
-- (Outcome.EMPTY with error_detail '… stripped_empty …') — and writes an
-- override here that raises the effective max_tokens floor on subsequent calls.
-- SommLLM.generate() applies the override with a fail-open lookup before
-- building the request; it only ever RAISES a parameter, never lowers it.
--
-- Keyed by (workload_id, provider, model). Additive: old code that doesn't
-- know this table simply never reads it.

CREATE TABLE IF NOT EXISTS learned_param_overrides (
    workload_id        TEXT NOT NULL,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    max_tokens_floor   INTEGER,                 -- raise max_tokens to at least this (NULL = no bump)
    failure_signature  TEXT NOT NULL,           -- e.g. 'capability_empty:stripped_empty'
    evidence_json      TEXT NOT NULL DEFAULT '{}',
    confidence         REAL NOT NULL DEFAULT 0.0,
    applied_count      INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workload_id, provider, model)
);

-- generate() looks up by (workload_id, model), preferring an exact provider
-- match — so index the hot-path read.
CREATE INDEX IF NOT EXISTS idx_learned_override_lookup
    ON learned_param_overrides(workload_id, model);

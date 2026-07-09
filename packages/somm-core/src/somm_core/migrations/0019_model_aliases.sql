-- Migration 0019 — canonical model aliases.
--
-- One underlying model may appear under multiple provider/model route IDs
-- (for example a native provider ID and an OpenRouter wrapper). This table
-- lets ranking and prior-decision recall treat those rows as the same model
-- while preserving the concrete route the caller can actually use.

CREATE TABLE model_aliases (
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    canonical_id  TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'manual',
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (provider, model)
);

CREATE INDEX idx_model_aliases_canonical
    ON model_aliases(canonical_id);

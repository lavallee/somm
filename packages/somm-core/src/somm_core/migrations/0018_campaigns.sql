-- Migration 0018 — experiment campaign ledger.
--
-- Campaigns are durable, append-only experiment records: a metric contract,
-- token budget, plateau settings, and a JSONL-shaped keep/revert event log.

CREATE TABLE campaigns (
    id              TEXT PRIMARY KEY,
    project         TEXT NOT NULL,
    workload_id     TEXT NOT NULL REFERENCES workloads(id),
    dataset_id      TEXT REFERENCES datasets(id),
    name            TEXT NOT NULL,
    metric          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('gte', 'lte')),
    threshold       REAL NOT NULL,
    token_budget    INTEGER,
    max_rounds      INTEGER NOT NULL,
    plateau_window  INTEGER NOT NULL,
    min_delta       REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'running',
    best_score      REAL,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_cost_usd  REAL NOT NULL DEFAULT 0.0,
    metadata_json   TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP
);

CREATE INDEX idx_campaigns_project_workload
    ON campaigns(project, workload_id, created_at);
CREATE INDEX idx_campaigns_dataset
    ON campaigns(dataset_id, created_at);
CREATE INDEX idx_campaigns_status
    ON campaigns(status, updated_at);

CREATE TABLE campaign_events (
    id             TEXT PRIMARY KEY,
    campaign_id    TEXT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    sequence       INTEGER NOT NULL,
    run_id         TEXT,
    event_type     TEXT NOT NULL,
    action         TEXT NOT NULL CHECK (action IN ('start', 'keep', 'revert', 'stop')),
    metric_score   REAL,
    threshold      REAL,
    tokens_in      INTEGER NOT NULL DEFAULT 0,
    tokens_out     INTEGER NOT NULL DEFAULT 0,
    total_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0.0,
    payload_json   TEXT NOT NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(campaign_id, sequence)
);

CREATE INDEX idx_campaign_events_campaign
    ON campaign_events(campaign_id, sequence);
CREATE INDEX idx_campaign_events_run
    ON campaign_events(run_id);

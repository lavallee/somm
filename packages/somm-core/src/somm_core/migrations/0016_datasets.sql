-- Migration 0016 — durable eval datasets promoted from sampled calls.
--
-- samples remains transient, opt-in capture storage. datasets/dataset_items
-- are explicit golden fixtures used by synchronous eval runs and CI gates.

CREATE TABLE datasets (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    workload_id TEXT NOT NULL REFERENCES workloads(id),
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project, workload_id, name)
);

CREATE INDEX idx_datasets_project_workload_name
    ON datasets(project, workload_id, name);
CREATE INDEX idx_datasets_workload
    ON datasets(workload_id, created_at);

CREATE TABLE dataset_items (
    id                     TEXT PRIMARY KEY,
    dataset_id             TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    source_call_id         TEXT REFERENCES calls(id),
    prompt_body            TEXT NOT NULL,
    expected_response_body TEXT NOT NULL,
    metadata_json          TEXT,
    created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, source_call_id)
);

CREATE INDEX idx_dataset_items_dataset
    ON dataset_items(dataset_id, created_at);
CREATE INDEX idx_dataset_items_source_call
    ON dataset_items(source_call_id);

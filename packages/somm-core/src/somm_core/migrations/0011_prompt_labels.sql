-- Migration 0011 — prompt labels and fork lineage.
--
-- Adds a mutable pointer layer on top of immutable, content-addressed prompts:
-- labels are moved by upsert and every move is recorded in append-only history.
-- Forks record their source prompt id on the new prompt row.

CREATE TABLE IF NOT EXISTS prompt_labels (
    workload_id TEXT NOT NULL REFERENCES workloads(id),
    label       TEXT NOT NULL,
    prompt_id   TEXT NOT NULL REFERENCES prompts(id),
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by  TEXT,
    PRIMARY KEY (workload_id, label)
);

CREATE TABLE IF NOT EXISTS prompt_label_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workload_id TEXT NOT NULL,
    label       TEXT NOT NULL,
    prompt_id   TEXT NOT NULL,
    moved_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    moved_by    TEXT
);
CREATE INDEX IF NOT EXISTS idx_prompt_label_history_lookup
    ON prompt_label_history(workload_id, label, moved_at);

ALTER TABLE prompts ADD COLUMN parent_prompt_id TEXT;
CREATE INDEX IF NOT EXISTS idx_prompts_parent_prompt_id ON prompts(parent_prompt_id);

-- Migration 0017 — first-class eval receipt records.
--
-- eval_results stays the compact scoring table. eval_receipts stores the
-- structured evidence behind dataset runs, shadow judges, and paired A/B
-- comparisons without packing every receipt shape into judge_reason.

CREATE TABLE eval_receipts (
    id                  TEXT PRIMARY KEY,
    eval_result_id      INTEGER REFERENCES eval_results(id),
    run_id              TEXT,
    receipt_type        TEXT NOT NULL,
    call_id             TEXT REFERENCES calls(id),
    dataset_id          TEXT REFERENCES datasets(id),
    dataset_item_id     TEXT REFERENCES dataset_items(id),
    source_call_id      TEXT REFERENCES calls(id),
    candidate_a_call_id TEXT REFERENCES calls(id),
    candidate_b_call_id TEXT REFERENCES calls(id),
    winner              TEXT,
    score               REAL,
    threshold           REAL,
    payload_json        TEXT NOT NULL,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_eval_receipts_eval_result
    ON eval_receipts(eval_result_id);
CREATE INDEX idx_eval_receipts_run
    ON eval_receipts(run_id, created_at);
CREATE INDEX idx_eval_receipts_call
    ON eval_receipts(call_id, created_at);
CREATE INDEX idx_eval_receipts_dataset
    ON eval_receipts(dataset_id, dataset_item_id, created_at);
CREATE INDEX idx_eval_receipts_pair
    ON eval_receipts(candidate_a_call_id, candidate_b_call_id, created_at);

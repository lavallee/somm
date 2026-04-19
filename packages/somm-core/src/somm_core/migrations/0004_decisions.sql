-- somm schema v4 — the sommelier layer.
-- Decisions are *advisory memory*: a user + coding agent reasoned through
-- a question ("what vision model should we use?"), considered a set of
-- candidates, and committed to a choice. Unlike calls (per-project by
-- default), decisions are always mirrored to the global store — their
-- value is explicitly cross-project: "last time I picked a vision model,
-- here's what worked."
--
-- `question_hash` is the stable_hash of a normalised question string,
-- letting us dedup & aggregate "how many projects decided this same
-- thing?" without storing the raw prompt.

CREATE TABLE IF NOT EXISTS decisions (
    id                    TEXT PRIMARY KEY,                -- UUID4
    ts                    TIMESTAMP NOT NULL,
    project               TEXT NOT NULL,
    workload_id           TEXT,                            -- nullable: not all decisions are workload-bound
    workload_name         TEXT,                            -- denormalised for cross-project readability
    question              TEXT NOT NULL,                   -- natural-language question
    question_hash         TEXT NOT NULL,                   -- stable_hash(question)
    constraints_json      TEXT,                            -- capability / price / provider filters used
    candidates_json       TEXT NOT NULL,                   -- list of {provider, model, score_reasons, ...}
    chosen_provider       TEXT,
    chosen_model          TEXT,
    rationale             TEXT NOT NULL,                   -- why this was picked
    agent                 TEXT,                            -- "claude-code" | "cursor" | "human" | ...
    superseded_by         TEXT,                            -- id of a later decision that replaced this
    outcome_note          TEXT                             -- optional retrospective: "worked well / had issues"
);
CREATE INDEX IF NOT EXISTS idx_decisions_project_ts    ON decisions(project, ts DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_question_hash ON decisions(question_hash);
CREATE INDEX IF NOT EXISTS idx_decisions_workload      ON decisions(workload_id);

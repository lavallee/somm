-- somm schema v1 — initial
-- Self-hosted local SQLite in WAL mode. All IDs are content-addressed or UUID4.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- First-class entities -------------------------------------------------------

CREATE TABLE IF NOT EXISTS workloads (
    id                    TEXT PRIMARY KEY,              -- hash(name + schemas)
    name                  TEXT NOT NULL,
    project               TEXT NOT NULL,
    description           TEXT DEFAULT '',
    input_schema_json     TEXT,
    output_schema_json    TEXT,
    quality_criteria_json TEXT,
    budget_cap_usd_daily  REAL,
    privacy_class         TEXT NOT NULL DEFAULT 'internal',
    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_workloads_project_name ON workloads(project, name);

CREATE TABLE IF NOT EXISTS prompts (
    id           TEXT PRIMARY KEY,                       -- hash(body)
    workload_id  TEXT NOT NULL REFERENCES workloads(id),
    version      TEXT NOT NULL,
    hash         TEXT NOT NULL,
    body         TEXT NOT NULL,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    retired_at   TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_prompts_workload ON prompts(workload_id);

CREATE TABLE IF NOT EXISTS model_intel (
    provider             TEXT NOT NULL,
    model                TEXT NOT NULL,
    price_in_per_1m      REAL,
    price_out_per_1m     REAL,
    context_window       INTEGER,
    capabilities_json    TEXT,
    last_seen            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source               TEXT NOT NULL,
    PRIMARY KEY (provider, model)
);

-- Events (immutable rows) ----------------------------------------------------

CREATE TABLE IF NOT EXISTS calls (
    id             TEXT PRIMARY KEY,                     -- UUID4
    ts             TIMESTAMP NOT NULL,
    project        TEXT NOT NULL,
    workload_id    TEXT,
    prompt_id      TEXT,
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    tokens_in      INTEGER NOT NULL DEFAULT 0,
    tokens_out     INTEGER NOT NULL DEFAULT 0,
    latency_ms     INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0.0,
    outcome        TEXT NOT NULL DEFAULT 'ok',
    error_kind     TEXT,
    prompt_hash    TEXT NOT NULL,
    response_hash  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_workload_ts       ON calls(workload_id, ts);
CREATE INDEX IF NOT EXISTS idx_calls_provider_ts       ON calls(provider, ts);
CREATE INDEX IF NOT EXISTS idx_calls_workload_model_ts ON calls(workload_id, model, ts);
CREATE INDEX IF NOT EXISTS idx_calls_project_ts        ON calls(project, ts);

-- Late-arriving metadata (outcome marks from result.mark(), etc.)
CREATE TABLE IF NOT EXISTS call_updates (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id   TEXT NOT NULL REFERENCES calls(id),
    field     TEXT NOT NULL,
    value     TEXT NOT NULL,
    ts        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_call_updates_call ON call_updates(call_id);

-- Prompt/response sampling — OFF by default; per-workload opt-in.
CREATE TABLE IF NOT EXISTS samples (
    call_id        TEXT PRIMARY KEY REFERENCES calls(id),
    prompt_body    TEXT NOT NULL,
    response_body  TEXT NOT NULL
);

-- Intelligence ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS eval_results (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id               TEXT NOT NULL REFERENCES calls(id),
    gold_model            TEXT NOT NULL,
    gold_response_hash    TEXT,
    structural_score      REAL,
    embedding_score       REAL,
    judge_score           REAL,
    judge_reason          TEXT,
    grading_started_at    TIMESTAMP,                     -- lease for crash recovery
    ts                    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_eval_results_call  ON eval_results(call_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_lease ON eval_results(grading_started_at);

CREATE TABLE IF NOT EXISTS recommendations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    workload_id       TEXT NOT NULL REFERENCES workloads(id),
    action            TEXT NOT NULL,
    evidence_json     TEXT NOT NULL,
    expected_impact   TEXT,
    confidence        REAL,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dismissed_at      TIMESTAMP,
    applied_at        TIMESTAMP
);

-- Operational ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS provider_health (
    provider              TEXT NOT NULL,
    model                 TEXT NOT NULL,
    last_ok_at            TIMESTAMP,
    cooldown_until        TIMESTAMP,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, model)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_name              TEXT PRIMARY KEY,
    due_at                TIMESTAMP NOT NULL,
    locked_until          TIMESTAMP,
    last_started_at       TIMESTAMP,
    last_success_at       TIMESTAMP,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    interval_seconds      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_name           TEXT PRIMARY KEY,
    last_run_at           TIMESTAMP,
    last_success_at       TIMESTAMP,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0
);

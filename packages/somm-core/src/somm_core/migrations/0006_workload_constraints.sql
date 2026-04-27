-- v6 — workload adequacy constraints + classified-call view.
--
-- Adds three optional constraint columns to workloads so "is this model
-- performing adequately?" becomes a query against the data, not a judgment
-- call.  Each constraint is nullable; null means "no opinion."
--
--   max_p95_latency_ms          — Tier 1 (timeliness) ceiling
--   max_capability_failure_rate — Tier 2/3 (model-traceable failures) ceiling, 0–1
--   max_cost_per_call_usd       — cost ceiling per ok call
--
-- Also creates v_calls_classified, which decorates each call row with a
-- `failure_class` token and two boolean columns splitting capability signals
-- (model is unfit) from detractors (provider/network is flaky).  This is the
-- substrate the frontier query in repository.py:workload_frontier joins on.

ALTER TABLE workloads ADD COLUMN max_p95_latency_ms INTEGER;
ALTER TABLE workloads ADD COLUMN max_capability_failure_rate REAL;
ALTER TABLE workloads ADD COLUMN max_cost_per_call_usd REAL;

DROP VIEW IF EXISTS v_calls_classified;
CREATE VIEW v_calls_classified AS
SELECT
    c.*,
    CASE c.outcome
        WHEN 'ok'             THEN 'none'
        WHEN 'bad_json'       THEN 'capability_payload'
        WHEN 'off_task'       THEN 'capability_payload'
        WHEN 'empty'          THEN 'capability_empty'
        WHEN 'timeout'        THEN 'detractor_timeout'
        WHEN 'rate_limit'     THEN 'detractor_rate_limit'
        WHEN 'upstream_error' THEN 'detractor_upstream'
        WHEN 'exhausted'      THEN 'meta_exhausted'
        ELSE                       'unknown'
    END AS failure_class,
    CASE c.outcome
        WHEN 'bad_json' THEN 1
        WHEN 'off_task' THEN 1
        WHEN 'empty'    THEN 1
        ELSE 0
    END AS is_capability_signal,
    CASE c.outcome
        WHEN 'timeout'        THEN 1
        WHEN 'rate_limit'     THEN 1
        WHEN 'upstream_error' THEN 1
        ELSE 0
    END AS is_detractor
FROM calls c;

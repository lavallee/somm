-- somm schema v2 — add shadow-eval config per workload + gold-model tracking.

ALTER TABLE workloads ADD COLUMN shadow_config_json TEXT;

-- Convenience view for picking sample candidates (most recent un-graded calls
-- per workload). Keeps the worker's SQL tight.
CREATE VIEW IF NOT EXISTS shadow_candidates AS
SELECT
    c.id            AS call_id,
    c.ts            AS ts,
    c.project       AS project,
    c.workload_id   AS workload_id,
    c.provider      AS provider,
    c.model         AS model,
    c.prompt_hash   AS prompt_hash,
    c.response_hash AS response_hash,
    w.name          AS workload_name,
    w.privacy_class AS privacy_class,
    w.shadow_config_json AS shadow_config_json
FROM calls c
JOIN workloads w ON w.id = c.workload_id
LEFT JOIN eval_results e ON e.call_id = c.id
WHERE e.id IS NULL
  AND c.outcome = 'ok'
  AND w.shadow_config_json IS NOT NULL
  AND w.privacy_class != 'private';

-- Migration 0025 — when a model first appeared in the catalog.
--
-- model_intel.last_seen is refreshed on every intel pass, so it says a model
-- is still listed, not that it is new. The agent's new_model_landed rule read
-- it as "appeared recently" and proposed every cheaper sibling already in the
-- catalog, once per workload, for as long as the catalog listed it: 40 of the
-- 49 open recommendations in the October 2026 census were that rule pointing
-- at models older than the one in production.
--
-- first_seen is set on insert and never updated. Existing rows are backfilled
-- from last_seen, so everything already known counts as having landed
-- together and nothing in the table is new relative to anything else in it.

ALTER TABLE model_intel ADD COLUMN first_seen TIMESTAMP;
UPDATE model_intel SET first_seen = last_seen WHERE first_seen IS NULL;

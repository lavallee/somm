-- Migration 0027 — data boundary.
--
-- workloads.data_boundary: the loosest data tier (local, zdr, subscribed,
-- unvetted) a workload's data may reach. NULL means derive it: a private
-- workload is local, otherwise the project default, otherwise unvetted.
--
-- calls.data_tier: the tier of the route that served the call, or was
-- attempted. NULL on rows written before this column existed and on boundary
-- refusals. `somm boundary audit` classifies NULL rows from provider and model.

ALTER TABLE workloads ADD COLUMN data_boundary TEXT;
ALTER TABLE calls ADD COLUMN data_tier TEXT;

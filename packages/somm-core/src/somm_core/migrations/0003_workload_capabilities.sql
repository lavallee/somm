-- somm schema v3 — per-workload required-capability defaults.
-- A JSON array of capability tokens (e.g. ["vision"]) that every call
-- under this workload requires of the serving (provider, model). The
-- router merges this with capabilities auto-inferred from the prompt.

ALTER TABLE workloads ADD COLUMN capabilities_required_json TEXT;

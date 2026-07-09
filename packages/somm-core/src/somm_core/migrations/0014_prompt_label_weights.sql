-- Migration 0014 — weighted prompt labels for production A/B tests.
--
-- A label remains either a single prompt pointer (weights_json NULL) or a
-- weighted distribution keyed by prompt id. prompt_id is retained as the
-- deterministic no-bucket fallback for weighted labels.

ALTER TABLE prompt_labels ADD COLUMN weights_json TEXT;

-- Migration 0024 — two things a row could not say.
--
-- reasoning_tokens: tokens a model spent thinking before it answered, when the
-- provider breaks them out (OpenAI-compatible completion_tokens_details).
-- They are billed as output and already counted in tokens_out; recording them
-- separately is what lets a cost question distinguish "long answer" from
-- "long think". NULL where the provider does not report them.
--
-- client_version: the somm library version that wrote the row. A fleet of
-- projects upgrades unevenly; in October 2026 most rows were written by
-- clients two minor versions behind, so columns the newer client fills were
-- empty for reasons the ledger could not explain. NULL on rows written before
-- this column existed.

ALTER TABLE calls ADD COLUMN reasoning_tokens INTEGER;
ALTER TABLE calls ADD COLUMN client_version TEXT;

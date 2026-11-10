-- Migration 0028 — prompt-cache prices beside the input and output prices.
--
-- Calls already record cache-read and cache-write tokens (cache_tokens_in,
-- cache_tokens_out), but cost_for_call priced every prompt token at the full
-- input rate, so a cached Anthropic call was overstated by up to ten times.
-- NULL means the source published no cache rate for the model.

ALTER TABLE model_intel ADD COLUMN price_cache_read_per_1m REAL;
ALTER TABLE model_intel ADD COLUMN price_cache_write_per_1m REAL;

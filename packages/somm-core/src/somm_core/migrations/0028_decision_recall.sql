-- Migration 0028 — decision recall.
--
-- decisions.recalled_from: the id of the prior decision this one followed,
-- set only when the caller cites a prior decision surfaced by the sommelier
-- and the new decision chooses the same provider and model. A non-NULL value
-- is one "decision recalled", somm's north-star count. NULL on every row
-- written before this column existed and on fresh choices.

ALTER TABLE decisions ADD COLUMN recalled_from TEXT;

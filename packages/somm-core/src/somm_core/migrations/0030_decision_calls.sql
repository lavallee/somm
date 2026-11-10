-- Migration 0030 — decision calls.
--
-- calls.kind: what sort of call the row records. NULL is an ordinary call
-- (generate, stream, structured, embed), which is every row written before
-- this column existed. 'decide' is a call to a decision model: it returns a
-- typed label and a probability instead of text. Decision models bill per
-- input token, so tokens_in / cost_usd already carry the meter.
--
-- calls.label: the label the decision chose (a choice name, a scale level,
-- or yes / no). calls.p_top: the probability the model gave that label.
-- calls.abstained: 1 when p_top fell below the caller's min_p, else 0.
-- calls.result_json: the full typed answer (every label and probability).
-- All NULL on rows that are not decisions.

ALTER TABLE calls ADD COLUMN kind TEXT;
ALTER TABLE calls ADD COLUMN label TEXT;
ALTER TABLE calls ADD COLUMN p_top REAL;
ALTER TABLE calls ADD COLUMN abstained INTEGER;
ALTER TABLE calls ADD COLUMN result_json TEXT;

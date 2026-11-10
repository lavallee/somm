-- Migration 0031 — what a replay needs.
--
-- samples.request_json: the request parameters a shadow-eval sample was
-- sent with, so a grader can replay the real request rather than the bare
-- prompt. A JSON object with these keys, as sent:
--   system (str), tools (list), tool_choice (str | object | null),
--   response_format (object | null), reasoning_effort (str | null),
--   temperature (float), max_tokens (int)
-- Written under the same privacy, opt-in and sampling rules as the bodies.
-- NULL on every sample captured before this column existed.

ALTER TABLE samples ADD COLUMN request_json TEXT;

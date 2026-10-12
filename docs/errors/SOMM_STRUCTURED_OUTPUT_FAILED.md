# SOMM_STRUCTURED_OUTPUT_FAILED

**Problem.** `generate_structured()` (or `agenerate_structured()`) ran out
of retries without producing JSON that validates against the supplied
schema. The exception carries the last raw model text and the last
validation error.

**Why.** Common triggers:

- The model answered in prose, wrapped the JSON in commentary, or
  returned a fenced block the parser could not isolate.
- The provider does not honor `response_format={"type": "json_schema"}`
  and silently ignored it, so nothing constrained the output shape.
- The schema is stricter than the task allows (a required field the
  prompt never asks for, an enum the model does not know).
- A reasoning model spent `max_tokens` on thinking and returned a
  truncated object. This usually shows up first as
  [`SOMM_EMPTY_RESPONSE`](./SOMM_EMPTY_RESPONSE.md).

**Behavior.** somm asks once, validates, and on failure retries with a
corrective system prompt that quotes the validation error, up to
`retries` times. Every attempt is an ordinary `calls` row, so the
failures are visible in `somm status` and `somm tail`. The call is
pinned where the caller pinned it; a validation failure never falls
back to a different model.

**Fix.**

1. Read `exc.last_raw` and `exc.last_error`. Nine times out of ten the
   model is answering a slightly different question than the schema
   describes. Tighten the prompt or loosen the schema.
2. Give the model room: raise `max_tokens`, or choose a non-thinking
   model for this workload.
3. If the provider ignores `json_schema`, prefer a provider that
   supports it, or keep the schema small enough that prompt-mode
   coercion succeeds.
4. For the permissive contract, use `extract_structured()`, which
   returns the best-effort parse instead of raising.

**Related.**
- [`SOMM_EMPTY_RESPONSE`](./SOMM_EMPTY_RESPONSE.md) — blank output
  before validation even ran.

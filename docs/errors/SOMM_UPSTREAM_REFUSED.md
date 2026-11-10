# SOMM_UPSTREAM_REFUSED

**Problem.** The model, or the provider's content filter, declined to answer.

**Why.** The provider said so in its stop reason: `refusal` (Anthropic) or
`content_filter` (OpenAI-compatible endpoints). The request was delivered;
the answer was withheld or cut short.

**Behavior.** The call is recorded as delivered, with `stop_reason` on the
result and the row. `somm.SommRefused` is raised only when the caller passed
`raise_on_failure=True`; it carries `stop_reason`, `provider`, `model` and
`call_id`. It is not a parsing or schema problem, so a JSON repair pass will
not help.

**Fix.**

1. Read the prompt for what the provider's policy objects to.
2. Retrying the same prompt on the same model rarely changes the answer.
   If the content is legitimate for your use, route the workload to a
   provider whose policy fits it.

**Related.**
- [`SOMM_EMPTY_RESPONSE`](./SOMM_EMPTY_RESPONSE.md) — no text came back,
  with no refusal reason given.

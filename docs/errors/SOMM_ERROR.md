# SOMM_ERROR

**Problem.** The generic base code for `somm.errors.SommError`, the
root exception class every other somm error inherits from.

**Why you're reading this.** Every concrete somm exception overrides
`code` with a specific `SOMM_*` value (see the rest of this reference).
`SOMM_ERROR` itself is not raised anywhere in the library — if you're
seeing this exact code in a message, either:

- You caught `SommError` generically and are printing `.code` on an
  exception whose specific subclass you didn't check, or
- You're looking at a third-party provider adapter or extension that
  subclasses `SommError` directly without setting its own `code`.

**Fix.**

1. If you caught the base class, narrow the `except` to the concrete
   subclass to get the specific code and matching fix page:
   ```python
   import somm
   from somm.errors import SommError, SommProviderError

   try:
       llm.generate(prompt, workload="my_workload")
   except SommError as e:
       print(somm.describe_error(e))  # stable JSON-safe code + retry metadata
   ```

2. If you're implementing a third-party provider or extension, give
   your exception its own `code`:
   ```python
   from somm.errors import SommTransientError

   class MyProviderError(SommTransientError):
       code = "SOMM_MY_PROVIDER_WEIRD_STATE"
   ```

**Fields every somm error carries.** Each is `None` when somm does not
know it. `provider`, `model` and `call_id` name the route and the ledger
row. When the error came from a provider's HTTP answer, the upstream
detail rides along too:

| Attribute | What it holds |
|---|---|
| `status_code` | the exact upstream HTTP status (a 403 stays 403) |
| `provider_code` | the provider's error `code` (OpenAI `credit_balance_exhausted`, Anthropic `details.error_code`, OpenRouter's numeric code, MiniMax `base_resp.status_code`), as a string |
| `provider_type` | the provider's error `type` (`invalid_request_error`, `permission_error`, …) |
| `provider_param` | the request field the provider blamed (OpenAI `param`) |
| `provider_request_id` | the provider's request id (`request-id` / `x-request-id` header, or Anthropic's body `request_id`), for its support desk |
| `retry_after_s` | seconds the provider asked for (`Retry-After`, or on a 429 its rate-limit reset headers) |

None of these holds the provider's error *message*, which can echo the
prompt; the exception text keeps at most 200 characters of the body.
`somm.describe_error(e)` includes the ones that are set.

**Related.**
- All other pages in this reference — `SommError` is their common
  ancestor. Start from the specific code in your traceback rather than
  this page when you have one.

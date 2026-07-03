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
   from somm.errors import SommError, SommProviderError

   try:
       llm.generate(prompt, workload="my_workload")
   except SommError as e:
       print(type(e).__name__, e.code)   # narrow from here
   ```

2. If you're implementing a third-party provider or extension, give
   your exception its own `code`:
   ```python
   from somm.errors import SommTransientError

   class MyProviderError(SommTransientError):
       code = "SOMM_MY_PROVIDER_WEIRD_STATE"
   ```

**Related.**
- All other pages in this reference — `SommError` is their common
  ancestor. Start from the specific code in your traceback rather than
  this page when you have one.

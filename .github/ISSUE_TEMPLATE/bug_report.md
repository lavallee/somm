---
name: Bug report
about: Report a problem with somm
title: ""
labels: bug
---

## Describe the bug

A clear, concise description of what went wrong.

## Steps to reproduce

1. …
2. …
3. …

## Expected behavior

What you expected to happen instead.

## `somm doctor` output

```
paste the output of `somm doctor` here
```

## `calls.error_detail` (if relevant)

If the bug involves a failed or misclassified call, include the
relevant row's `error_detail` (and `error_kind`) from your local
telemetry database, or the CLI output that surfaces it
(e.g. `somm calls --status error`).

```
paste error_detail here
```

## Environment

- somm version:
- OS:
- Python version:
- Provider(s) in use:

## Additional context

Anything else that might help — config snippets (redact API keys),
logs, screenshots.

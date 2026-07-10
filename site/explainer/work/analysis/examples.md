## Purpose

This area provides runnable examples for adding Somm to existing Python applications with minimal code changes. It demonstrates compatibility with legacy LLM wrappers and OpenAI-style calls, plus privacy controls for sensitive workloads (`examples/README.md:3`).

## How it works

For an existing wrapper, `GenericLLMCompat` can replace the original LLM class at import time. Calls to `.generate()` retain familiar result fields while adding cost and provenance metadata; each client is explicitly closed after use (`examples/drop_in_wrapper.py:33`, `examples/drop_in_wrapper.py:36`). `.probe_providers()` supplies provider slots for distributing a batch across available providers (`examples/drop_in_wrapper.py:62`).

The OpenAI shim accepts chat-completion arguments and preserves access through `resp.choices[0].message.content`. A `provider/model` prefix pins execution, while an unprefixed model delegates provider selection to Somm’s router (`examples/openai_swap_in.py:24`, `examples/openai_swap_in.py:44`).

Sensitive workloads are registered once with `PrivacyClass.PRIVATE`. Subsequent calls using that workload are restricted to local providers; if none are available, the call fails instead of sending data upstream. Shadow evaluation is also rejected with `SommPrivacyViolation` (`examples/private_workload.py:18`, `examples/private_workload.py:27`, `examples/private_workload.py:36`).

## Key surfaces

- `GenericLLMCompat(project=...)` — drop-in legacy-wrapper client (`examples/drop_in_wrapper.py:33`).
- `llm.generate(...)` — submits a routed, instrumented generation request (`examples/drop_in_wrapper.py:40`).
- `llm.probe_providers(n)` — returns provider slots for batch work striping (`examples/drop_in_wrapper.py:67`).
- `openai_chat_completions(...)` — OpenAI-shaped chat-completions compatibility function (`examples/openai_swap_in.py:21`).
- `somm.llm(project=...)` — creates the native Somm client (`examples/private_workload.py:16`).
- `llm.repo.register_workload(...)` — records workload privacy and budget policy (`examples/private_workload.py:19`).
- `llm.enable_shadow(...)` — configures sampled shadow evaluation, subject to privacy enforcement (`examples/private_workload.py:38`).
- `somm status`, `somm tail`, and `somm serve` — inspect telemetry and run the dashboard (`examples/README.md:70`, `examples/README.md:78`).

## Design decisions

Compatibility layers minimize migration work while exposing Somm-specific metadata as optional extra attributes (`examples/openai_swap_in.py:36`). Provider pinning is explicit in model names, whereas unprefixed models remain routable (`examples/openai_swap_in.py:44`). Private workloads fail closed when local inference is unavailable, with a zero-dollar daily budget used as an additional safeguard (`examples/private_workload.py:24`, `examples/private_workload.py:27`). Explicit `try/finally` cleanup appears consistently around client lifecycles (`examples/drop_in_wrapper.py:39`, `examples/private_workload.py:17`).

## One-liner

These examples show how to graft Somm’s routing, telemetry, cost tracking, and privacy enforcement onto existing Python LLM call sites.
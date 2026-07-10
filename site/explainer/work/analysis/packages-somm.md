## Purpose

This package is Somm’s main Python library and CLI: it wraps LLM calls with provider routing, local telemetry, cost and budget controls, prompt management, evaluation, and operational diagnostics. It is designed to work with minimal configuration while keeping the primary call path self-hosted and locally recorded (`packages/somm/README.md:3`).

## How it works

`SommLLM` initializes a project repository, configured provider adapters, health tracker, router, pricing data, and an asynchronous telemetry writer (`packages/somm/src/somm/client.py:666`). `generate()` registers or validates the workload, merges declared and inferred capabilities, runs mutable pre-call hooks, checks the daily budget, and dispatches either to a pinned provider or the preference-ordered router (`packages/somm/src/somm/client.py:910`). The result and a normalized `Call` record contain provider/model attribution, tokens, cost, latency, hashes, outcome, errors, correlation data, cache usage, and citations (`packages/somm/src/somm/client.py:1265`).

The router filters incapable models before network access, applies quota pacing, skips cooled providers, and falls through on transient failures (`packages/somm/src/somm/routing.py:158`). Telemetry is batched into the local repository; database failures spill permission-restricted JSONL files for later atomic replay (`packages/somm/src/somm/telemetry.py:38`, `packages/somm/src/somm/telemetry.py:270`). Opt-in samples feed durable dataset evaluations and campaigns; failing graded calls can produce a versioned prompt proposal, while applying learned recommendations is a separate explicit operation (`packages/somm/src/somm/evals.py:80`, `packages/somm/src/somm/optimize.py:39`, `packages/somm/src/somm/recommendations.py:133`).

## Key surfaces

- `somm.llm(...)` / `SommLLM`: construct the project-scoped client (`packages/somm/src/somm/__init__.py:18`, `packages/somm/src/somm/client.py:658`).
- `SommLLM.generate(...)`: routed generation with tools, multimodal input, fallback, budgets, and telemetry (`packages/somm/src/somm/client.py:910`).
- `SommLLM.stream(...)`, `embed(...)`, and `generate_structured(...)`: streaming, local embeddings, and validated structured output (`packages/somm/src/somm/client.py:1634`, `packages/somm/src/somm/client.py:1430`, `packages/somm/src/somm/client.py:1575`).
- `SommLLM.register_prompt(...)` / `prompt(...)`: version and resolve prompts, including deterministic weighted labels (`packages/somm/src/somm/client.py:1846`, `packages/somm/src/somm/client.py:1880`).
- `hooks.register_hook(...)`: add prioritized `pre_call`, `post_call`, or background `post_process` behavior (`packages/somm/src/somm/hooks.py:160`).
- `somm` CLI: generate, inspect telemetry, manage prompts, run evals/campaigns, handle recommendations, and diagnose operation (`packages/somm/src/somm/cli.py:2051`).

## Design decisions

- Observe mode auto-registers workloads; strict mode makes missing registration an error, favoring initial adoption without removing enforceable governance (`packages/somm/src/somm/client.py:658`).
- Provider pins are “try first” by default; `no_fallback=True` supplies pinned-or-bust semantics for experiments where rerouting would invalidate results (`packages/somm/src/somm/client.py:1133`).
- Hooks, mirroring, learned overrides, and most telemetry side effects fail open so auxiliary intelligence cannot break live inference; hard budgets deliberately fail closed before dispatch (`packages/somm/src/somm/hooks.py:246`, `packages/somm/src/somm/client.py:571`).
- Shadow body capture is opt-in, deterministically sampled, size-capped, and forbidden for private workloads (`packages/somm/src/somm/client.py:850`).
- Optimization only creates a proposal label; it never promotes directly to staging or production (`packages/somm/src/somm/optimize.py:49`).

## One-liner

Somm turns ordinary LLM calls into a locally governed feedback loop for routing, telemetry, evaluation, and controlled improvement.
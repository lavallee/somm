## Purpose

Somm is a self-hosted intelligence loop for selecting, routing, observing, and improving LLM workloads across projects. Its Python library provides a zero-config local sensor, while optional service, MCP, evaluation, and admin surfaces turn accumulated telemetry into evidence-backed recommendations (`notes/CHARTER.md:3`, `notes/PLAN.md:5`).

## How it works

Calls enter through the library, which resolves a workload, selects a model, builds and executes the request with retries or fallbacks, parses the result, records telemetry, and runs post-processing hooks (`notes/GAMEPLAN-2026-07.md:68`). A per-process `WriterQueue` batches short writes into a project-local SQLite database; repeated lock failures spill to JSONL for later draining, preserving the no-service-required hot path (`notes/PLAN.md:1179`). Core records include workloads, immutable prompt versions, calls, samples, model intelligence, evaluation results, recommendations, and provider health (`notes/PLAN.md:162`).

When `somm serve` is active, scheduled workers grade sampled calls, refresh model metadata, and produce recommendations; MCP and the web admin read the same local state (`notes/PLAN.md:69`). Later phases add durable datasets, grading receipts, propose-only prompt optimization, experiment campaigns, and operator-controlled recommendation application (`notes/GAMEPLAN-2026-07.md:216`, `notes/GAMEPLAN-2026-07.md:248`). The latest security worklist says some read APIs, proxy dispatch, OTLP ingestion, and MCP comparison limits still require hardening (`notes/SOUNDCHECK-WORKLIST-2026-07-10.md:23`).

## Key surfaces

- `somm.llm(...)` / `SommLLM.generate()`, `.stream()`, and structured extraction: primary application-facing call path (`notes/PLAN.md:82`).
- `somm workload` and `somm prompt`: register, inspect, version, fork, and promote workload configuration and prompts (`notes/PLAN.md:1540`).
- `somm compare` and `somm replay`: compare models or rerun a recorded call (`notes/PLAN.md:1537`).
- `somm eval promote-call`, `somm eval run`, `somm optimize`, and `somm campaign run`: durable evaluation and experimentation surfaces (`notes/GAMEPLAN-2026-07.md:227`).
- `somm serve` / `somm service …`: start or manage the local API, dashboard, and workers (`notes/PLAN.md:121`, `notes/PLAN.md:1545`).
- MCP tools such as `somm_recommend`, `somm_compare`, `somm_replay`, and `somm_stats`: coding-agent interface (`notes/PLAN.md:96`).

## Design decisions

- SQLite remains the default because WAL fits continuous local writes plus concurrent readers; Postgres is optional for shared deployments (`notes/PLAN.md:189`, `notes/GAMEPLAN-2026-07.md:273`).
- Prompt versions and call records are immutable; labels and revisions provide mutable pointers and auditability (`notes/GAMEPLAN-2026-07.md:95`).
- Samples, cross-project mirroring, and shadow evaluation are opt-in, with privacy classification and budget ceilings (`notes/PLAN.md:422`).
- Optimization and recommendations are propose-only by default; automatic promotion is deliberately excluded (`notes/GAMEPLAN-2026-07.md:131`).
- Deferred features are promoted only by demonstrated demand, favoring optional extras over core bloat (`notes/TODOS.md:52`).

## One-liner

Somm turns local LLM calls into durable telemetry, safer routing, reproducible evaluations, and operator-controlled improvements without requiring a hosted commercial control plane.
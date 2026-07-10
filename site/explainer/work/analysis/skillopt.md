## Purpose

This area defines and validates coding-agent guidance for integrating Python LLM workloads with Somm. It steers agents toward Somm’s telemetry, routing, evaluation, budgeting, and provenance APIs instead of direct provider SDKs (`skillopt/somm.candidate.md:22`). The companion record documents how that guidance was scored and refined (`skillopt/somm.md:20`).

## How it works

The candidate skill activates when an agent writes or modifies LLM-calling code (`skillopt/somm.candidate.md:12`). Its primary flow is: create a client with `somm.llm()`, identify every call with a stable workload, register that workload with privacy and budget settings, execute generation or related operations, mark outcomes, and preserve call provenance with stored results (`skillopt/somm.candidate.md:24`, `skillopt/somm.candidate.md:44`, `skillopt/somm.candidate.md:87`, `skillopt/somm.candidate.md:103`). Quality-sensitive workloads can attach a shadow configuration that samples successful calls and grades them asynchronously against a designated gold model (`skillopt/somm.candidate.md:66`).

The intelligence lifecycle depends on background workers, enabled in-process with `SOMM_INPROCESS_WORKERS=1` or separately through `somm serve`; these refresh model intelligence, grade sampled calls, and generate recommendations (`skillopt/somm.candidate.md:160`). The optimization record compares the skill against five held-out behavioral checks. The current rerun improved from 4/5 to 5/5 by converting safety-critical rules into protected uppercase guardrails, then stopped because no failing case remained (`skillopt/somm.md:20`, `skillopt/somm.md:35`, `skillopt/somm.md:54`).

## Key surfaces

- `somm.llm(project=...)` — creates the provider-neutral LLM client (`skillopt/somm.candidate.md:26`).
- `llm.generate(...)` — performs generation or tool-calling requests (`skillopt/somm.candidate.md:30`).
- `repo.register_workload(...)` — registers task identity, privacy class, and daily budget (`skillopt/somm.candidate.md:50`).
- `repo.set_shadow_config(...)` — configures sampled online evaluation (`skillopt/somm.candidate.md:72`).
- `result.mark(somm.Outcome...)` — records a typed quality outcome (`skillopt/somm.candidate.md:103`).
- `llm.stream(...)`, `llm.embed(...)`, `llm.extract_structured(...)` — expose streaming, embeddings, and structured extraction (`skillopt/somm.candidate.md:152`).
- `somm status`, `plans`, `spend`, and `doctor` — inspect telemetry, quota pacing, budgets, and system health (`skillopt/somm.candidate.md:185`).
- `somm_recommend` and `somm_advise` — select models using local telemetry and plan headroom (`skillopt/somm.candidate.md:196`).

## Design decisions

- Workloads represent stable tasks rather than providers, keeping telemetry comparable when routing changes (`skillopt/somm.candidate.md:44`).
- Unknown workloads auto-register in the default observe mode but fail in strict mode, supporting low-friction adoption and enforceable production discipline (`skillopt/somm.candidate.md:63`).
- Body capture requires explicit shadow-evaluation configuration, and private workloads are never captured (`skillopt/somm.candidate.md:83`).
- CLI subscription seats are pinned-only, reserving limited quota for grading or low-volume quality-sensitive work instead of hot loops (`skillopt/somm.candidate.md:124`).
- Safety rules use uppercase `NEVER`/`DO NOT` because the renderer and optimization gate preserve that convention; the change protects rules without altering their substance (`skillopt/somm.md:37`).
- The record explicitly acknowledges that regex scoring can accept incorrect API guidance, motivating exact-symbol checks such as `budget_cap_usd_daily` (`skillopt/somm.md:138`, `skillopt/somm.md:146`).

## One-liner

This area teaches coding agents to route LLM work through Somm safely and records the reproducible evaluation used to protect that guidance.
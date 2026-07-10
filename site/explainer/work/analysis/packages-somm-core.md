## Purpose

`somm-core` is the dependency-free foundation shared by the Somm packages: typed telemetry records, local SQLite persistence, configuration, parsing, pricing, quota accounting, and evaluation primitives (`packages/somm-core/README.md:3`). It provides the durable data and intelligence substrate; provider dispatch and user-facing commands live elsewhere.

## How it works

Configuration resolves defaults, project `pyproject.toml`, environment variables, and explicit arguments, then selects a project-local or registered database (`packages/somm-core/src/somm_core/config.py:84`). Constructing `Repository` creates a permission-restricted SQLite database, applies migrations automatically, and configures WAL, foreign keys, and per-thread, fork-aware connections (`packages/somm-core/src/somm_core/repository.py:177`). Schema v19 evolves the original workload/prompt/call ledger into decisions, workload revisions, datasets, evaluation receipts, campaigns, and canonical model aliases (`packages/somm-core/src/somm_core/version.py:7`).

Calls are immutable telemetry events containing route, token, latency, cost, outcome, tracing, cache, and citation fields; late outcome changes go into `call_updates` (`packages/somm-core/src/somm_core/models.py:225`, `packages/somm-core/src/somm_core/repository.py:1403`). Opt-in captured samples can be promoted idempotently into durable golden datasets, graded with deterministic structural and text comparisons, and linked to structured evaluation receipts (`packages/somm-core/src/somm_core/repository.py:757`, `packages/somm-core/src/somm_core/graders.py:28`). Workload routing constraints remain on the live row for fast reads while every mutation appends a revision snapshot; rollback copies an old snapshot forward as a new revision (`packages/somm-core/src/somm_core/repository.py:402`, `packages/somm-core/src/somm_core/repository.py:579`).

Pricing is seeded and synchronized from an offline bundled snapshot, cached for ten minutes, and converted from token counts into per-call USD cost (`packages/somm-core/src/somm_core/pricing.py:108`, `packages/somm-core/src/somm_core/pricing.py:233`). Machine-local plan and registry data aggregate usage across project databases, calculate PAYG burn or metered-window pacing, and can infer quota ceilings from observed 429 events (`packages/somm-core/src/somm_core/plans.py:370`, `packages/somm-core/src/somm_core/plans.py:521`).

## Key surfaces

- `Config` / `load()` — resolve project, provider, storage, budget, service, and routing settings (`packages/somm-core/src/somm_core/config.py:24`, `packages/somm-core/src/somm_core/config.py:84`).
- `Repository` — primary persistence API for workloads, calls, datasets, evals, campaigns, decisions, aliases, and learned overrides (`packages/somm-core/src/somm_core/repository.py:177`).
- `ensure_schema()` — atomically applies packaged SQL migrations on startup (`packages/somm-core/src/somm_core/schema.py:80`).
- `grade_response_pair()` — runs dependency-free structural and text graders (`packages/somm-core/src/somm_core/graders.py:28`).
- `cost_for_call()` / `sync_bundled_pricing()` — calculate costs and populate offline model intelligence (`packages/somm-core/src/somm_core/pricing.py:233`, `packages/somm-core/src/somm_core/pricing.py:108`).
- `load_plans()` / `limit_statuses()` / `learn_observed_limits()` — load quota policy, report pacing, and learn limits from telemetry (`packages/somm-core/src/somm_core/plans.py:285`, `packages/somm-core/src/somm_core/plans.py:770`, `packages/somm-core/src/somm_core/plans.py:627`).
- `register_project()` / `fleet_db_paths()` — maintain the local cross-project database registry (`packages/somm-core/src/somm_core/registry.py:122`, `packages/somm-core/src/somm_core/registry.py:157`).

## Design decisions

- Privacy is structural: databases and registries use owner-only permissions, while prompt/response bodies are stored only through opt-in sampling (`packages/somm-core/src/somm_core/repository.py:185`, `packages/somm-core/src/somm_core/repository.py:642`).
- Migrations commit DDL and version stamps together, preventing partially upgraded databases (`packages/somm-core/src/somm_core/schema.py:91`).
- Advisory intelligence fails open: missing pricing becomes zero cost, and unreadable fleet databases contribute zero rather than breaking calls (`packages/somm-core/src/somm_core/pricing.py:240`, `packages/somm-core/src/somm_core/plans.py:376`).
- Binary judge prompt/parsing helpers exist, but `grade_response_pair()` still receives `None` from the placeholder `judge_score`; actual judge execution is not implemented here (`packages/somm-core/src/somm_core/graders.py:211`).

## One-liner

`somm-core` is the local, migration-backed ledger and intelligence toolkit that turns LLM calls into durable telemetry, cost, evaluation, and routing evidence.
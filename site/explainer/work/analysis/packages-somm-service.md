## Purpose

This package adds somm’s local service tier: a web dashboard, telemetry APIs, an Anthropic-compatible proxy, and background intelligence workers. It keeps routing, budget enforcement, telemetry, evaluation, and recommendations around the same local SQLite repository (`packages/somm-service/README.md:3`).

## How it works

`create_app()` opens the configured repository, creates or loads a service token, registers dashboard, recommendation, OTLP-ingest, and proxy routes, then applies local-security middleware (`packages/somm-service/src/somm_service/app.py:1285`). OTLP JSON spans are bounded, normalized into `Call` records, and written without partially ingesting an over-limit batch (`packages/somm-service/src/somm_service/app.py:1204`). `/v1/messages` validates and translates Anthropic requests, resolves a workload, enforces its budget before dispatch, calls LiteLLM in a thread pool, and records success or failure in the same call ledger (`packages/somm-service/src/somm_service/proxy.py:269`).

`run_server()` starts Uvicorn plus a daemon scheduler (`packages/somm-service/src/somm_service/app.py:1328`). The scheduler stores jobs and leases in SQLite; defaults are daily model-intel, 15-minute shadow evaluation, and weekly agent analysis, with rescheduling on success and bounded backoff on failure (`packages/somm-service/src/somm_service/workers/_runner.py:32`). Model intelligence merges static pricing, OpenRouter metadata, and local Ollama inventory; optional Hugging Face enrichment adds modality metadata (`packages/somm-service/src/somm_service/workers/model_intel.py:86`, `packages/somm-service/src/somm_service/workers/hf_intel.py:124`). Shadow evaluation grades captured samples against a configured gold model and writes eval receipts; the agent then converts calls, evals, provider health, and model metadata into deduplicated recommendations or explicitly opted-in parameter overrides (`packages/somm-service/src/somm_service/workers/shadow_eval.py:102`, `packages/somm-service/src/somm_service/workers/agent.py:70`).

## Key surfaces

- `somm-serve` / `main()` — starts the service or runs intel, shadow, and agent admin commands (`packages/somm-service/src/somm_service/cli.py:161`).
- `create_app()` — constructs the embeddable Starlette application (`packages/somm-service/src/somm_service/app.py:1285`).
- `run_server()` — runs the web service and scheduler lifecycle (`packages/somm-service/src/somm_service/app.py:1328`).
- `messages_endpoint()` — budget-gated Anthropic Messages proxy (`packages/somm-service/src/somm_service/proxy.py:269`).
- `start_inprocess_scheduler()` — runs the intelligence loop without the web stack (`packages/somm-service/src/somm_service/inprocess.py:44`).
- `Scheduler` — lease-based persistent background job runner (`packages/somm-service/src/somm_service/workers/_runner.py:40`).
- `ModelIntelWorker`, `ShadowEvalWorker`, `AgentWorker` — refresh, evaluation, and recommendation stages (`packages/somm-service/src/somm_service/workers/__init__.py:8`).

## Design decisions

- Budgets are checked before provider dispatch; rejected calls create neither spend nor telemetry records (`packages/somm-service/src/somm_service/proxy.py:322`).
- Request bodies are bounded both through `Content-Length` and incremental streaming, covering chunked requests (`packages/somm-service/src/somm_service/http_limits.py:15`).
- Local access uses a generated file token plus a tightly constrained same-origin localhost path designed to resist DNS rebinding (`packages/somm-service/src/somm_service/app.py:91`, `packages/somm-service/src/somm_service/app.py:265`).
- Shadow evaluation is workload-opt-in, excludes private workloads, requires captured bodies, and uses recoverable leases so crashes or transient gold-model failures do not orphan samples (`packages/somm-service/src/somm_service/workers/shadow_eval.py:102`, `packages/somm-service/src/somm_service/workers/shadow_eval.py:262`).
- Self-healing remains recommendation-only unless the workload explicitly sets `auto_heal: true` (`packages/somm-service/src/somm_service/workers/agent.py:84`).

## One-liner

`somm-service` is the localhost control plane that turns locally recorded LLM traffic into budgeted routing, operational visibility, quality evidence, and cautious recommendations.
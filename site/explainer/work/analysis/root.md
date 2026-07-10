## Purpose

This repository-level area defines somm’s public product contract, developer workflow, security posture, packaging, and release lifecycle. It presents a local-first system that routes LLM calls, records telemetry, evaluates outcomes, and turns accumulated evidence into model recommendations without a hosted control plane.

## How it works

Applications create `somm.llm(project=...)` and submit named workloads through `generate`; each call is routed through eligible providers and recorded in a local `.somm/calls.sqlite` database with provider, model, token, cost, latency, and outcome data (`README.md:67`, `roster.toml:9`). Workload names unify policies, budgets, evaluation, and recommendations (`README.md:92`). Optional workers refresh model intelligence, shadow-grade opted-in production samples, and produce recommendations; MCP tools expose the resulting telemetry and cross-project decision memory to coding agents (`README.md:102`, `README.md:244`).

The uv workspace separates schema/storage, the client and router, service workers and web UI, MCP, LangChain integration, and agent onboarding into six packages (`README.md:528`, `pyproject.toml:7`). Tests run across packages and top-level tests, while an autouse fixture redirects machine-wide registry, plan, and global-database state into temporary paths and disables mirroring (`pyproject.toml:32`, `conftest.py:13`).

Changes require tests, an Unreleased changelog entry, and full lint/test checks (`CONTRIBUTING.md:28`). Releases move all package versions together, refresh bundled pricing and plan data, update agent-skill documentation, and require an explicit readiness file before any 1.x publication (`RELEASING.md:8`, `RELEASING.md:28`, `RELEASING.md:94`).

## Key surfaces

`somm.llm(...).generate(...)` — route a named workload and record canonical call telemetry (`roster.toml:9`).

`SommLLM.stream()` / `.embed()` — streamed generation and embedding calls with telemetry (`README.md:210`).

`somm` CLI — generation, status, workloads, prompts, evals, campaigns, plans, diagnostics, and service startup (`README.md:330`).

`somm-mcp` — 14 stdio tools for telemetry, recommendations, comparisons, replay, evaluation, and decision memory (`README.md:300`).

`somm.sommelier:advise` / `somm_advise` — rank provider/model candidates against constraints and accumulated evidence (`roster.toml:25`).

`somm-serve` and its local HTTP API — dashboard, scheduled workers, authenticated reads, and OTLP ingest (`README.md:362`).

## Design decisions

- SQLite is the default user-owned store; prompt bodies require opt-in, private workloads cannot egress, and storage permissions are restricted (`README.md:153`, `SECURITY.md:41`).
- Failures are deliberately visible and bounded in telemetry rather than silently swallowed (`README.md:162`).
- Intelligence signals remain distinct instead of being collapsed into an opaque composite quality score; paid sources stay off the default path (`ROADMAP.md:237`).
- Test isolation protects real fleet-wide state because cross-project mirroring can otherwise touch `~/.somm/global.sqlite` (`conftest.py:18`).
- The root metadata is currently inconsistent: the workspace and README say `0.12.0` beta, while `roster.toml` still says `1.0.0`; the changelog identifies 1.0.0 as mistakenly published (`pyproject.toml:3`, `roster.toml:5`, `CHANGELOG.md:14`). The README also documents cross-project mirroring as default-off while the changelog says it now defaults on (`README.md:518`, `CHANGELOG.md:206`).

## One-liner

This area is somm’s top-level contract for a local, privacy-first LLM routing and telemetry loop that learns from real workloads and exposes that intelligence to developers and agents.
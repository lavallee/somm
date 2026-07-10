## Purpose

This package exposes Somm’s local telemetry, recommendations, model intelligence, and decision history to MCP-capable coding agents over stdio. It lets agents inspect LLM usage, manage workloads and prompts, compare or replay model calls, and reuse routing decisions without adding a commercial service to the hot path.

## How it works

The `somm-mcp` command loads project configuration, constructs the full configured provider chain, builds a `FastMCP` server, and starts its stdio transport (`packages/somm-mcp/src/somm_mcp/cli.py:27`). `build_server` opens the project’s SQLite-backed `Repository`, indexes supplied providers by name, and registers closures as MCP tools (`packages/somm-mcp/src/somm_mcp/server.py:59`). Most handlers delegate persistence and domain logic to `somm-core` or `somm`, while query helpers aggregate calls, recommendations, and shadow-evaluation results.

The intelligence loop combines open recommendations with model rankings from shadow evaluations; when no evaluation data exists, it falls back to model-intelligence candidates and prior decisions (`packages/somm-mcp/src/somm_mcp/server.py:117`). Decisions are written locally and best-effort mirrored to a global cross-project repository (`packages/somm-mcp/src/somm_mcp/server.py:360`). Compare and replay instantiate `SommLLM` with explicit providers, record normal telemetry, and always close the client; replay additionally requires captured samples and refuses private workloads (`packages/somm-mcp/src/somm_mcp/server.py:663`).

## Key surfaces

- `somm-mcp [--project PROJECT]` — launch the stdio MCP server (`packages/somm-mcp/src/somm_mcp/cli.py:12`).
- `build_server(config, providers)` — construct the server and register its tools (`packages/somm-mcp/src/somm_mcp/server.py:59`).
- `somm_stats` / `somm_search_calls` — summarize or filter recorded calls (`packages/somm-mcp/src/somm_mcp/server.py:72`).
- `somm_recommend` / `somm_inbox` — inspect workload guidance and recommendation items (`packages/somm-mcp/src/somm_mcp/server.py:117`).
- `somm_apply_recommendation` / `somm_dismiss_recommendation` — consume recommendation lifecycle actions (`packages/somm-mcp/src/somm_mcp/server.py:197`).
- `somm_advise` — rank models under capability, provider, price, context, and modality constraints (`packages/somm-mcp/src/somm_mcp/server.py:232`).
- `somm_record_decision` / `somm_search_decisions` — persist and recall routing judgments (`packages/somm-mcp/src/somm_mcp/server.py:310`).
- `somm_register_workload` / `somm_register_prompt` — define workloads and version prompts (`packages/somm-mcp/src/somm_mcp/server.py:436`).
- `somm_eval_promote_call` — promote a sampled call into an evaluation dataset (`packages/somm-mcp/src/somm_mcp/server.py:509`).
- `somm_compare` / `somm_replay` — run explicit side-by-side calls or rerun captured calls (`packages/somm-mcp/src/somm_mcp/server.py:549`).

## Design decisions

- Provider-dependent tools remain discoverable without providers and return structured errors, keeping the MCP catalog stable (`packages/somm-mcp/src/somm_mcp/server.py:28`).
- Recorded response bodies are truncated to 4,000 characters and wrapped as untrusted data to reduce prompt-injection risk (`packages/somm-mcp/src/somm_mcp/server.py:49`).
- Compare enforces configurable fan-out and token ceilings, with explicit elevated—but still bounded—caps (`packages/somm-mcp/src/somm_mcp/server.py:577`).
- The README advertises 10 tools, while the current server registers 14; this appears to be documentation drift (`packages/somm-mcp/README.md:7`, `packages/somm-mcp/src/somm_mcp/server.py:1`).

## One-liner

`somm-mcp` turns Somm’s private local telemetry and accumulated routing knowledge into a stable tool interface for coding agents.
# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
`somm` uses a single unified version across all workspace packages
(`somm`, `somm-core`, `somm-service`, `somm-mcp`, `somm-skill`).

## [Unreleased / 0.1.0-dev0]

Initial implementation milestones (development; not yet published).

### Added — D1 — skeleton end-to-end
- uv workspace with 5 packages.
- Schema v1: workloads, prompts, calls, call_updates, samples, model_intel,
  eval_results, recommendations, provider_health, jobs, worker_heartbeat.
- `somm-core`: schema runner, typed models, SQLite repository, parse helpers
  (markdown fences, brace extraction, qwen2.5 double-quote, `<think>`-block
  stripping, content-addressed hashing).
- `somm`: `SommLLM.generate(prompt, workload, …)`, ollama provider, per-process
  writer queue with JSONL spool fallback on SQLITE_BUSY/disk-full.
- `somm-service`: starlette web admin on `localhost:7878` + `/health` +
  `/api/stats` + `/api/version`. XSS-safe rendering.
- `somm-mcp`: stdio MCP server with `somm_stats` tool.
- `somm-skill`: canonical SKILL.md for coding agents.

### Added — D2 — library breadth
- **Routing**: `ProviderHealthTracker` (SQLite-backed cooldowns, per-
  `(provider, model)` entries), `Router` (preference order + circuit
  breaker + bounded exhausted-sleep).
- **Provider adapters**: ollama, openrouter (free roster + cooldowns),
  minimax, anthropic (Messages API), openai (+ OpenAI-compat base for
  any Groq/Together/Fireworks/vLLM/LM Studio/custom gateway).
- **Streaming**: `SommLLM.stream()` with `<think>`-block buffered strip
  across arbitrary chunk boundaries. Native ollama streaming + SSE for
  OpenAI-compatible providers.
- **Prompt versioning**: content-addressed `register_prompt` with
  minor/major/explicit bump; `get_prompt(workload, version="latest")`.
- **`extract_structured`**: returns `dict | list` or
  `{"raw": text, "_somm_parse_err": True}`.
- **`provenance(result)`**: stable schema-versioned dict for stamping on
  output data rows.
- **`parallel_slots(n)`**: striped worker assignment across providers
  (renamed from `probe_providers` per DX review).
- **Error taxonomy**: `SommTransientError`, `SommRateLimited`,
  `SommAuthError`, `SommProvidersExhausted`, `SommStrictMode`,
  `SommPrivacyViolation`.

### Added — D3 — workers + web + CLI
- **Schema v2**: `shadow_config_json` column on workloads +
  `shadow_candidates` view (filters out private + already-graded).
- **ModelIntelWorker**: scrapes OpenRouter `/v1/models`, probes ollama
  `/api/tags`, seeds static pricing for anthropic/openai/minimax.
  Cost calculation from `model_intel` lands on every `.generate()` call.
- **ShadowEvalWorker**: opt-in per workload. Structural (JSON overlap) +
  text-similarity (bigram Jaccard) graders. Budget-capped per-workload.
  Privacy-gated at the view AND in Python.
- **AgentWorker**: emits `switch_model`, `new_model_landed`,
  `chronic_cooldown` recommendations with evidence; deduplicates against
  open recs.
- **Scheduler**: polls `jobs` table; atomic lease via `UPDATE…WHERE`;
  crash-safe; default jobs (model_intel 24h, shadow_eval 15min, agent
  7d). Daemon thread started by `somm serve`.
- **Web admin**: recommendations above charts. `GET/POST
  /api/recommendations`, dismiss/apply endpoints. Full XSS escaping.
- **CLI**: `somm tail --workload …`, `somm compare <prompt> --models
  p/m,p/m`, `somm doctor` (intel freshness + worker heartbeats +
  cooldowns). `somm-serve admin refresh-intel / list-intel / run-agent
  / run-shadow`.

### Added — D4 — MCP breadth
- MCP expanded from 1 tool to 7: `somm_stats`, `somm_search_calls`,
  `somm_recommend`, `somm_register_workload`, `somm_register_prompt`,
  `somm_compare`, `somm_replay`.
- `somm-mcp` CLI now builds the full provider chain from config.
- `somm_replay` enforces `SOMM_PRIVACY_VIOLATION` for
  privacy_class=private + errors clearly on missing-sample.

### Added — D5 — compat shims + examples
- `somm.compat.GenericLLMCompat`: drop-in for codebases with a
  `.generate(prompt, system, max_tokens, provider) -> LLMResult` shape.
- `somm.compat.openai_chat_completions`: OpenAI-SDK-compatible
  function. `provider/model` prefix picks a provider.
- `examples/`: `drop_in_wrapper.py`, `openai_swap_in.py`,
  `private_workload.py`, + README.

### Added — D6 — cross-project mirror
- Opt-in via `SOMM_CROSS_PROJECT=1` or
  `Config.cross_project_enabled=True`.
- WriterQueue post-batch mirrors calls to `~/.somm/global.sqlite`
  (configurable). Workload registrations also replicate.
- Mirror failures are isolated — primary writes never blocked.
- `somm status --global` reads the mirror with per-project stacked
  rollups.

### Added — D7 — OSS prep
- README.md, CHANGELOG.md (this file).
- `docs/errors/`: canonical pages for `SOMM_WORKLOAD_UNREGISTERED`,
  `SOMM_PORT_BUSY`, `SOMM_PROVIDERS_EXHAUSTED`, `SOMM_SCHEMA_STALE`,
  `SOMM_PRIVACY_VIOLATION`.
- `tests/test_blocklist.py` — CI guard against accidental internal-
  name leaks.

### Notes

- Privacy posture is PLAN-defined and test-enforced. No beacon
  telemetry; no prompt/response capture by default; file perms 0600/0700.
- Everything works offline with just ollama. Commercial providers are
  all opt-in.
- Not yet published to PyPI. License: TBD.

# somm — TODOs (deferred scope)

_Items deferred during autoplan review to keep v0.1 focused. Tracked for post-v0.1 consideration._

## Tool-calling — in flight (2026-05-19)

Anthropic + OpenAI-compat shipped tonight. See [docs/tool-calling.md](./docs/tool-calling.md) for the spec; CHANGELOG `[Unreleased]` for the shipped scope.

Remaining work, ordered by impact for the Starboard driving project:

1. **`GeminiProvider` tool support.** Gemini uses `functionDeclarations` nested under `tools`, and `tool_config.function_calling_config.mode` for tool_choice ("AUTO"/"ANY"/"NONE"). Response carries `functionCall` blocks (singular) in `candidates[0].content.parts[]`. Tool results in subsequent turns use `functionResponse` blocks. Different enough from Anthropic that translation lives entirely in `gemini.py`, not in a shared helper. Tests follow the same `httpx.MockTransport` pattern as tonight's commits.

2. **`OllamaProvider` tool support.** Ollama 0.4+ exposes a `tools` field on `/api/chat`. Check the version actually installed on the box (the daily-driver model server) — some 0.3.x installs in the wild. If 0.4+: payload key is `tools`, response carries `message.tool_calls[]` with `function.name` + `function.arguments` (already a dict, not a JSON string — different from OpenAI). Tool_choice not yet supported by Ollama as of the last check; either silently drop or raise. **Decision needed.**

3. **`model_intel` seeding pass for the `tools` capability.** Tonight's design doc plans `capabilities_required=["tools"]` filtering, but no seeded rows declare `tools` yet. Add a small one-off in `somm_core.pricing.seed_known_pricing` (or a new `seed_known_capabilities`) for the obvious tool-capable models: Claude 3+, GPT-4+, GPT-5+, Gemini 1.5+, Llama 3.1+, Qwen 2.5+, DeepSeek-Chat. Without this, capability-filtered workloads route to nobody.

4. **Schema 0008 — telemetry columns.** Deferred from tonight by design. Once Starboard accumulates real tool-call workload calls, decide whether to lift `tool_calls_count`, `tools_offered_count`, `stop_reason` out of `raw_json` into dedicated columns for index-able queries. Don't migrate preemptively — wait for the query pattern.

5. **Streaming tool calls.** Out of scope tonight. Anthropic streams `input_json_delta` events per tool_use block; OpenAI streams `tool_calls[].function.arguments` chunks via SSE deltas. Reassembly is non-trivial and matters mainly for low-latency UX, which agent loops don't typically have. Open issue when a project asks.

6. **Defensive raise on tools-unsupported providers.** Currently if a caller passes `tools=[…]` to a provider that hasn't yet implemented it (Gemini, Ollama as of tonight), the tools are silently dropped — the model responds with text and `stop_reason="end_turn"` rather than `"tool_use"`. The signal exists but is implicit. Adding `raise SommBadRequest("…doesn't support tools yet")` in those providers would make this loud. Easy to add; just decide whether the lift-to-loud is worth the brittle behavior during the provider rollout.

7. **Prompt-hash for `messages`.** `SommLLM.generate()` currently hashes only `prompt`. When `messages=` is used, the hash is taken of an unused string ("ignored" or whatever the caller passed for `prompt`). For replay/cache/dedup, hash `messages` when present. Small surgery in client.py around `stable_hash(prompt)`.

## Deferred (in priority order)

### Core product
- **A/B routing** — agent recommendations become live shadow traffic splits with lift calculation. Currently agent only recommends; no closed loop. (~2–3d CC.)
- **`somm.ensemble(prompt, models=[…], aggregate=fn)`** — parallel-model call primitive for ensembling. (~2–3d CC.)
- **Auto-eval generation from production samples** — frontier writes grading rubrics from sampled call pairs; builds eval suites automatically. (~2d CC.)

### Infrastructure
- **Postgres backend** for small-team shared deployments as an optional `somm[postgres]` extra. SQLite remains default. (Phase 3 codex.)
- **Windows service lifecycle** support (Task Scheduler integration). Linux + macOS day-one. (Phase 3.)
- **HF trending model-intel source** — behind feature flag; OpenRouter is primary source. Fragile DOM scraping. (Phase 3 H1.)
- **Release-feed model-intel sources** (RSS/Atom per-provider) — most are dead; feature-flagged. (Phase 3 H1.)
- **Provider-specific tokenizers** as `somm[tokenizers]` extras (tiktoken, etc.). Default approximation (4 chars/token) ships in v0.1. (Phase 3 H2.)
- **Broader model-intel signal sources for sommelier ranking** — see [`docs/intel-sources.md`](./docs/intel-sources.md). Candidates: LMArena Elo (quality), Artificial Analysis (price/speed/quality), canirun.ai (local GPU feasibility), LiveBench (contamination-resistant), Open LLM Leaderboard (per-benchmark). Each needs: stable source URL, refresh cadence, failure mode when the source is down. Likely feature-flagged per-source like the HF scraper.

### DX
- **Beacon telemetry** for DX measurement — v0.1 is local-only `somm admin dx-report`; beacon (opt-in) deferred. (Phase 3.5.)
- **`somm plugin` command** — install/list/remove plugins (providers, graders, etc.) with supply-chain checks. Currently pip-based. (Phase 3.5 extension.)
- **GUI installer / macOS .dmg / Linux .deb** — v0.1 ships pipx/uv tool install; packaged installers post-v0.1.

### Design
- **Recommendation evidence detail panel design review** — v0.1 spec says inline in card; deep drawer/modal design deferred.
- **Dashboard filtering/search** — v0.1 has per-project toggle + time window dropdown; richer filtering deferred.
- **Dark-mode polish** — tokens.css has `prefers-color-scheme` media query; light-mode a second-class citizen for v0.1.

## Principles for pulling items back into scope

- If a deferred item becomes load-bearing for a post-v0.1 user demand, promote.
- If a deferred item can be shipped as an optional extra (`somm[X]`) without bloating core, that path is preferred over blocking v0.1.
- If a deferred item would be a days-of-work surprise to someone trying to build it themselves (plugin protocol, extensibility), promote to earlier.

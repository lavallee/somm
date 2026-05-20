# somm — TODOs (deferred scope)

_Items deferred during autoplan review to keep v0.1 focused. Tracked for post-v0.1 consideration._

## Tool-calling — provider sweep complete (2026-05-19)

Anthropic + OpenAI-compat shipped first; Gemini, Ollama, and capability seeding followed. Every shipping provider now supports tools. See [docs/tool-calling.md](./docs/tool-calling.md) for the spec; CHANGELOG `[Unreleased]` for the shipped scope.

Shipped:

1. ~~**`GeminiProvider` tool support.**~~ **Done.** Resolved via the inherited OAI-compat path, not a native `functionDeclarations` adapter: `GeminiProvider` extends `OpenAICompatProvider` and Google's OAI-compat endpoint accepts standard OpenAI `tools`/`tool_choice` and returns `tool_calls` in OpenAI shape. Stale "use native endpoint for function calling" note in `gemini.py` corrected; passthrough test added. (If a future need surfaces that the OAI-compat endpoint can't serve — multimodal tool use beyond image input, or generateContent-only features — revisit a native adapter then.)

2. ~~**`OllamaProvider` tool support.**~~ **Done.** Box runs Ollama 0.20.3 (well past the 0.4 floor). `tools` sent in OpenAI shape on `/api/chat`; multi-turn `messages` reuse the shared OpenAI translator; `function.arguments` parsed as a dict (no `json.loads`). `tool_choice` raises `SommBadRequest` (Ollama has no such knob) — decided in favor of loud-fail over silent-drop, per the spec's no-silent-drop rule.

3. ~~**`model_intel` seeding pass for the `tools` capability.**~~ **Done.** `model_has_capability` gained a name-hint `tools` branch (Claude 3+, GPT-4+/5+, Gemini 1.5+, Llama 3.1+, Qwen 2.5+, DeepSeek-Chat, Mistral/Mixtral, o-series); seeded frontier rows declare `{"tools": true}`. Unknown models fall through as capable (None = allow) — no negative case, since unsupported providers raise at call time.

6. ~~**Defensive raise on tools-unsupported providers.**~~ **Done (subsumed).** With Gemini and Ollama landed, no shipping provider silently drops a `tools=` request. Ollama's `tool_choice` raise is the concrete instance of the loud-fail behavior. The router treats `SommBadRequest` as a permanent (provider, model) error and falls through.

7. ~~**Prompt-hash for `messages`.**~~ **Done.** `SommLLM.generate()` now hashes `messages` when present (`stable_hash(messages if messages is not None else prompt)`), so replay/cache/dedup keys off the real conversation rather than the ignored `prompt` placeholder.

Still deferred by design:

4. **Schema 0008 — telemetry columns.** Once Starboard accumulates real tool-call workload calls, decide whether to lift `tool_calls_count`, `tools_offered_count`, `stop_reason` out of `raw_json` into dedicated columns for index-able queries. Don't migrate preemptively — wait for the query pattern.

5. **Streaming tool calls.** Anthropic streams `input_json_delta` events per tool_use block; OpenAI streams `tool_calls[].function.arguments` chunks via SSE deltas. Reassembly is non-trivial and matters mainly for low-latency UX, which agent loops don't typically have. Open issue when a project asks.

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

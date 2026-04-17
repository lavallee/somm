# somm — Plan

_Execution doc: what we build, how the pieces cohere, what ships on day one. Captures CEO/Design/Eng/DX review findings with a decision audit trail._

## Vision

**somm is the intelligence loop for LLM workloads in code.** Not a library that grows into an ecosystem — an ecosystem where the library is one organ.

```
   library (sensor) ──► local store ◄── service (brain)
       ▲                    ▲                  │
       │                    │                  ├─► shadow-eval worker
       │                    │                  ├─► model-intel worker
       │                    │                  ├─► agent (improvement engine)
       │                    │                  └─► web admin
       │                    │
       └──── skill (onboarding) ─── MCP (interface for coding agents)
```

Every piece needs every other piece for any piece to matter:

- A **library** without the MCP is just another logger.
- An **MCP** without the brain has no intel to recommend.
- A **brain** without the agent stagnates into a passive archive.
- An **agent** without the skill never gets invoked.
- A **skill** without the library has nothing to onboard people to.

Build all of them from day one. Thin, but real.

## Architectural shape

**Monorepo, all packages active in v0.1.** Python library is the only pip-installable piece; the rest live alongside it.

```
somm/
├── PLAN.md  README.md
├── pyproject.toml                  # uv workspace root
├── packages/
│   ├── somm/                       # Python library — the sensor
│   │   └── src/somm/
│   │       ├── client.py           #  SommLLM
│   │       ├── providers/          #  ollama, openrouter, minimax, anthropic, openai
│   │       ├── routing.py          #  roster cycling, cooldowns, circuit breakers
│   │       ├── telemetry.py        #  SQLite writer (WAL), sampling
│   │       ├── prompts.py          #  versioned prompt objects
│   │       ├── workloads.py        #  first-class workload registry
│   │       ├── provenance.py       #  stable stamp for output rows
│   │       └── cli.py              #  somm {status,tail,doctor,serve,compare}
│   ├── somm-mcp/                   # MCP server — interface for coding agents
│   ├── somm-skill/                 # Claude skill — onboarding
│   │   └── SKILL.md
│   ├── somm-service/               # Brain: web + scheduled workers
│   │   └── src/somm_service/
│   │       ├── app.py              #  FastAPI / starlette; serves web + MCP HTTP
│   │       ├── workers/
│   │       │   ├── model_intel.py  #  scrape OpenRouter / HF / release feeds
│   │       │   ├── shadow_eval.py  #  replay sampled calls through gold standard
│   │       │   └── agent.py        #  weekly analysis → recommendations
│   │       ├── web/                #  one HTML page + htmx + three charts
│   │       └── recommend.py        #  the recommendation engine (MCP-callable)
│   └── somm-agent/                 # Thin wrapper around agent worker (CLI invocations)
├── schemas/                        # SQL migrations, JSON schemas for API/MCP
├── examples/                       # Synthetic/anonymized workloads only
└── tests/                          # VCR-style recorded fixtures; no live calls
```

**Write path is zero-dependency**: library writes directly to `./.somm/calls.sqlite` (WAL mode). Nothing needs to be running for the library to function. The brain turns on via `somm serve`, which starts web + workers.

## The intelligence loop — how the pieces actually talk

1. **Library** writes every call (prompt-version, workload, provider, model, tokens, latency, cost, outcome) to local SQLite. Always on. No network required.
2. **Shadow-eval worker** (inside the service) periodically picks recent un-graded calls, re-runs a configured fraction through a gold-standard frontier model (e.g. `claude-opus-4-7`), stores the delta (structural + judge-based grade). **This is how we build a real-world eval dataset without anyone writing evals.**
3. **Model-intel worker** scrapes OpenRouter pricing + available models, HF trending, release feeds. Cached in `model_intel` table. **This is how the MCP returns *today's* best candidates, not outdated training data.**
4. **Agent worker** runs weekly (or on demand): reads calls + shadow-eval + model-intel, emits `recommendations` rows with evidence and expected impact. Notifies via configurable sink (file, webhook, email).
5. **MCP** exposes tools to the coding agent: `somm_recommend(workload)`, `somm_compare(prompt, models)`, `somm_replay(call_id, with_model)`, `somm_search_calls`, `somm_register_workload`, `somm_register_prompt`, `somm_stats`. Reads local store + service recommendations.
6. **Skill** ensures the coding agent reaches for somm instead of rolling LLM code, uses the MCP during workload design, and stamps provenance on outputs.
7. **Web admin** reads the same SQLite + recommendations. One page, three charts, one list.
8. **Cross-project knowledge commons**: per-project `./.somm/calls.sqlite` + mirror into `~/.somm/global.sqlite` (post-hook, opt-in). When Project A learns `gemma-3-27b` beats `llama-3.3-70b` on JSON extraction, Project B's agent sees that signal the next run. The hosted service (post-v0.1) is just a further mirror.

## v0.1 scope — all six pieces, day one

### Library (`somm/`)

- `somm.llm(project=…)` factory; lazy single-instance per process.
- `.generate(prompt, system, workload, max_tokens, prompt_version="latest", judge=None)` → `SommResult`.
- `.extract_structured(…)` with markdown-fence + brace-extraction + qwen2.5 double-quote fix.
- `.stream(…)` — streaming from day one. Lots of real workloads want it.
- `somm.workload("name")` context manager + `somm.register_workload(name, description, schema)`.
- `somm.prompt(workload, version="latest"|"vN")` — prompt objects are first-class; hashed, stored, immutable once committed.
- `somm.provenance(result)` → stable dict for stamping on output rows.
- `somm.probe(n_slots)` → parallel-worker slot assignment.
- `result.mark("ok"|"bad_json"|…)` — outcome hook for lightweight quality signal without a judge.
- Optional `judge=callable` param on `.generate` for auto-grading on the hot path.
- Providers: ollama, openrouter (free roster + cooldowns), minimax, anthropic, openai. Adapter strips provider-specific quirks (`<think>…</think>`, double-quoted JSON) before telemetry so token counts reflect real output.

### MCP (`somm-mcp/`)

FastMCP or stdio-based MCP server exposing seven tools:

- `somm_recommend(workload_spec)` → top 3 candidates with rationale from model-intel + shadow-eval history.
- `somm_compare(prompt, models=[…], n_runs=1)` → runs prompt through each model, returns side-by-side.
- `somm_replay(call_id, with_model)` → replays a stored call against a different model, returns delta.
- `somm_search_calls(filters)` → query the call store (workload, date range, outcome, cost band).
- `somm_register_workload(name, description, input_schema, output_schema)` → commits a workload.
- `somm_register_prompt(workload, body, bump="major"|"minor")` → commits a new prompt version.
- `somm_stats(workload=None, since="7d")` → roll-up for dashboards.

### Skill (`somm-skill/SKILL.md`)

Markdown skill that activates when the coding agent is about to write or modify LLM code. Directs the agent to:

1. Use `somm.llm(...)` instead of raw provider SDKs.
2. Register the workload via MCP before first call.
3. Commit prompt as a versioned object, not inline string.
4. Call `somm_recommend` during model selection rather than guessing from training-era knowledge.
5. Run `somm_compare` for any workload where cost/quality might be non-obvious.
6. Stamp `somm.provenance(result)` on output rows stored in the project's DB.

### Service (`somm-service/`)

- `somm serve` starts a process with:
  - HTTP server (FastAPI or starlette) on `localhost:7878`.
  - Web admin: one HTML page + htmx, three charts (cost over time stacked by workload, provider-health heatmap, quality trend per workload) + recommendations list.
  - Scheduled workers (APScheduler or simple asyncio loops):
    - `model_intel` every 24h.
    - `shadow_eval` continuous — sample rate configurable, default 2%.
    - `agent` weekly (and on-demand via CLI).
- Runs against both per-project `./.somm/calls.sqlite` and `~/.somm/global.sqlite`.
- HTTP endpoints: `/recommend`, `/compare`, `/replay`, `/stats`, `/health`. MCP tools proxy these; web UI calls them too.

### Agent (`somm-agent/`)

v0.1 job: **"Your top 3 optimization opportunities this week."**

Inputs: last N days of calls, shadow-eval grades, model-intel diffs, cost totals. Outputs: `recommendations` rows with schema `(id, workload, action, evidence, expected_impact, confidence, created_at, dismissed_at)`.

Actions it can recommend:
- `switch_model(workload, from, to)` — based on shadow-eval quality on the workload + cost/latency.
- `revert_prompt(workload, from_version, to_version)` — if quality regressed after a prompt change.
- `add_fallback(workload, provider)` — if the current chain is hitting cooldowns often.
- `tune_context(workload, suggestion)` — if prompts are token-heavy with low information density (embedding/compression heuristic).
- `new_model_landed(model)` — model intel caught a cheaper-or-better candidate.

Notification sinks: stdout (default), file, webhook URL, SMTP (email). Config in `[tool.somm.agent]`.

### Shadow-eval worker

- Configurable gold-standard model(s): `shadow.gold_standard = ["claude-opus-4-7"]`.
- Sample rate: default 2% of calls per workload, max 50/day/workload to cap cost.
- Graders (composable):
  - Structural: JSON-shape match, field presence.
  - Embedding similarity (via a small local model).
  - Judge-model grade (cheap frontier, e.g. haiku): "rate 1–5 how well the production response answers the prompt vs. the gold one."
- Stores: `eval_results (call_id, gold_model, gold_response, structural_score, embedding_score, judge_score, judge_reason, ts)`.

### Model-intel worker

- Sources: OpenRouter models API (prices, context windows, availability), HF trending (optional), model release feeds / blog RSS where available.
- Cache: `model_intel (provider, model, price_in_per_1m, price_out_per_1m, context_window, capabilities_json, last_seen, source)` with TTL.
- Cost model for the library is fed from this table — prices aren't hand-maintained after day three; they refresh nightly.

## Data model (SQLite, WAL mode)

Per-project `./.somm/calls.sqlite`; mirrored into `~/.somm/global.sqlite` via opt-in post-hook.

```sql
-- first-class entities
workloads        (id, name, project, description, input_schema_json, output_schema_json, created_at)
prompts          (id, workload_id, version, hash, body, created_at, retired_at)
model_intel      (provider, model, price_in, price_out, context_window, capabilities_json, last_seen, source)

-- events
calls            (id, ts, project, workload_id, prompt_id, provider, model,
                  tokens_in, tokens_out, latency_ms, cost_usd, outcome, error_kind,
                  prompt_hash, response_hash)
samples          (call_id, prompt_body, response_body)            -- sampled fraction only

-- intelligence
eval_results     (id, call_id, gold_model, gold_response_hash,
                  structural_score, embedding_score, judge_score, judge_reason, ts)
recommendations  (id, workload_id, action, evidence_json, expected_impact, confidence,
                  created_at, dismissed_at, applied_at)

-- routing
provider_health  (provider, model, last_ok_at, cooldown_until, consecutive_failures)
schema_version   (version, applied_at)
```

**Why SQLite, not DuckDB:** WAL mode gives concurrent readers + one writer across processes — the exact shape of our load (library writes continuously, web admin + agent + MCP read). DuckDB's single-writer-per-file lock breaks this. Analytical queries can use DuckDB's `ATTACH … (TYPE SQLITE)` read-only when we later want column-engine rollups; the write path stays SQLite.

## First graft — Project A as the day-one full-stack user

Not "the library drops in and everything else comes later." Project A is the first project that gets **the whole loop**:

1. `uv add somm somm-service somm-mcp` + install `somm-skill` at project level.
2. Replace `the target's existing LLM wrapper` with `somm.compat.GenericLLMCompat` (drop-in). Every existing call site unchanged.
3. Register workloads: `claim_extract`, `topic_gate`, `relevance_score`, `query_generate`, `enrich`, `artifact_mine`.
4. Commit the current inline prompts as `v1` prompt objects for each workload.
5. Run `somm serve` in a background terminal.
6. Run a scouting tick.
7. Open `localhost:7878` — charts are already populating.
8. After 24h: agent runs, posts its first "top 3 opportunities" note. Expected first rec: *"Switch `claim_extract` from `gemma4:e4b` (local, 8s median) to `gemma-3-27b-free` on OpenRouter — shadow eval grades it 4.1/5 vs. 3.2/5 over 47 calls, with lower latency and no local GPU bottleneck."*

**Acceptance for v0.1**: that demo works. Everything downstream of it is polish.

## Open-source hygiene

- `examples/` uses only synthetic/anonymized workloads and prompts. No nouns that point to our projects or data domains. Canonical examples: `classify_article`, `extract_contacts`, `summarize_doc`, `generate_queries`.
- `tests/` use recorded HTTP fixtures (VCR-style) — no live API calls.
- CI: grep-based blocklist over `packages/`, `tests/`, `examples/`, `docs/`. Blocklist file itself is `.gitignored`; only the check ships. Fails the build if any of our project names / internal domain terms leak.
- README examples are synthetic (news-classifier, invoice-extractor), not transplanted from our work.
- No DB dumps, no config files with internal URLs, no model aliases carrying our project names.

## Build sequence — how we ship six coherent pieces without vaporware

Not "v0.1 → v0.5 staircase" but a **week-long vertical-slice sprint**: we stand up every piece crudely end-to-end first, then thicken.

- **D1 — skeleton end-to-end.** uv workspace, SQLite schema + migrations, ollama adapter only, `generate()` writes a row, `somm serve` stands up a web page showing "hello: 1 call logged," MCP server exposes one tool (`somm_stats`), skill file in place.
- **D2 — widen the library.** All 5 providers, `extract_structured`, routing (cooldowns, circuit breaker), streaming, prompt/workload registries, provenance helper.
- **D3 — light up the workers.** Shadow-eval worker (2% sample, one structural grader + one judge), model-intel worker (OpenRouter scrape), cost table populated from intel, CLI `somm status` / `tail` / `doctor` / `compare`.
- **D4 — agent and web.** Agent's weekly analysis pass, recommendations table, web page thickened to three charts + recs list, MCP expanded to all seven tools.
- **D5 — graft into Project A.** Compat shim, replace `the target's existing LLM wrapper`, register workloads, run a real tick, fix the gotchas that fall out. This is where the plan meets reality.
- **D6 — cross-project mirror.** `~/.somm/global.sqlite` post-hook, web admin reads from global, agent's recs consider cross-project signal.
- **D7 — polish + open-source prep.** Blocklist CI check, README with synthetic examples, examples folder, PyPI `somm` + `somm-service` + `somm-mcp` publish under a unified version.

## Open questions (called out for the review pass)

1. **Service packaging**: one binary (`somm serve` starts everything) or split (`somm-web`, `somm-agent` as separate processes)? Simpler to start unified; fine-grained control comes if someone needs it.
2. **Gold-standard model for shadow evals**: always `claude-opus-4-7`, or rotate (opus / sonnet / gpt-frontier) for reduced judge bias? Probably rotate, but it roughly doubles shadow cost.
3. **Judge model for grading**: cheap frontier (haiku), or embedding similarity + structural-only (no judge model)? Haiku gives qualitative reasons; embeddings alone are cheaper and more reproducible.
4. **MCP transport**: stdio-only (canonical), or also HTTP for headless/remote-agent cases? HTTP opens the door to the hosted future without extra work.
5. **Prompt-versioning ergonomics**: auto-bump prompt version on body change, or require explicit `bump=`? Auto-bump is friendlier; explicit is safer against noise commits.
6. **How permissive should the library be by default?** Opinion: strict — registered workload + registered prompt required for `.generate()`. Loose/demo mode behind a flag. Forces the discipline that makes telemetry meaningful.
7. **Web admin rendering**: server-side HTML + htmx (simpler, no build step) vs. a real SPA. htmx wins on v0.1; a real SPA if/when this grows into a product surface worth polishing.
8. **Cost of the intelligence loop**: the shadow-eval + judge-model calls are not free. Need a per-project budget ceiling + a config that disables shadow eval for sensitive workloads (privacy, PII).

---

# CEO Review (autoplan Phase 1)

_Mode: SELECTIVE EXPANSION. Generated 2026-04-17 via `/autoplan`. Full codex transcript at `~/.gstack/projects/somm/ceo-plans/2026-04-17-codex-ceo-voice.md`._

## Step 0: Nuclear Scope Challenge + Mode Selection

### 0A. Premise Challenge

Premises riding under this plan:

1. **Free-tier / ollama bottlenecks are the primary routing pain.** True for the author's own rig + usage pattern. Uncertain generality — most serious teams pay through a gateway or accept latency.
2. **Shadow-eval against a frontier gold-standard produces a useful quality signal.** Works for structured/classification/extraction workloads. Weak for open-ended generation, summarization, agentic flows, freshness-sensitive tasks, and anywhere the frontier is confidently wrong.
3. **The intelligence loop is strictly more valuable than any piece alone.** Asserted, not shown. Existing platforms (Langfuse, Braintrust, Helicone, Portkey, LiteLLM) bundle observability + prompt mgmt + evals + dashboards + gateway with wildly bigger budgets.
4. **All six packages must ship together day-one for value to materialize.** The weakest strategic claim. Library + SQLite traces + replay/compare CLI delivers value independently. The "no piece matters alone" framing smuggles in a false necessity.
5. **Open-source adoption is plausible outcome.** Unsupported. The plan has no distribution hypothesis, no falsifiable adoption thesis, no named first external users. The skill (Claude-specific) is a distribution dead-end for broad OSS adoption.
6. **The parallel projects are representative pattern sources.** Partially valid. Selection bias — all are news/content/research Python workloads. Cross-project "learning" only works with normalized workload definitions and comparable outputs.

**Hidden premises** (not stated but load-bearing):
- That the user's actual primary pain is visibility (vs. discoverability, vs. cross-project learning — these are not ranked).
- That sampling prompt/response bodies by default (`samples` table) is an acceptable privacy posture. It probably isn't.
- That model-intel scraping of OpenRouter/HF/release feeds is cheap to maintain. Provider APIs change; this will rot.
- That prompt shape, input preprocessing, and downstream validation aren't the dominant optimization levers (they often are).

### 0B. Existing Code Leverage

Sub-problem → existing pattern source:
- Provider routing with model-roster cycling + per-model cooldowns → pattern source 1's `llm_openrouter.py` (mature).
- Manual JSON extraction (markdown fences, brace-extraction, qwen2.5 double-quote fix) → pattern source 1 (battle-tested on real flaky output).
- Per-call metadata collection pattern → pattern source 1's `metadata_collector` list param; pattern source 2's `LLMStats`.
- Telemetry schema shape → pattern source 2's DuckDB evals table; pattern source 1's ORM provenance columns.
- Multi-platform eval runner → pattern source 2's runner.
- Parallel slot assignment → Project A's `the target's existing LLM wrapper.probe_providers()`.
- Env-var-driven provider selection → pattern source 1 + pattern source 2 convention.

Fresh builds (no pattern to steal): MCP server, skill file, service (web + workers), shadow-eval worker, model-intel worker, agent worker, cross-project mirror.

**~40% of the library code is pattern-recovery. ~100% of the non-library code is fresh.** The scope risk concentrates in the 60% that isn't pattern-backed.

### 0C. Dream State Mapping

```
CURRENT STATE               THIS PLAN                     12-MONTH IDEAL
──────────────              ──────────                    ──────────────
Per-project LLM             All projects share a          Shared knowledge graph across
wrappers, no visibility,    sensor + brain + agent        projects. Agents proactively run
ollama bottleneck = manual  + interface. Shadow eval      experiments. "New model landed,
juggling. Provenance = not  builds eval data. Agent       auto-A/B" is ambient. Cross-team
tracked. Frontier agents    sends weekly recs. Graft-     adoption as open source; a
recommend stale models.     ready into one project.       knowledge commons of what works
                                                          for what workload.
```

**Dream state delta:** The plan gets us ~60% of the way there: the library + telemetry + shadow-eval scaffold the commons. Missing from v0.1 that the ideal requires: closed-loop A/B routing, workload ontology, and private federated learning across projects.

### 0C-bis. Implementation Alternatives

**APPROACH A: Monorepo, all six pieces day one (current plan).**
  Summary: uv workspace, library + MCP + skill + service + web + agent all in v0.1.
  Effort: XL (human: 1–2 wks / CC+gstack: 5–7 days).
  Risk: Medium-high — six workstreams, integration drag, any regression hurts the demo.
  Pros: Coherent intelligence loop, validates pieces against each other, forces discipline.
  Cons: Implementation drag before PMF clarity; re-invents surfaces that Langfuse/LiteLLM/Braintrust/Helicone/Portkey already ship; weak moat at ship time.
  Reuses: pattern source 1 routing + pattern source 2 telemetry patterns + Project A compat shim target.

**APPROACH B: Narrow-wedge v0.1 — library + SQLite + CLI + one MCP tool + opt-in shadow.**
  Summary: Library wraps calls → SQLite trace store with workload/prompt provenance. CLI: `somm replay`, `somm compare`, `somm report`. One MCP tool: `somm_recommend`. Shadow-eval only on explicitly opted-in workloads with budget/privacy gates. NO service, NO web admin, NO agent worker, NO model-intel scraper in v0.1.
  Effort: M (human: 3–5 days / CC+gstack: 1–2 days).
  Risk: Low — single shippable artifact, fast iteration cycle.
  Pros: Falsifiable thesis — "turn production calls into evidence-backed model/prompt decisions locally." Exports to Langfuse/Braintrust instead of competing. Project A graft still works. Service/web/agent earned by proving v0.1 useful.
  Cons: Loses the "loop is the differentiator" narrative for now. OSS adoption story is narrower. User's stated "boil-the-ocean" preference not honored literally.
  Reuses: same as A.

**APPROACH C: Hold scope but patch specific blind spots.**
  Summary: Keep six-piece day-one plan, but fix named weaknesses: (i) privacy-first defaults (samples OFF by default; hashes-only), (ii) shadow-eval budget ceiling + break-even gate baked in, (iii) compat adapters for LiteLLM/Langfuse/Braintrust I/O formats, (iv) falsifiable adoption thesis written into README, (v) Claude-skill reframed as one of several agent-onboarding artifacts (Codex, Cursor, Windsurf), (vi) workload ontology as explicit first-class concept.
  Effort: XL+ (adds ~1 day to the XL).
  Risk: Medium — patches address known concerns but don't reduce the six-workstream integration burden.
  Pros: Honors user's scope intent. Addresses the specific critiques both models raised without collapsing to the library-only story.
  Cons: Still dismisses incumbents' presence. Still bets on the integrated loop being worth the burden.
  Reuses: same as A + adapter patterns from LiteLLM/Langfuse SDKs.

**RECOMMENDATION:** Taste decision — defer to user. Approach B is the models' consensus recommendation. Approach C is the "honor user intent but patch blind spots" middle path. Approach A (status quo) held only if user judges the incumbents' surfaces non-competitive with somm's integrated loop.

### 0D. Mode-Specific Analysis (SELECTIVE EXPANSION)

**Complexity check.** Plan touches 5 packages, ~6 workers/surfaces, ~2000 lines of new code. Far above the "8 files / 2 classes" smell threshold. The smell is real; both dual voices flag it.

**Minimum set that achieves the stated goal.** If the goal is "the intelligence loop as described," the minimum is the six pieces. If the goal is "visibility + evidence-backed model/prompt decisions for Project A-class projects," the minimum is library + SQLite + CLI + opt-in shadow (Approach B).

**Expansion scan (candidates):**
1. **Deterministic replay with captured context** — full prompt/system/temp/seed/context stored so any call is re-runnable. Currently partial (hashes only).
2. **Cost/quality Pareto view in web admin** — scatter plot of cost vs. quality per workload per model.
3. **Budget circuit breaker** — per-project/per-workload ceilings with auto-pause. Already noted as open question.
4. **A/B routing** — agent recs become live shadow traffic splits with lift calculation.
5. **Cached prompt optimization** — detect Anthropic/OpenAI cache opportunities for prompts >1k tok.
6. **Privacy classifier** — workloads tagged private → local-only routing; never upstream.
7. **Ensemble workflow primitive** — `somm.ensemble(prompt, models=[...], aggregate=fn)`.
8. **Auto-evals from production samples** — frontier writes grading rubrics from sampled call pairs.

Auto-decisions (per autoplan 6 principles):
- #1 ACCEPT (in blast radius, <1d, schema addition + fidelity upgrade to existing call_id path).
- #2 ACCEPT (web admin already in scope; one more chart is cosmetic).
- #3 ACCEPT (resolves open question #8; touches client + telemetry + config; ~1d CC).
- #4 DEFER (2–3d CC, significant new design). → TODOS.md.
- #5 TASTE DECISION — 3–4 files, ~1d, but provider-specific and easy to get wrong.
- #6 TASTE DECISION — config shape matters a lot for user trust; needs thought.
- #7 DEFER (2–3d CC, new API concept). → TODOS.md.
- #8 DEFER (2d CC, builds on shadow-eval). → TODOS.md.

### 0E. Temporal Interrogation

- **Hour 1 (foundations):** uv workspace setup, SQLite schema, package imports. Ambiguities: Python version target (3.12? 3.11?), inter-package dependency direction, migration strategy for schema changes.
- **Hour 2–3 (core logic):** ollama + openrouter adapters, `generate()` writes a row. Ambiguities: `<think>` stripping uniformity, where call_id emits, cost estimation when model_intel hasn't refreshed yet.
- **Hour 4–5 (integration):** MCP reading SQLite, service starts, skill in place, workers stubbed. Surprises: MCP SDK quirks, FastAPI async vs library sync coexistence, SQLite WAL + worker scheduling interactions.
- **Hour 6+ (polish/tests):** Project A graft, VCR fixtures, README. Will wish they'd planned: provider response schema drift (OpenRouter free roster especially), cost-table staleness, web admin htmx bundling.

With CC+gstack, these compress to ~30–60 minutes per phase.

### 0F. Mode Selection

**Mode:** SELECTIVE EXPANSION (per autoplan).
**Approach:** Deferred to premise gate below — A / B / C is a USER CHALLENGE, not auto-decidable.

---

## Step 0.5: Dual Voices

### CLAUDE SUBAGENT (CEO — strategic independence)

7 findings; consensus ratings DISAGREE / UNCLEAR on all 6 dimensions. Top concerns:
- The "every piece needs every other piece" framing is asserted, not argued. Library + model-intel is load-bearing; agent/web/shadow are speculative.
- Wrong problem framing — visibility vs. discoverability vs. cross-project learning are unranked.
- Shadow-eval economics: $5–$30/day frontier cost to identify $1–$5/day savings at small scale is inverted ROI.
- Most-likely 6-month regret: all-six-pieces collapses to "library + weak dashboard," agent ignored.
- Alternatives dismissed without argument (LiteLLM, Langfuse, Helicone, Braintrust, Portkey).
- No moat against funded incumbent. Shadow-eval can be shipped in a sprint by Langfuse.
- OSS adoption premise unsupported; no distribution hypothesis.

### CODEX (CEO — strategy challenge, gpt-5.4)

Extended review, same directional conclusion. Top concerns that extend the subagent's:
- Free-tier/ollama bottlenecks are the *author's* pain, not the general market's. OpenRouter's own activity export + serious teams paying-through-gateway erodes that wedge.
- Privacy: sampling prompt/response bodies into `samples` is a liability. Default should be hashes-only with per-workload capture opt-in.
- Workload specificity: "JSON extraction" is not one workload. Global DB risks anecdote storage without normalized task definitions.
- Claude-skill as core architecture is fragile — skill conventions across Claude/Codex/Cursor/Windsurf are moving fast. Shouldn't be load-bearing.
- Model-intel scraper is maintenance sludge risk.
- The plan confuses "all pieces are conceptually related" with "all pieces required for initial value."
- No falsifiable adoption thesis. Suggested form: "Within 30 minutes, a Python project with three LLM call sites gets one actionable recommendation that the maintainer accepts."

**Hard recommendation (codex):** Narrow v0.1. Library + SQLite trace store + CLI + workload/prompt provenance + shadow-eval only for opted-in workloads + export to LiteLLM/Langfuse/Braintrust. One MCP tool after CLI proves recommendation usefulness.

### CEO DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════════════
  Dimension                              Claude     Codex     Consensus
  ──────────────────────────────────────  ────────  ────────  ─────────
  1. Premises valid?                     DISAGREE  DISAGREE  DISAGREE
  2. Right problem to solve?             UNCLEAR   UNCLEAR   UNCLEAR
  3. Scope calibration correct?          DISAGREE  DISAGREE  DISAGREE
  4. Alternatives sufficiently explored? DISAGREE  DISAGREE  DISAGREE
  5. Competitive/market risks covered?   DISAGREE  DISAGREE  DISAGREE
  6. 6-month trajectory sound?           DISAGREE  UNCLEAR   DISAGREE
═══════════════════════════════════════════════════════════════════════
0/6 CONFIRMED. Both voices independently converge on: narrow v0.1.
Convergent recommendations: privacy defaults (hashes-only), incumbent
export/import compat, falsifiable adoption thesis, library-first over
six-piece-day-one.
```

**This is a USER CHALLENGE per autoplan rules.** Both voices independently recommend the user's stated "boil-the-ocean / all-six-day-one" direction be narrowed. The user's original direction holds by default — the premise gate below is the intervention point.

## Step 0.6: Reframe (Post-CEO-Review, User Decision)

**Gate outcome: the user rejected both the scope narrowing (B) and the startup/wedge framing entirely.** The review's market/competitive lens was misapplied; somm is not a product seeking PMF. Below is the reframed intent that supersedes the original framing. The 10-star integrated loop holds; the architectural/upkeep critiques are accepted.

### Reframed positioning

- **Target user:** individual or small team, free, self-hosted. Not a startup. Not a commercial offering. Not seeking PMF.
- **Dependency stance:** OSS dependencies that go with the grain are acceptable (LiteLLM as optional provider adapter, self-hosted Langfuse or similar as optional export sink). NOT reliant on commercial offerings — even "free tier" claims are rejected. Incumbents are sources to yoink ideas from, not to depend on.
- **Quality bar:** higher than "personal project" — designed so others can integrate and self-host with confidence. But explicit non-goal: competing with commercial offerings.
- **Rejected framings:** wedge, moat, competitive positioning, "defensible differentiation," startup validation theses.
- **Accepted framings:** architectural rigor, upkeep sustainability, privacy by default, design philosophy fit.

### Design philosophy: scales with use

The integrated loop is the *design target*, not a day-one activation requirement. Sophistication unlocks as usage matures; ceremony stays minimal throughout.

- **Library** activates immediately on `somm.llm(...)` — zero config, writes to local SQLite.
- **Service (`somm serve`)** activates when you run it. Optional. Library works without it.
- **Shadow-eval** activates when you opt in per workload with a budget ceiling. Uses local ollama + structural + embedding similarity by default; frontier judge is an opt-in upgrade.
- **Cross-project commons** activates when your second project connects to the same `~/.somm/global.sqlite`.
- **Agent recommendations** accumulate as evidence does; first useful output is when enough calls have landed.
- **Web admin** is a `somm serve`-time surface; read-only, no login story needed for single-user or small-team local use.
- **MCP** surfaces when a frontier coding agent connects; optional.

"Drop-in replacement that scales up in complexity with use/maturity" is the design contract with the user.

### Architectural patches accepted (apply to the rest of the plan)

1. **Privacy-first telemetry defaults.** The `samples` table is OFF by default. Store hashes only. Per-workload opt-in for prompt/response capture. `somm.workload(name, capture=...)` or config-level toggle. Never implicitly ship bodies upstream.
2. **Shadow-eval gated behind explicit opt-in + budget ceiling + ROI check.** Off by default. Per-workload enable. Config: `shadow.budget_usd_per_day`, `shadow.break_even_required=True`. Agent pauses shadow on a workload if cost_spent > expected_savings_delta. Prefer local-model judges (ollama) + structural + embedding similarity; frontier judge is an opt-in enhancement not a default.
3. **Workload ontology is first-class.** Workload objects have: `name`, `description`, `input_schema`, `output_schema`, `quality_criteria` (list of checks), `budget_cap`, `privacy_class`. Cross-project learning only federates signal across workloads with matching schema + quality_criteria shape — no global "anecdote storage."
4. **Skill is one of many agent-onboarding templates.** `packages/somm-skill/` contains `claude.md`, `codex.md`, `cursor.md`, `windsurf.md` derived from a shared template. The onboarding logic is the core; the agent-specific adapter is a thin wrapper. Prevents lock-in to Claude's skill conventions as they drift.
5. **Optional incumbent adapters (ergonomic, not positional).** `somm.adapters.litellm` for users who already use LiteLLM as their gateway — somm becomes an additional callback. `somm.exporters.langfuse_format` for users who want to ship traces into a self-hosted Langfuse. These are *ergonomic conveniences* for users migrating, not the product's positioning.
6. **No hot-path commercial dependencies.** The library MUST function with zero commercial API access (ollama local + self-hosted provider). Commercial APIs (Anthropic, OpenAI, Gemini PAYG) are *supported providers*, never *required dependencies*.
7. **Model-intel: self-hosted-first.** Scraper pulls from public endpoints (OpenRouter public API, HF public endpoints, RSS/Atom release feeds). Cached locally. If an upstream source disappears, the library degrades to the last cached snapshot; it doesn't break.
8. **Falsifiable success thesis (for us, not market).** "A Python project with 3+ LLM call sites drops `somm` in, runs a tick of real traffic, and within 24h can answer: which workload burned the most tokens on which provider, what fraction failed, and (if shadow-eval was opted-in) which model tested better." That's the v0.1 proof.

### Rejected patches (will not apply)

- **Narrow to library-first wedge (Approach B).** User rejected — integrated loop is the design, not a market wedge.
- **Compete-with-incumbents framing.** User rejected — yoink ideas, don't compete.
- **Commercial-incumbent export compat as primary positioning.** Optional ergonomic adapter only.

### Scope of v0.1 post-reframe

Unchanged from original plan: library + MCP + skill + service + web admin + agent + shadow-eval + model-intel workers. The Approach C patches are applied above; v0.1 feature list is otherwise intact. The 10-star integrated loop holds.

## Step 1–10: Review Sections (auto-decided on reframed plan)

### Section 1 — Architecture

**Dependency graph (ASCII):**

```
                  ┌─────────────────────────────────────┐
                  │          local filesystem           │
                  │   ./.somm/calls.sqlite (WAL)        │
                  │   ~/.somm/global.sqlite (opt-in)    │
                  └───────▲───────▲───────────▲─────────┘
                          │writer │reader     │reader
                          │       │           │
         ┌────────────────┴──┐ ┌──┴────────┐ ┌┴──────────┐
         │  library          │ │ service   │ │ MCP       │
         │  (sensor)         │ │ (workers  │ │ (tools)   │
         │                   │ │  + web)   │ │           │
         │  generate()       │ │           │ │ recommend │
         │  extract()        │ │ model     │ │ compare   │
         │  stream()         │ │ intel     │ │ replay    │
         │  workload()       │ │           │ │ stats     │
         │  prompt()         │ │ shadow    │ │ register  │
         │  provenance()     │ │ eval      │ │ workload  │
         │  probe()          │ │           │ │ register  │
         │                   │ │ agent     │ │ prompt    │
         │  providers:       │ │           │ │ search    │
         │   ollama          │ │ web admin │ │           │
         │   openrouter      │ │ (htmx)    │ └───────────┘
         │   minimax         │ │           │
         │   anthropic       │ │ /recommend│
         │   openai          │ │ /compare  │
         │   (litellm opt.)  │ │ /replay   │
         └───────────────────┘ │ /stats    │
                               │ /health   │
                               └───────────┘
                                     ▲
                                     │ stdio | http
                                ┌────┴─────┐
                                │ coding   │
                                │ agent    │
                                │ (skill)  │
                                └──────────┘
```

**Data flow (happy / nil / empty / error):**

```
INPUT ──▶ VALIDATION ──▶ ROUTE ──▶ PROVIDER ──▶ PERSIST ──▶ OUTPUT
   │          │            │          │           │           │
   ▼          ▼            ▼          ▼           ▼           ▼
 nil?      unregistered  all        timeout   SQLite     result
 empty?    workload?     cooled?    429       busy       + call_id
 >ctx?     privacy_cls   fallback   500       disk full  + provenance
           violation?    chain      bad JSON  schema
                         exhausted  empty     mismatch
                                    resp
```

**State machine — provider_health:**

```
     register
       │
       ▼
   ┌────────┐  fail_transient   ┌─────────┐
   │healthy │─────────────────▶│ cooling │
   │        │◀─────────────────│ (timer) │
   └────────┘  timer_expires    └────┬────┘
       ▲                             │ N consecutive
       │                             ▼ fails
       │                        ┌─────────┐
       │       manual           │ cooled  │
       └────────────────────────│ (long)  │
               recovery         └─────────┘
```

**Coupling:** library writes SQLite schema; service/MCP/web read + small writes. Schema is the coupling boundary. Migrations are coordinated via `schema_version` table + versioned migration files. Acceptable.

**Scaling:** 10× (100k calls/day) — SQLite WAL handles comfortably. 100× (1M calls/day) — shadow-eval worker contention possible. Flag for later; don't solve in v0.1.

**SPoFs (self-hosted context):** local ollama (transient, routing handles), SQLite file (persistent — include `somm doctor --repair`), no critical centralized SPoF (per sovereignty principle).

**Security arch:** no external network surface by default. All services bind to `localhost` only. README warns explicitly against exposing `somm serve` to non-localhost.

**Rollback:** `pip uninstall somm` + optionally `rm -rf .somm/`. Data is user-portable (SQLite export). Reversibility 4/5.

**Findings & auto-decisions:**
- Coupling through SQLite schema — acceptable with migrations. No action.
- 100× scale concern — flag in Long-Term Trajectory; no v0.1 action. (P6 bias-to-action.)
- Service-can-be-exposed risk — document in README. ACCEPT to scope (P1 completeness).

### Section 2 — Error & Rescue Map

```
METHOD / CODEPATH          | WHAT CAN GO WRONG             | EXCEPTION
───────────────────────────|───────────────────────────────|──────────────────
SommLLM.generate           | Provider timeout              | ProviderTimeout
                           | 429 rate limit                | ProviderRateLimit
                           | 5xx upstream                  | ProviderServerErr
                           | All providers cooled          | ProvidersExhausted
                           | Prompt > context window       | ContextOverflow
                           | Unregistered workload (strict)| WorkloadNotRegistered
                           | Unregistered prompt (strict)  | PromptNotRegistered
                           | Privacy violation (upstream   | PrivacyViolation
                           |   call on private workload)   |
SommLLM.extract_structured | Upstream JSON malformed       | StructuredExtractFail
                           | Response empty                | EmptyResponse
telemetry.write_call       | SQLite busy (WAL retry)       | TelemetryBusy
                           | Disk full                     | TelemetryDiskFull
                           | Schema version mismatch       | SchemaVersionMismatch
somm serve                 | Port in use                   | ServerPortBusy
                           | SQLite corrupted              | SqliteCorrupt
                           | Worker crash                  | WorkerCrashed
shadow_eval_worker         | Judge API fail                | JudgeUnavailable
                           | Budget exceeded               | ShadowBudgetExceeded
model_intel_worker         | Source unavailable (network)  | ModelIntelStale
                           | Source schema changed         | ModelIntelParseErr
agent_worker               | Insufficient data             | InsufficientEvidence

EXCEPTION                  | RESCUED?  | ACTION                      | USER SEES
───────────────────────────|───────────|─────────────────────────────|──────────
ProviderTimeout            | Y         | Retry 1x, then fallback     | (transparent — next provider)
ProviderRateLimit          | Y         | Cooldown + fallback         | (transparent)
ProviderServerErr          | Y         | Fallback                    | (transparent)
ProvidersExhausted         | Y         | Sleep smallest cooldown +   | Err result with outcome="exhausted"
                           |           | retry once                  | + provider="", model=""
ContextOverflow            | N         | Raise w/ suggested models   | SommContextOverflow exception
                           |           | from model_intel            |
WorkloadNotRegistered      | N         | Raise in strict mode; warn  | SommStrictMode exception
                           |           | in demo mode                |
PromptNotRegistered        | N         | Same as above               | SommStrictMode exception
PrivacyViolation           | N         | Raise — hard stop           | SommPrivacyViolation (loud)
StructuredExtractFail      | Y         | Return {"raw": text}        | {"raw": "..."} with .parse_err
EmptyResponse              | Y         | Outcome="empty"; fallback   | (transparent)
TelemetryBusy              | Y         | WAL retry 3x then log       | No impact on caller
TelemetryDiskFull          | Y         | Log, disable telemetry,     | Warn once; calls still work
                           |           | continue                    |
SchemaVersionMismatch      | N         | Raise; require somm doctor  | SommSchemaMismatch
ServerPortBusy             | N         | Raise with recovery hint    | "already running or port N busy"
SqliteCorrupt              | Partial   | somm doctor --repair        | Tells user to run doctor
WorkerCrashed              | Y         | Log + restart w/ backoff    | worker_heartbeat shows drift
JudgeUnavailable           | Y         | Requeue for next worker pass| Shadow-eval falls behind gracefully
ShadowBudgetExceeded       | Y         | Disable shadow on workload  | Alert + recommendation
                           |           | until next budget window    |
ModelIntelStale            | Y         | Use last-good cache         | Banner: "model intel N hours stale"
ModelIntelParseErr         | Y         | Log + skip, last-good cache | Same as stale
InsufficientEvidence       | Y         | Wait for more data          | Agent says: "need N more calls"
```

**Findings & auto-decisions:**
- ProvidersExhausted vs ContextOverflow — distinct exceptions; both are users' problems to handle. Keep distinct. (P5 explicit.)
- `somm doctor --repair` is required for SQLite corruption recovery — ACCEPT to scope (P1 completeness).
- PrivacyViolation is a loud exception, never silent — this is load-bearing. Enforced in router AND telemetry writer (defense in depth). (P1.)

No catch-all `except Exception`. Every handler names its class.

### Section 3 — Security & Threat Model

| Threat | Likelihood | Impact | Mitigated? |
|---|---|---|---|
| Attacker on localhost reads trace DB | Med | Med (if samples captured) | Defense: samples OFF default; file perms 0600 on `.somm/`. ACCEPT to scope. |
| `somm serve` exposed to LAN/internet | Low | High (trace leak, RCE on web eval) | Mitigation: bind localhost only by default; require explicit `--bind 0.0.0.0` with README warning. ACCEPT. |
| API key leak via telemetry | Low | High | Mitigation: adapters strip auth headers before logging; never log request headers. TEST. ACCEPT. |
| Prompt injection into upstream call | High | Low–Med | Not somm's problem — caller responsibility. Document clearly. |
| Supply-chain via LiteLLM/Langfuse adapter | Med | High | Optional deps only; pin + audit; lockfile required; README warns. ACCEPT. |
| Samples table leaks PII | Med | High | OFF by default; per-workload opt-in; privacy_class enforces upstream-routing ban. ACCEPT. |
| Cross-project commons leaks data | Med | High | `global.sqlite` opt-in; hashes-only by default same as per-project. ACCEPT. |
| Shadow-eval sends sensitive prompt to frontier | Med | High | `privacy_class=private` on workload bans shadow-eval outright. Hard enforcement. ACCEPT. |

**Findings:** all high-likelihood/high-impact threats have named mitigations in scope. No catch-all auth (intentional — local-only surface).

### Section 4 — Data Flow & Interaction Edge Cases

| Interaction | Edge case | Handled? | How |
|---|---|---|---|
| Concurrent `generate()` in threads | Race on call_id | Y | UUID4 per call; SQLite WAL handles row insert |
| Workload registered twice (different schemas) | Which wins? | Y | Content-address by schema hash; differing = new version. Last write wins on name pointer. ACCEPT. |
| Prompt body changed without bump | Silent drift | Y | Auto-hash + auto-bump minor version; explicit bump for major. (Resolves open question #5.) ACCEPT as default. |
| `somm serve` already running | Error clean? | Y | Detect pidfile; refuse second instance. ACCEPT. |
| Shadow-eval retroactive on old calls | Time-travel cost? | Y | Worker only considers calls in last N days (configurable). |
| All providers cooled simultaneously | Thundering herd on retry | Y | Sleep smallest cooldown before retry, not `now+small_const`. |
| Schema migration mid-run | Crash consistency | Y | Migrations atomic; rollback on failure; doctor detects. |
| `somm status` while writes in flight | Stale read? | Y | WAL allows reader to see consistent snapshot. |
| User exports then re-imports | Dup call_ids? | Y | UUID collision ≈ 0; exporter has `--dedupe`. |
| Model_intel scrape blocked by rate limit | Recovery? | Y | Backoff; last-good cache serves reads. |

### Section 5 — Code Quality

- **Organization:** src/somm/ modules match responsibilities. Provider adapters share `BaseProvider` abstract class (`call_chat`, `call_stream`, `estimate_tokens`, `parse_response`). ACCEPT.
- **DRY:** `<think>` block stripping + markdown-fence extraction + brace-balanced JSON extraction live in `somm.parse` — shared by all adapters.
- **Naming:** `SommLLM`, `SommResult`, `SommWorkload`, `SommPrompt` — prefix consistent. `somm.llm()` / `somm.workload()` / `somm.prompt()` lowercase factories. Matches library conventions.
- **Over-engineering flag:** cross-project mirror could slide into "federation protocol." Cap at v0.1: post-hook appends row to `~/.somm/global.sqlite` with `project` column. No federation, no sync protocol, no conflict resolution. (P5 explicit, P3 pragmatic.) ACCEPT.
- **Under-engineering flag:** shadow-eval grading is a lot at v0.1 (structural + embedding + optional judge). **Taste decision surfaced for final gate**: v0.1 with structural + embedding only (judge deferred) vs. all three. Default: structural + embedding day-one; frontier judge as a workload-level opt-in config flag, off until user turns it on.
- **Complexity:** no new method expected >5 branches.

### Section 6 — Test Review (CRITICAL — not compressed)

**NEW UX FLOWS:** CLI (status, tail, doctor, serve, compare, replay, audit, export), web admin dashboard, per-workload recommendation acceptance.

**NEW DATA FLOWS:** `call → telemetry`; `workload|prompt register → registry`; `shadow_eval_worker → eval_results`; `model_intel_worker → model_intel cache`; `agent_worker → recommendations`.

**NEW CODEPATHS:** 5 provider adapters × `{generate, extract_structured, stream, estimate}`; routing (cooldown, circuit breaker, probe, privacy_class filter); JSON extraction (markdown, brace, qwen2.5 quirk); registry hashing + version bump; provenance stamping; 7 MCP tools; 3 service workers; ~6 web routes.

**NEW BACKGROUND JOBS:** model_intel (24h), shadow_eval (continuous), agent (weekly).

**NEW INTEGRATIONS:** ollama, openrouter, minimax, anthropic, openai (optionally HF via model_intel scrape, optional LiteLLM adapter).

**NEW ERROR/RESCUE PATHS:** see Section 2.

**Test matrix:**

| Area | Unit | Integration | E2E |
|---|---|---|---|
| Provider adapters | mocked per-provider | VCR-recorded fixtures | via Project A graft |
| Routing | state machine under fuzzed fault injection | multi-provider against fixture | Project A graft |
| JSON extraction | quirk corpus (qwen2.5, openrouter, minimax think-blocks) | — | — |
| Workload/prompt registry | hash stability + version bump | registry → SQLite → registry read | — |
| Telemetry | row shape + schema versioning | concurrent write + read | — |
| Privacy enforcement | `privacy_class=private` bans upstream | shadow-eval refuses on private | Project A graft w/ a private workload |
| Shadow-eval | structural grader on known-good/bad | worker dispatches + stores result | — |
| Model-intel | parser against recorded OpenRouter/HF responses | scrape → cache → read | — |
| Agent | rec generation with synthetic evidence | agent → recommendations table | — |
| MCP tools | each tool mocked | stdio + HTTP transport | actual Claude Code session using the MCP |
| CLI | argparse correctness | commands against temp dir | — |
| Web | route returns 200 + correct HTML | htmx partial refresh | manual smoke |
| Cost estimation | price table snapshot | cost from recorded calls | — |

**Test ambition:**
- 2am-Friday test: `test_private_workload_never_upstream` — if privacy_class=private, no upstream provider is called even if local ollama is down.
- Hostile QA: `test_sql_injection_in_workload_name` — reject; `test_malformed_response_does_not_crash` — outcome=bad_json, no swallowed exception.
- Chaos: `test_all_providers_simultaneously_cooled` — library returns exhausted result, not deadlock.

**Flakiness risk:** none depending on wall-clock, randomness, or external services (VCR fixtures). Concurrent SQLite tests parameterized with explicit WAL flag.

**Test plan artifact:** written to `~/.gstack/projects/somm/master-test-plan-20260417-131441.md` (see Phase 3 for full artifact).

**For LLM/prompt changes:** since somm IS the LLM infrastructure, every PR touching provider adapters or shadow-eval must run the cost-model-snapshot + privacy-enforcement + fallback suites. Add to CI as required checks.

### Section 7 — Performance

- **N+1 risk:** `somm_stats` roll-up is GROUP BY `(workload, model)`. Add compound index `calls(workload, model, ts)`. ACCEPT.
- **Memory:** sample bodies on: ~10k tokens × 5% rate × 100k calls/day = ~500MB/day per project w/ opt-in. Compress column w/ zstd in SQLite blob? Defer. Hashes-only default means negligible by default.
- **Indexes (committed):** `calls(call_id PK)`, `calls(workload, ts)`, `calls(provider, ts)`, `calls(workload, model, ts)`, `eval_results(call_id)`, `model_intel(provider, model)`.
- **Caching:** `model_intel` read-through cache in-process (TTL 1h, refreshed by worker every 24h). Cost estimation hits cache.
- **Slow paths:** ollama cold-start (~10s model load, one-time per session), shadow-eval judge (1–5s per grade). Bounded.
- **Connection pool:** none for SQLite (per-thread connections via `check_same_thread=False` + WAL). Clean.

### Section 8 — Observability & Debuggability

- **Logging:** structured via Python `logging`, fields: `call_id`, `workload`, `provider`, `model`, `outcome`, `duration_ms`. Level INFO default; DEBUG available.
- **Metrics:** exposed via `/stats` HTTP endpoint + `somm_stats` MCP tool + `somm status` CLI. Optional Prometheus-compatible output at `/metrics`. ACCEPT.
- **Tracing:** `call_id` (UUID4) is the trace ID; links to `eval_results`, `recommendations`, `samples`. `somm replay {call_id}` replays.
- **Alerting:** `recommendations` has a sink config (stdout | file | webhook | smtp). Users wire per taste.
- **Dashboards:** web admin has three charts day-one. Adding a Pareto chart was accepted in 0D.
- **Debuggability at T+3 weeks:** every call's full metadata + (opted-in) body is queryable. `somm audit --call-id ...` gives full reconstruction.
- **Admin tooling:** `somm doctor` (health), `somm audit` (query calls), `somm export` (dump to JSON/Parquet), `somm doctor --repair` (SQLite integrity).
- **Runbooks:** README's troubleshooting section (ollama unreachable, all providers cooled, schema migration stuck, shadow-eval budget exhausted, worker not running).

**Gap & auto-decision:** worker observability — are workers running? `worker_heartbeat(worker_name, last_run_at, last_success_at, consecutive_failures)` table. Exposed via `somm doctor` + web dashboard. ACCEPT to scope (P1).

### Section 9 — Deployment & Rollout

- **Migrations:** pre-v0.1 ships with initial schema. Future schemas via versioned SQL files; `somm doctor` applies.
- **Feature flags:** `shadow_eval.enabled` (per-workload), `cross_project.enabled`, `model_intel.enabled` — all default OFF or opt-in post v0.1 install.
- **Rollout order:** N/A pre-v0.1. Post-v0.1: pip upgrade → somm doctor (applies migrations) → restart service.
- **Rollback:** `pip install somm==X.Y.Z-prev` + schema rollback file.
- **Deploy-time risk window:** running `somm serve` during a library-only upgrade is safe (WAL). Schema upgrades require service-stop → doctor → service-start.
- **Env parity:** developer's machine = production machine in self-hosted model. `somm doctor` validates config at install.
- **Post-deploy verification:** `somm smoke` runs a synthetic ollama call + stats query + MCP ping.

### Section 10 — Long-Term Trajectory

- **Tech debt:** shadow-eval grading is the thickest v0.1 surface; add a pluggable grader interface at v0.1 so future graders (classifier models, per-workload custom) slot in cleanly. ACCEPT (P5 explicit).
- **Path dependency:** SQLite schema locks in some decisions. Migrations are the release valve.
- **Knowledge:** PLAN.md + ARCHITECTURE.md (to be written) + README + inline docstrings. CLAUDE.md for contributor setup.
- **Reversibility:** 4/5 — users can export; data portable.
- **Ecosystem fit:** Python 3.12 + uv + SQLite + htmx + MCP spec. No SPA, no ORM, no microservice framework. Clean.
- **1-year question:** new contributor reads PLAN.md + ARCH + code — can they find where to work? Plan to write ARCHITECTURE.md as part of D7.
- **Platform potential:** workload ontology + cross-project commons are the platform features. Others can build on them.

### Section 11 — Design & UX (brief pass; deep review in Phase 2)

- **IA:** web admin is one page, three sections (status strip, Pareto chart, rec list). Clear hierarchy.
- **State coverage:**

| Feature | Loading | Empty | Error | Success | Partial |
|---|---|---|---|---|---|
| Status strip | Y spinner | Y "no calls yet — run somm.llm()" | Y (service down banner) | Y | N/A |
| Pareto chart | Y | Y "no shadow evals yet" | Y | Y | Y (tip: "N calls w/o shadow grade") |
| Rec list | Y | Y "no recs yet — need more data" | Y | Y | N/A |

- **User journey:** launch `somm serve` → open browser → glance at strip → scan recs → click a rec → see evidence → accept/dismiss.
- **AI slop risk:** design is opinionated (not generic dashboard templates). Phase 2 will audit.
- **DESIGN.md:** doesn't exist yet; Phase 2 will produce.
- **Responsive:** desktop-first developer tool. Graceful mobile degradation.
- **Accessibility:** semantic HTML, keyboard nav, high-contrast, prefers-reduced-motion honored.

## Step 1–10: Required outputs

### "NOT in scope" section
- Closed-loop A/B routing (deferred to TODOS.md)
- `somm.ensemble()` workflow primitive (TODOS.md)
- Auto-eval generation from production samples (TODOS.md)
- Federation protocol for cross-project (beyond local mirror)
- Multi-tenant web admin auth
- Commercial incumbent hosted-service integrations (intentional non-goal)
- Judge model at v0.1 (shipped off-by-default; structural + embedding are default graders)

### "What already exists" section
- pattern source 1: routing with roster cycling + cooldowns; metadata collector pattern; JSON extraction quirks
- pattern source 2: multi-platform eval runner; DuckDB evals table shape; LLMStats pattern
- Project A: `the target's existing LLM wrapper` compat surface — target for drop-in; `probe_providers()` parallel-slot pattern

### Error & Rescue Registry
See Section 2 table above.

### Failure Modes Registry

| Failure mode | Blast radius | Detection | Recovery |
|---|---|---|---|
| Provider cascades (all cooled) | single workload | `provider_health` all cooled | Sleep smallest cooldown + retry once; result=exhausted; agent flags chronic |
| SQLite corrupt | whole library | doctor startup check | `somm doctor --repair` |
| Schema mismatch (old lib + new schema) | library writes fail | telemetry writer error | `somm doctor` applies migrations |
| Shadow-eval budget exceeded | one workload | worker pre-check | disable shadow on workload; re-enable next budget window; alert |
| Model-intel scrape down | recommendations stale | worker logs + web banner | cached last-good serves reads; banner until refresh |
| Worker dead | drift over time | heartbeat table | `somm doctor` restarts; alert after 2× expected interval |
| Sample table growth unbounded | disk fill | telemetry disk-full handler | rotate + archive; auto-disable capture at threshold |
| Cross-project mirror stale | recommendations miss recent project data | heartbeat on global | post-hook retries; warn in doctor |
| Privacy violation attempt (blocked) | single call fails loud | exception + audit log | caller must fix workload classification |

### Dream state delta
Plan gets us ~60% to the 12-month ideal. Missing for v0.1: closed-loop A/B routing, workload ontology federation across projects, privacy-preserving cross-project learning. All tracked in TODOS.md.

### Completion Summary

| Dimension | Result |
|---|---|
| Mode | SELECTIVE EXPANSION |
| Premise gate | User rejected scope narrowing; rejected startup/wedge framing; applied Approach C patches |
| Consensus | 0/6 CONFIRMED on original plan; framing reset resolves most concerns |
| Scope decisions | Accepted: deterministic replay, Pareto chart, budget guardrails, worker heartbeat, doctor repair. Deferred: A/B routing, ensemble, auto-eval generation |
| Critical gaps found | Worker observability (fixed); SQLite corruption recovery (fixed); privacy default flip (fixed via reframe); shadow-eval budget gate (fixed via reframe); workload ontology as first-class (fixed via reframe) |
| Open taste decisions for final gate | Shadow-eval judge at v0.1 (off-by-default ACCEPTED, but whether to ship structural+embedding+judge all three or just two); caching opportunity detection (#5 from 0D); privacy classifier automation (#6 from 0D) |
| Ready for Phase 2 | ✓ |

---

# Design Review (autoplan Phase 2)

_Scope: somm-service web admin at `localhost:7878`. Classification: APP UI (local dev tool, utility-first, no marketing/brand framing). Consensus verdict: plan names components, not a designed screen — both voices converge on specific fixes._

## Step 0: Design Scope Assessment

- **Initial design rating:** 3/10. Plan mentions "one HTML page + three charts + recommendations list" (line 124 of pre-review draft), later expanded to mention a fourth Pareto chart (0D section) — already an internal conflict. No chart types, no palette, no typography, no spacing, no real states, no a11y specifics, no DESIGN.md.
- **DESIGN.md status:** does not exist. Recommended minimum = tokens.css (below), not full brand system.
- **Existing patterns:** none — greenfield. No repo UI to align with.
- **Focus areas:** information hierarchy, states, chart specificity, minimum design system.

## Step 0.5: Dual Voices

### CLAUDE SUBAGENT (design — independent review)

Rated 2–4/10 across 7 passes. Litmus scorecard: 0 passes. Top recommendations:
- Lead with single hero number + delta ("$3.42 yesterday, -18%").
- Rec list inline at top; charts as supporting evidence, not the payoff.
- Specify chart types: stacked area (cost/time), grid heatmap (provider health), sparkline small-multiples (quality per workload).
- Write `packages/somm-service/web/tokens.css` with 12 CSS variables: monospace + Inter; 8px scale; neutral + one accent; 3 status colors; no shadows; radius ≤4px.
- States matrix with real copy per cell; include stale, degraded, no-data-yet, budget-exceeded, insufficient-evidence.
- A11y: focus ring, tab order, `aria-live=polite` on recs, chart alt/table fallback, skip-link, color-independent status.

### CODEX (design — UX challenge, gpt-5.4)

Same direction, additional signals:
- Plan has a chart-count conflict: line 124 says "three charts + rec list"; reframe section mentions four. Standardize.
- First screen must answer: "OK / needs attention / not enough data" — a blunt diagnosis, not a dashboard.
- Missing recommendation states: `accept pending`, `success`, `failed`, `already applied`, `dismissed`.
- "What does accept DO?" is completely unspecified — mark applied? copy a command? open a diff? That's a gate decision.
- Workload color/status severity consistency: a recommendation, a chart segment, and a status badge for the same workload must share a color/hue. Otherwise the page becomes a rainbow.

### DESIGN LITMUS SCORECARD — CONSENSUS

```
═════════════════════════════════════════════════════════════════════
  Dimension                            Claude     Codex     Consensus
  ─────────────────────────────────── ────────  ────────  ──────────
  1. Information hierarchy specified? DISAGREE  DISAGREE  DISAGREE
  2. States covered?                  DISAGREE  DISAGREE  DISAGREE
  3. User journey clear?              UNCLEAR   UNCLEAR   UNCLEAR
  4. AI slop avoided?                 DISAGREE  DISAGREE  DISAGREE
  5. Design system aligned?           DISAGREE  DISAGREE  DISAGREE
  6. Responsive & a11y covered?       DISAGREE  UNCLEAR   DISAGREE
═════════════════════════════════════════════════════════════════════
0/6 CONFIRMED. Both voices converge on: hero status line, recs before
charts, explicit chart types, tokens.css, real states matrix, a11y spec.
```

## Pass 1 — Information Architecture (was 3/10 → fix to 10)

**Issue:** "Three charts + rec list" inverts the value. The payoff is "what should I change?" — burying it under charts breaks the emotional arc.

**Fix — page anatomy (ASCII):**

```
┌──────────────────────────────────────────────────────────────────┐
│  somm                        [ project ▾ ]     [ 7d ▾ ]          │  nav (sticky)
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [HEALTHY] 842 calls · 2.1% failed · $1.84 today · fresh 3m ago  │  status diagnosis (hero line)
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  Top recommendations                                    [3 new]  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ claim_extract → try gemma-3-27b-free                       │  │
│  │ 12% better quality, 40% lower latency, est. $0.30/d saving │  │
│  │ Evidence: 47 shadow calls · 5d window    [apply] [skip] [?] │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ topic_gate → bump prompt v2 → v3                           │  │
│  │ regression suspected (quality -14% since v3 shipped)       │  │
│  │ Evidence: 200 graded calls               [apply] [skip] [?] │  │
│  └────────────────────────────────────────────────────────────┘  │
│  [ show dismissed ]                                              │
├──────────────────────────────────────────────────────────────────┤
│  Evidence                                                        │
│                                                                  │
│  Cost over time  [stacked area, per workload]                    │
│  ▁▂▃▅▅▆▇▇▆▅▄▃▂▁                                                │
│                                                                  │
│  Provider health  [heatmap: provider × hour, success rate]       │
│  ollama        ■■■■■■■■■■■■■■■■■■■■■■■■                        │
│  openrouter    ■■░■■■■■■░░■■■■■■■■■■■■                          │
│  minimax       ■■■■■■■■■■■■■■■░░■■■■■■■                        │
│                                                                  │
│  Quality per workload  [small-multiples sparkline grid]          │
│  claim_extract  ▁▂▂▃▄▅▆▇                                        │
│  topic_gate     ▇▇▆▅▄▃▃▂  ← regressing                          │
│  relevance      ▆▇▇▇▇▇▇▇                                        │
│                                                                  │
│  Cost vs quality Pareto  [scatter, per (workload × model)]       │
│  (only renders when ≥2 models shadow-graded for the workload)    │
└──────────────────────────────────────────────────────────────────┘
```

**Order of visual weight:** hero status line → top recs → evidence charts. Rec list is the hero. Charts are subordinate.

**Rating after fix:** 10/10.

## Pass 2 — Interaction State Coverage (was 4/10 → fix to 10)

**Fix — full states matrix with copy:**

| Surface | Loading | Empty | Error | Success | Partial / stale |
|---|---|---|---|---|---|
| Status strip | `—· loading…` | "No calls yet. Run `somm.llm()` in your project." | `service unreachable (retrying…)` | `[HEALTHY] N calls · X% failed · $Y today` | `[HEALTHY · stale] model intel N hrs old` |
| Top-rec card | skeleton w/ pulse | "No recommendations yet. Agent needs N more calls." | "Couldn't load recs — check worker heartbeat" | card populated | "Evidence partial (shadow-eval disabled for this workload)" |
| Rec accept state | `applying…` | N/A | "Couldn't apply — check doctor" | `✓ applied 2m ago` | `pending verification` |
| Rec dismiss state | N/A | N/A | N/A | `dismissed · [undo]` | N/A |
| Cost chart | skeleton lines | "No paid calls yet." | "Chart failed to load" | chart | "Last bucket partial (still accruing)" |
| Provider heatmap | skeleton grid | "Only one provider used — heatmap needs ≥2." | "Chart failed" | heatmap | "One provider cooled; showing best-effort data" |
| Quality sparklines | skeleton rows | "No graded calls yet. Enable shadow-eval per workload." | — | sparkline grid | "Only N workloads have ≥10 grades; others show `—`" |
| Pareto scatter | skeleton | "Need ≥2 shadow-graded models per workload." | — | scatter | "N workloads below threshold; ghosted" |
| Budget-exceeded banner (shadow) | — | — | — | banner: "shadow-eval paused on `claim_extract` · budget $2.00 reached · [details]" | — |
| Schema-migration banner | — | — | "Schema migration required · [run somm doctor]" | — | — |
| Worker-drift banner | — | — | — | — | "model_intel last ran 3d ago · [check doctor]" |

**Rating after fix:** 10/10.

## Pass 3 — User Journey & Emotional Arc (was 4/10 → fix to 10)

**Fix — journey storyboard:**

| Step | User does | User feels | UI supports it with |
|---|---|---|---|
| 1 (0–2s) | Lands on page | "Am I burning money? Is something broken?" | Hero status line. Single sentence. Green/amber/red. |
| 2 (2–5s) | Scans recs | "Anything I can act on?" | Top-2 recs as distinct cards with one-sentence rationale + $ or % impact |
| 3 (5–30s) | Clicks a rec's `[?]` | "Do I trust this?" | Evidence panel: N shadow calls, confidence interval, time window, sample outputs side-by-side |
| 4 (30s–2m) | Drills into evidence | "Yes, apply." / "No, defer." / "This is bullshit." | `[apply]` triggers a *copy-safe* command or config diff; `[skip]` marks dismissed; `[?]` can reach into the sample pair |
| 5 (ongoing) | Glances again tomorrow | "Did my change help?" | Rec status: `applied · improving` / `applied · no change yet` / `applied · reverted` |

**What "apply" does (taste decision resolved via auto-decide per P5 explicit):** v0.1 = `[apply]` copies a config diff to clipboard AND marks the rec as `applied pending verification`. The user pastes the diff into their `pyproject.toml` or `somm.config`. NOT auto-writing to project files (too magic for v0.1; filesystem blast radius unclear). Verification occurs when subsequent calls on that workload show the expected delta.

**Rating after fix:** 10/10.

## Pass 4 — AI Slop Risk (was 2/10 → fix to 10)

**Fix — explicit UI vocabulary:**

| Element | Spec |
|---|---|
| Layout | single max-width column (1024px); sticky nav; NO sidebar; NO card mosaic; NO 3-column grid |
| Typography | `Inter` for prose, `JetBrains Mono` for numbers + identifiers. Scale: 12 / 14 / 16 / 20 / 28. Line-height 1.5 for prose, 1.2 for metrics. |
| Color | neutrals: bg `#0a0a0a` (dark) / `#fafafa` (light); text `#e8e8e8` / `#1a1a1a`; muted `#6b7280`; border `#27272a`. Semantic: ok `#059669`, warn `#d97706`, danger `#dc2626`, info `#3b82f6`. One accent for interactive: `#818cf8` (muted indigo). NO gradients. NO purple-on-white default. |
| Status pill | monochrome rectangle w/ semantic color + text + icon (color-independent). Example: `[● HEALTHY]` / `[▲ NEEDS ATTENTION]` / `[○ NO DATA]`. |
| Rec card | subtle border (`1px solid border`), radius 4px, no shadow, 16px padding. Title row (workload + arrow + proposal). Rationale line (one sentence, muted). Evidence line (numbers only, monospace). Action row at bottom. |
| Charts | direct-label axes; muted categorical palette (5 steps max); same workload → same hue across all charts and status badges; failures always danger-red; stale/unknown always muted-gray. |
| Motion | <100ms entrance transitions only. NO decorative. `prefers-reduced-motion: reduce` → instant. |
| Decoration | NONE — no blobs, no wavy dividers, no emoji in UI chrome, no colored-circle icons, no floating geometry. |

**Kill list:** no cards-with-icons-in-colored-circles, no centered-everything, no stacked-card-mosaic, no purple gradient, no large-radius bubbles.

**Rating after fix:** 9/10 (the last point earned by shipping the actual palette + typography file).

## Pass 5 — Design System Alignment (was 2/10 → fix to 10)

**Fix — ship `packages/somm-service/web/tokens.css` on D1:**

```css
:root {
  /* typography */
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;

  /* scale */
  --text-xs: 12px;  --text-sm: 14px;  --text-md: 16px;
  --text-lg: 20px;  --text-xl: 28px;

  /* spacing (4px grid) */
  --sp-1: 4px;  --sp-2: 8px;   --sp-3: 12px;
  --sp-4: 16px; --sp-5: 24px;  --sp-6: 32px;  --sp-7: 48px;

  /* neutrals */
  --bg: #0a0a0a;          --bg-alt: #111111;
  --fg: #e8e8e8;           --fg-muted: #9ca3af;
  --border: #27272a;

  /* semantic */
  --ok: #059669;           --warn: #d97706;
  --danger: #dc2626;       --info: #3b82f6;
  --accent: #818cf8;

  /* data-viz ramp (5 categorical steps, muted) */
  --dv-1: #93c5fd; --dv-2: #86efac; --dv-3: #fcd34d;
  --dv-4: #f9a8d4; --dv-5: #c4b5fd;

  /* radius + motion */
  --radius: 4px;
  --motion-fast: 80ms;

  /* workload color map is deterministic (hash of workload name into --dv-N) */
}

@media (prefers-color-scheme: light) {
  :root { --bg: #fafafa; --bg-alt: #ffffff; --fg: #1a1a1a; --fg-muted: #6b7280; --border: #e5e7eb; }
}
@media (prefers-reduced-motion: reduce) {
  :root { --motion-fast: 0ms; }
}
```

Document in `packages/somm-service/web/DESIGN.md` (short — just rules-of-thumb + token list + component anatomy).

**Rating after fix:** 10/10.

## Pass 6 — Responsive & A11y (was 3/10 → fix to 10)

**Fix — a11y spec (concrete):**

- Every interactive element is a `<button>` or `<a>` (never a `<div onclick>`).
- Tab order: skip-link → nav → status line → rec list (top rec first) → dismissed toggle → chart summaries.
- `aria-live="polite"` on rec list container; new recs announce without focus steal.
- Rec list is `<ol>` (implicit order by recency/impact).
- Status line is `role="status"` with `aria-live="polite"`.
- Every chart has a `<figure>` wrapper with `<figcaption>` summarizing the data (e.g., "Cost over the last 7 days. Peak $3.42 on day 4."). A `[view as table]` disclosure expands to `<table>` fallback.
- Focus ring: 2px solid `--accent`, 2px offset; never removed.
- Color-independent status: every semantic color is paired with an icon + text.
- Touch targets: 44px minimum on interactive elements.
- WCAG AA: all body text ≥ 4.5:1 contrast; chart axes ≥ 3:1.
- Keyboard shortcuts (optional): `J/K` to navigate recs, `A` to apply focused rec, `D` to dismiss, `?` to open evidence. Respecting `/` for search if added later. Document in a help modal.
- Responsive: single column at any width; charts have minimum widths below which they collapse to tables. No sidebar nav. `prefers-reduced-motion` honored.

**Rating after fix:** 10/10.

## Pass 7 — Unresolved Design Decisions (was 3/10 → decisions made)

| Decision | Resolution |
|---|---|
| What's the hero first-glance answer? | Single status line with N calls + failure % + $ today + freshness. (Pass 3) |
| Rec interaction model? | Inline in rec card; `[apply]` copies config diff to clipboard + marks pending-verification; `[skip]` dismisses with undo; `[?]` opens inline evidence panel. (Pass 3 resolution) |
| Default time window? | 7d on first open; persists per-project in localStorage. Configurable via `?window=24h|7d|30d|session`. |
| Per-project vs. global default view? | Per-project (current SQLite) by default; global toggle in nav switches to `~/.somm/global.sqlite`. Projects dropdown filters global view. |
| Chart count conflict (3 vs 4)? | Resolved: **4 charts** — cost over time (stacked area), provider health (heatmap), quality per workload (sparkline grid), cost-vs-quality (Pareto scatter). Pareto is conditional-render (needs ≥2 shadow-graded models per workload). |
| Workload color consistency? | Deterministic hash → `--dv-N` ramp. Same workload = same hue everywhere on the page. |
| Empty-state copy ownership? | Ship real copy day-one, not `TODO:`. See Pass 2 matrix. |
| Accessibility: chart fallbacks? | `<figure>` + `<figcaption>` + `[view as table]` disclosure → HTML `<table>`. See Pass 6. |

**Rating after fix:** 10/10.

## Design Review Completion Summary

| Dimension | Before | After | Fix artifact |
|---|---|---|---|
| Information architecture | 3/10 | 10/10 | Page anatomy ASCII |
| States coverage | 4/10 | 10/10 | Full states matrix with copy |
| User journey | 4/10 | 10/10 | Journey storyboard + "what apply does" spec |
| AI slop avoidance | 2/10 | 9/10 | Explicit UI vocabulary + kill list |
| Design system | 2/10 | 10/10 | `tokens.css` + `DESIGN.md` committed D1 |
| Responsive & a11y | 3/10 | 10/10 | A11y spec (concrete, not aspirational) |
| Unresolved decisions | 3/10 | 10/10 | 8 decisions resolved inline |

**Required outputs committed to the plan:**
- Page anatomy ASCII (Pass 1)
- States matrix with copy (Pass 2)
- Journey storyboard (Pass 3)
- `tokens.css` content (Pass 5)
- A11y spec (Pass 6)
- 8 resolved design decisions (Pass 7)

**Ready for Phase 3:** ✓

---

# Eng Review (autoplan Phase 3)

_Both voices returned ~40 combined findings. All accepted as auto-decidable architectural fixes per 6 principles (completeness, explicit, DRY, pragmatic). Consensus on 0/6 dimensions CONFIRMED — full patch list below. Test plan written to `~/.gstack/projects/somm/master-test-plan-20260417.md`._

## Step 0: Scope Challenge

Plan remains SELECTIVE EXPANSION scope (all six pieces day-one). Both voices validate the *scope* (given the reframe) but flag concrete architectural gaps. No sub-problem claimed "already solved" that needs reuse re-mapping — Phase 1 Step 0B's mapping holds. Complexity check: plan still touches >8 files / >2 classes, but the "smell" is the nature of an integrated local-first tool.

## Step 0.5: Dual Voices

### CLAUDE SUBAGENT (eng — independent)

Identified ~18 findings across 8 areas. Critical:
- SQLite WAL is **single-writer** — 4-thread library + 3 workers + cross-process writers = SQLITE_BUSY under load. Plan's "WAL handles it" is wrong.
- Schema version skew across 5 packages — library on v0.2 writing schema v4 while service on v0.1 reads v3 = silent data corruption.
- Workers stampede on service restart (no last-run gate).

High:
- XSS via workload names in htmx endpoints (no Jinja autoescape committed).
- File perms on shared machines (SQLite world-readable by default on Unix).
- VCR fixtures rot when provider APIs change.
- `<think>` block stripping across stream chunks undefined.
- Worker scheduler + job store undecided ("APScheduler or simple asyncio loops" punts).
- Package dependency graph (library ↔ service ↔ MCP) has version skew risk.

Medium:
- Immutable `calls` rows required for shadow-eval worker correctness.
- MCP dual-path (SQLite + HTTP) creates semantic drift.
- Missing threats: prompt leaking via export, credentials echoed in prompts, judge-egress of PII.
- Pareto chart conditional-render risk of dead UI.
- Workload-color hash collisions at >5 workloads.

### CODEX (eng — architecture challenge, gpt-5.4)

Extended the subagent with concrete structural recommendations. Did NOT rehash. Key extensions:

1. **Introduce `somm-core`** as shared package: schema migrations, config loading, DB repository layer, workload/prompt models, parse helpers, version constants. Every other package depends on `somm-core>=X.Y`. Prevents version skew everywhere.

2. **Keep SQLite but with disciplined write design** (disagrees with subagent's "single-writer queue per process" being central):
   - Per-process writer actor/queue so threads don't stampede
   - Short append-only transactions + `busy_timeout` + WAL + explicit retry
   - **JSONL spool fallback** at `.somm/spool/` on repeated `SQLITE_BUSY` or disk pressure
   - `somm doctor` or service drains the spool
   - This preserves zero-service hot-path telemetry (plan's best property)
   - DuckDB rejected for write path. LMDB tempting but loses SQL ergonomics. Postgres is a post-v0.1 backend.

3. **Custom `jobs` table, NOT APScheduler SQLAlchemy jobstore** — plan leans no-ORM; simple table with leases:
   ```sql
   jobs (job_name, due_at, locked_until, last_started_at, last_success_at, consecutive_failures)
   ```
   Atomic conditional UPDATE for lease acquisition. `somm doctor` can explain drift.

4. **Service lifecycle: ship first-class commands day-one** — workers are part of the value loop; casual `somm serve` in a terminal lets model-intel silently rot.
   - `somm service install --user`
   - `somm service start|stop|restart|status`
   - `somm service logs`
   - Backends: Linux = systemd user unit; macOS = launchd user agent; optional Docker image for home-server/NAS users; Windows deferred.

5. **Testing infra:**
   - GitHub Actions for public OSS (free, not a commercial dependency)
   - `uv run pytest` is authoritative locally
   - Fixture-freshness canary job (behind secret, non-required) nightly
   - Contract fixtures vs. canary tests distinguished explicitly

6. **Explicit migration commands:** `somm migrate` applies, `somm migrate --check` previews, `somm doctor` diagnoses + offers repair hints. Migrations are NOT a side effect of doctor.

7. **MCP one code path, two transports:** `somm-core` owns repository/query/recommendation APIs. MCP stdio → calls `somm-core` directly (service-optional). MCP HTTP → proxies to `somm serve`. Same typed responses either way. The duality is transport choice, not two implementations.

### ENG DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════════════
  Dimension                            Claude     Codex     Consensus
  ─────────────────────────────────── ────────  ────────  ──────────
  1. Architecture sound?              DISAGREE  UNCLEAR   DISAGREE
  2. Test coverage sufficient?        DISAGREE  UNCLEAR   DISAGREE
  3. Performance risks addressed?     DISAGREE  DISAGREE  DISAGREE
  4. Security threats covered?        DISAGREE  UNCLEAR   DISAGREE
  5. Error paths handled?             UNCLEAR   UNCLEAR   UNCLEAR
  6. Deployment risk manageable?      UNCLEAR   DISAGREE  DISAGREE
═══════════════════════════════════════════════════════════════════════
0/6 CONFIRMED. Both voices converge on specific architectural fixes.
No strategic disagreement — only implementation gaps.
```

## Architectural patches (auto-decided, ACCEPT all)

### Package structure (was: 5 packages; now: 5 packages with `somm-core` at the root)

```
packages/
├── somm-core/             # NEW: shared schema + repository + config + version
│   └── src/somm_core/
│       ├── schema.py      # schema version constant + migration runner
│       ├── migrations/    # versioned SQL files (0001_initial.sql, ...)
│       ├── repository.py  # query/write API used by library, service, mcp
│       ├── models.py      # workload, prompt, call, recommendation typed dicts
│       ├── config.py      # config loading (env + pyproject.toml + runtime)
│       ├── parse.py       # JSON extraction, <think> stripping (w/ streaming buf)
│       └── version.py     # single VERSION const; all packages import from here
├── somm/                  # library — depends on somm-core
├── somm-service/          # service (web + workers) — depends on somm-core
├── somm-mcp/              # MCP — depends on somm-core
└── somm-skill/            # templates: claude.md, codex.md, cursor.md, windsurf.md
```

- Unified version across all packages; released together.
- `somm-agent/` folded into `somm-service/` (it was just a CLI entry for a service worker, not a standalone package — subagent's D1 finding).

### Write path (patched)

```python
# somm-core/writer.py (used by library AND service workers)
class WriterQueue:
    """Per-process append-only writer. Threads enqueue; one writer drains."""
    # busy_timeout=5000ms, WAL, batched 100-row transactions
    # on SQLITE_BUSY after retries → spool to JSONL at .somm/spool/
    # spool drained by somm doctor OR service on startup

# cross-process writers (library in project A + library in project B + service):
# each writes to its own project DB (./.somm/calls.sqlite) without contention.
# global.sqlite mirror is opt-in post-hook, WAL + busy_timeout, idempotent on call_id UUID.
```

### Worker scheduler (patched)

```sql
CREATE TABLE jobs (
  job_name          TEXT PRIMARY KEY,
  due_at            TIMESTAMP NOT NULL,
  locked_until      TIMESTAMP,      -- lease expiry
  last_started_at   TIMESTAMP,
  last_success_at   TIMESTAMP,
  consecutive_failures INTEGER DEFAULT 0,
  interval_seconds  INTEGER NOT NULL
);
```

Lease acquisition:
```sql
UPDATE jobs
SET locked_until = ?, last_started_at = ?
WHERE job_name = ? AND due_at <= ? AND (locked_until IS NULL OR locked_until < ?)
```

No double-fire. Crashed worker's lease expires → next tick reclaims. Backoff on consecutive failures.

### Shadow-eval worker (patched for crash consistency)

`eval_results` has `grading_started_at TIMESTAMP`. Worker picks rows where `grading_started_at IS NULL OR grading_started_at < now()-10min`. On completion, writes result + clears started_at via UPDATE. Mid-crash leases reclaim.

### `calls` immutability (patched)

`calls` rows are immutable after insert. Late-arriving info goes in `call_updates (call_id, field, value, ts)`. Shadow-eval never reads mid-update state.

### Security hardening (patched)

- Jinja2 `autoescape=True` mandatory for web admin; CSP header `default-src 'self'`; no inline scripts.
- SQLite files chmod 0600; `.somm/` dir 0700; `somm doctor` verifies.
- `privacy_class=private` gate applies to router, shadow-eval, `recommendations.evidence_json`, `eval_results.gold_response`, and `somm export`.
- `somm serve --bind 0.0.0.0` prints interactive warning + requires confirmation flag (`--yes-i-know`) in CI.
- Optional deps via extras: `pip install somm[litellm]`, `somm[langfuse]`. Not core. Each extra documented for supply-chain audit.
- API key redaction: adapters strip auth headers before any logging; `test_api_key_never_in_telemetry` is load-bearing.
- `eval` / `exec` banned on user-origin strings (lint rule + pre-commit).

### Service lifecycle (NEW)

- `somm service install --user` writes systemd user unit (Linux) or launchd user agent (macOS).
- `somm service start|stop|restart|status|logs` — friendly wrappers around systemctl/launchctl.
- Optional Dockerfile at `packages/somm-service/Dockerfile` for home-server deployment.
- Windows: documented as manual (user runs `somm serve` in a terminal or Task Scheduler).

### Migration story (NEW)

- `somm migrate` — applies pending migrations, requires service stopped (pidfile check).
- `somm migrate --check` — previews without applying.
- `somm doctor` — diagnoses schema mismatch, perms issues, worker drift, spool backlog. Offers repair hints but DOES NOT apply migrations (that's `migrate`'s job).
- Library writer catches `OperationalError` on write; if `schema_version < expected`, raises `SommSchemaStale` with hint to run migrate + restart the process.

### MCP unification (patched)

- `somm-core.repository` has all query + recommend + compare + replay APIs.
- MCP stdio → calls `somm-core` directly (zero-service mode).
- MCP HTTP → proxies to `somm serve`'s routes; same typed responses.
- Either transport returns the same pydantic model shapes.

### Streaming + `<think>` buffer (patched)

- Stream adapter buffers a configurable lookahead window (default 2 KB).
- `<think>` open-tag detected → buffer chunks until close-tag or budget exhausted.
- Budget exhausted → emit whatever's in buffer with `stream_think_cap_hit=True` flag in metadata.
- `test_think_block_across_stream_chunks` covers multi-chunk think blocks.

### Workload color ramp (patched — TASTE DECISION surfaced)

**Surface for final gate:** workload color strategy.
- Option A: expand ramp from 5 to 12 slots (HSL rotation at fixed S/L); handles most projects; collisions above 12 workloads.
- Option B: top-5 by volume get hue slots; "+N others" aggregated into muted gray; scales infinitely.
- Option C: hue + label always; accept visual collision at charts, defer to labels for disambiguation.

Recommendation: Option A for v0.1 (simplest expand; label-disambiguation as backup).

### Fixture policy (NEW)

Contract fixtures (tracked in git) = shapes somm's parsers expect. Canary job (CI, behind secret, nightly, non-required) = proof live APIs still return that shape. PR auto-opener on drift.

### Tests added (all mandatory for v0.1)

- `test_private_workload_never_upstream` (privacy invariant)
- `test_xss_in_workload_name_via_web`
- `test_api_key_never_in_telemetry`
- `test_sql_injection_in_workload_name`
- `test_migration_while_writer_active`
- `test_concurrent_mirror_append`
- `test_grader_stable_across_runs`
- `test_think_block_across_stream_chunks`
- `test_all_providers_simultaneously_cooled`
- `test_sqlite_disk_full` (spool fallback)
- `test_worker_crash_mid_grading` (lease reclamation)
- `test_service_restart_no_stampede`
- `test_shared_machine_file_perms`

## Required outputs

### Architecture ASCII (updated with `somm-core`)

```
  ┌───────────────────────────────────────────────────────────────┐
  │                        somm-core                              │
  │   schema · repository · config · parse · version · migrations │
  └───────────▲──────────────────▲──────────────────▲─────────────┘
              │                  │                  │
              │                  │                  │
  ┌───────────┴────┐   ┌─────────┴──────┐   ┌───────┴────────┐
  │     somm       │   │ somm-service   │   │   somm-mcp     │
  │  (library)     │   │  (web+workers) │   │ (stdio + HTTP) │
  │                │   │                │   │                │
  │ .generate()    │   │ app.py (FastAPI│   │ 7 tools        │
  │ .extract()     │   │  / starlette)  │   │                │
  │ .stream()      │   │ workers:       │   │ stdio → core   │
  │ .workload()    │   │  model_intel   │   │ http → service │
  │ .prompt()      │   │  shadow_eval   │   │                │
  │ .provenance()  │   │  agent         │   │                │
  │ .probe()       │   │ web admin      │   │                │
  │ routing        │   │ /recommend     │   │                │
  │ WriterQueue    │   │ /compare       │   │                │
  │ spool fallback │   │ /replay /stats │   │                │
  │                │   │                │   │                │
  │ providers[5]   │   │ service CLI:   │   │                │
  │  (opt LiteLLM) │   │  install/start │   │                │
  └───────┬────────┘   │  /stop/status  │   │                │
          │            └────────┬───────┘   └────────────────┘
          │ writes              │ reads + small writes
          ▼                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              local filesystem (self-hosted)                  │
  │  ./.somm/calls.sqlite (WAL, chmod 0600)                      │
  │  ./.somm/spool/*.jsonl (writer backpressure)                 │
  │  ~/.somm/global.sqlite (opt-in mirror, chmod 0600)           │
  │  ~/.somm/model_intel.sqlite (scraped cache)                  │
  └──────────────────────────────────────────────────────────────┘
```

### Failure Modes Registry (updated with Phase 3 findings)

See PLAN.md Phase 1 Section 2 + Eng Review additions above. Critical gaps all patched.

### TODOS.md updates (accumulated across phases)

To write on finalization:
- A/B routing (Phase 1 deferred)
- `somm.ensemble()` primitive (Phase 1)
- Auto-eval generation from production samples (Phase 1)
- Postgres backend for small-team shared deployments (Phase 3 codex)
- Windows service lifecycle support (Phase 3)
- HF + release-feed model-intel sources (Phase 3 H1 deferred behind feature flag)
- Provider-specific tokenizers as `somm[tokenizers]` extras (Phase 3 H2)
- HF trending as model-intel source (Phase 3 H1)

### Eng Review Completion Summary

| Dimension | Before | After (with patches) |
|---|---|---|
| Architecture | DISAGREE | ACCEPT w/ `somm-core` + per-process writer + JSONL spool + custom jobs table |
| Test coverage | DISAGREE | ACCEPT w/ 13 new mandatory tests + contract/canary fixture policy |
| Performance | DISAGREE | ACCEPT w/ writer queue + batching + spool backpressure |
| Security | DISAGREE | ACCEPT w/ autoescape + CSP + chmod 0600 + privacy_class gate expansion + key redaction |
| Error paths | UNCLEAR | ACCEPT w/ immutable calls + call_updates + lease reclamation + SommSchemaStale |
| Deployment | DISAGREE | ACCEPT w/ somm service install/start/stop + systemd/launchd + explicit migrate cmd |

**Test plan artifact:** written to `~/.gstack/projects/somm/master-test-plan-20260417.md` ✓
**Ready for Phase 3.5:** ✓

---

# DX Review (autoplan Phase 3.5)

_Scope: Python library + CLI + MCP server + skill templates. Mode: DX POLISH. Both voices converge on concrete developer-moment-to-moment fixes. 0/6 CONFIRMED pre-patches; full fix list below._

## Step 0: DX Investigation

### 0A. Developer persona

**Primary:** solo developer or small-team maintainer. Python 3.12 + uv. Has a project with 3+ LLM call sites (provider SDKs or raw httpx). Runs ollama locally + has OpenRouter/Anthropic keys. Self-hosts everything. Evaluating somm at 9pm Tuesday after hitting a free-tier cooldown for the fourth time tonight.

**Emotional state at first touch:** curious-but-impatient. Has seen LiteLLM (2 lines to call) and Langfuse (decorator). Will abandon if TTHW > ~8 min or first rendered error lacks a fix-hint.

### 0B. Competitive TTHW benchmark

- LiteLLM: 2 lines (`completion(model=..., messages=...)`). TTHW ~1 min.
- Langfuse: decorator + one env var (self-hosted or cloud). TTHW ~3 min self-hosted.
- Braintrust: `braintrust init` then `Eval()` — TTHW ~5 min (but cloud-first, violates sovereignty).
- **somm target:** <5 min for zero-config path. Post-patches below: achievable at ~3 min.

### 0C. Magical moment

The user's explicit "magical moment" from the original plan: agent emits *"switch `claim_extract` to gemma-3-27b — 12% better on the shadow eval, 40% faster, 60% cheaper."* This only fires after enough telemetry accumulates + user opts into shadow-eval. Onboarding must get the user to the first accumulating call in <5 min so this moment is reachable.

### 0D. Developer journey map

| Stage | Time (CC+gstack) | User action | User feels | Plan support (pre-patch) | Post-patch |
|---|---|---|---|---|---|
| 1 | 0–30s | Lands on README | "Show me the shape in 20 seconds" | README claims 2-liner; reality requires registration | Copy-paste 4-liner works in demo mode |
| 2 | 30s–2m | `uv add somm`, run 4-liner | "Does this actually work?" | May hit `SommStrictMode` | Demo mode accepts; telemetry row written |
| 3 | 2m–4m | Runs `somm show status` | "What did I just log?" | Works, but CLI name discovery unclear | Grouped CLI; `somm show` shows the options |
| 4 | 4m–6m | `somm serve`, opens dashboard | "Ooh, a UI" | Port-collision risk | Smart port selection + `SOMM_PORT_BUSY` error |
| 5 | 6m–10m | `somm init --strict` | "Let me do this properly" | Unclear upgrade path from demo→strict | `init` scaffolds workloads + prompts + privacy_class |
| 6 | 10m–15m | `somm mcp install --client=claude-code` | "Integrate with my coding agent" | Plan had no install command | One command; auto-edits config; confirms health |
| 7 | Day 2+ | First agent recommendation | "*This* is why I installed" | Plan supports | Plan supports |

### 0E. Mode selection: DX POLISH

Scope is fixed (DX POLISH per autoplan override for this phase). All fixes target "bulletproof every touchpoint." No scope expansion; no triage.

### 0F. TTHW + first-time roleplay

**Pre-patch TTHW: 8–12 min happy path, 20+ min with friction.**
**Post-patch TTHW target: 3 min (zero-config demo); 10 min (strict init + service + MCP install).**

## Step 0.5: Dual Voices

### CLAUDE SUBAGENT (DX — independent)

Scored passes 1–8 at 1–6/10. Top findings:
- Strict-mode default (from reframe) breaks README 2-liner.
- No `somm init`, no `somm scaffold workload`, no `somm mcp install --client=X`.
- CLI ~20 verbs without grouping.
- `somm.probe()` opaque; `result.mark()` stringly-typed.
- Error taxonomy named but strings not committed.
- No cookbook/recipes dir; no comparison-to-incumbents page.
- No `SommProvider` protocol for third-party adapters.
- No DX measurement beacon.
- Realistic TTHW 8–12 min.

### CODEX (DX — developer experience challenge, gpt-5.4)

CONFIRMED and EXTENDED the subagent. Additional concrete artifacts:
- 4 canonical error strings committed (SOMM_WORKLOAD_UNREGISTERED, SOMM_PORT_BUSY, SOMM_PROVIDERS_EXHAUSTED, SOMM_SCHEMA_STALE) with problem/cause/fix/link shape.
- CLI grouping structure specified (`somm show|admin|service|workload|prompt|provider|mcp`).
- `SommProvider` protocol as typing.Protocol with specific methods (generate, stream, health, models).
- Provider entry-points discovery: `pip install somm-provider-groq` + `somm provider add groq`.
- `somm mcp install --client=cursor|claude-code|windsurf` + `somm mcp status` as required DX.
- Deprecation format: `SOMM_DEPRECATED_API` string with removal version.
- Docs layout: README + quickstart + errors/* + recipes/ + mcp/ + plugins/ + upgrading/ + examples/.

**Verdict (both):** "keep the integrated loop, but make the first five minutes radically boring."

### DX DUAL VOICES — CONSENSUS TABLE

```
═══════════════════════════════════════════════════════════════════════
  Dimension                           Claude     Codex     Consensus
  ──────────────────────────────────── ────────  ────────  ──────────
  1. Getting started < 5 min?         DISAGREE  DISAGREE  DISAGREE
  2. API/CLI naming guessable?        UNCLEAR   DISAGREE  DISAGREE
  3. Error messages actionable?       UNCLEAR   UNCLEAR   UNCLEAR
  4. Docs findable & complete?        DISAGREE  UNCLEAR   DISAGREE
  5. Upgrade path safe?               CONFIRMED UNCLEAR   UNCLEAR
  6. Dev environment friction-free?   DISAGREE  DISAGREE  DISAGREE
═══════════════════════════════════════════════════════════════════════
5/6 DISAGREE or UNCLEAR. Upgrade path is the closest to sound.
Fixes are mechanical — no strategic disagreement.
```

## DX Patches (auto-decided, ACCEPT all)

### Demo-mode default (USER CHALLENGE resolution)

**Change:** library default mode = `observe` (demo-like). Strict mode opt-in via `somm init --strict` or `somm.llm(mode="strict")`.

**Why:** both models independently argue that strict-by-default kills TTHW. Plan's own 2-liner README example is incompatible with strict-by-default. The original "strict discipline" goal is preserved via `init --strict` for production use. Demo/observe still writes telemetry (with auto-registered ad_hoc workloads) + warns on unregistered usage with a path to harden.

**Auto-decide reasoning:** this was Plan's open question #6, not a core user-stated direction. The user accepted the architectural/upkeep patches broadly; demo-by-default IS a DX patch. Auto-decide (P3 pragmatic + P6 bias-to-action + the 6-principle test "what a new contributor reads in 30 seconds").

### Canonical error format (MANDATORY, committed now)

Every SommError subclass renders via `__str__` as:

```
SOMM_ERROR_CODE

Problem: <one sentence>
Cause: <one sentence>
Fix:
  <command(s) or code change>
Docs: docs/errors/SOMM_ERROR_CODE.md
```

**4 canonical error strings committed to the plan (implementer ships these verbatim):**

```
SOMM_WORKLOAD_UNREGISTERED

Problem: This call used workload "invoice_extract", but it is not registered.
Cause: strict mode requires workload metadata before calls are logged.
Fix:
  somm workload add invoice_extract --from-example structured-extraction
  # or switch to observe mode:
  export SOMM_MODE=observe
Docs: docs/errors/SOMM_WORKLOAD_UNREGISTERED.md
```

```
SOMM_PORT_BUSY

Problem: localhost:7878 is already in use.
Cause: another somm service or local process is bound to this port.
Fix:
  somm service status
  somm serve --port 7879
  lsof -i :7878
Docs: docs/errors/SOMM_PORT_BUSY.md
```

```
SOMM_PROVIDERS_EXHAUSTED

Problem: No configured provider is currently available for workload "summarize_doc".
Cause: ollama timed out; openrouter is rate-limited until 14:32:10; anthropic is not configured.
Fix:
  somm show providers
  somm provider add ollama --model llama3.1
  somm run summarize_doc --wait
Docs: docs/errors/SOMM_PROVIDERS_EXHAUSTED.md
```

```
SOMM_SCHEMA_STALE

Problem: ./.somm/calls.sqlite is schema v3, but this somm version requires v4.
Cause: somm was upgraded and migrations have not been applied.
Fix:
  somm service stop
  somm migrate --check
  somm migrate
  somm service start
Docs: docs/errors/SOMM_SCHEMA_STALE.md
```

Every other exception class follows the same shape. Test: `test_all_sommerror_subclasses_have_problem_cause_fix_docs` enumerates and validates.

### CLI grouping (from ~20 flat verbs to top-level groups)

```
somm init                        # scaffold new project integration
somm run <workload> ...          # one-shot execute a registered workload
somm show {status, tail, calls, recs, providers, workloads}
somm compare <prompt> --models=[...]
somm replay <call_id> [--with=model]

somm workload {add, list, show, rm}
somm prompt {add, list, show, bump}
somm provider {add, list, test, rm}
somm mcp {install --client=X, status, remove --client=X}

somm service {install, start, stop, restart, status, logs}
somm admin {doctor, migrate, migrate --check, repair, export, smoke}
```

Top-level `somm --help` shows only the daily-path front (init, run, show, compare, replay, workload, prompt, provider, mcp). Destructive/operational commands live under `admin` and `service`. `somm help all` expands. Shell completion ships via `somm completion {bash,zsh,fish}`.

### API renames (breaking early — before any release)

- `somm.probe(n_slots)` → `somm.parallel_slots(n)`
- `result.mark("ok")` → `result.mark(somm.Outcome.OK)` (typed enum; `Outcome.{OK, BAD_JSON, EMPTY, OFF_TASK, …}`)
- `somm.llm()` gains optional `mode="observe"|"strict"` parameter (default `observe`)

### `somm init` scaffold

```
somm init [--strict] [--with-service] [--with-mcp-client=claude-code|cursor|windsurf|codex]
```

Does:
1. Creates `./.somm/` directory + initial SQLite + chmod 0600.
2. Writes `somm.toml` at project root (or merges `[tool.somm]` into existing `pyproject.toml`).
3. `--strict` → registers starter workloads (`default`, `extract`, `classify`), starter prompts, asks for privacy_class.
4. `--with-service` → installs systemd/launchd user unit.
5. `--with-mcp-client=X` → configures the named coding agent's MCP client for stdio connection to somm.
6. Runs `somm admin smoke` to verify ollama reachability.

### `somm mcp install` — editor integration

```
somm mcp install --client=claude-code
somm mcp install --client=cursor
somm mcp install --client=windsurf
somm mcp install --client=codex
somm mcp install --print-config     # prints JSON instead of auto-editing
somm mcp status                     # shows installed clients + health
somm mcp remove --client=X
```

Auto-edits the agent's config file (with --dry-run option). Uses `somm-core` for the MCP server entry. Detects config path per client. Manual JSON editing should never be required.

### `SommProvider` protocol for extensibility

```python
from typing import Protocol, Iterator

class SommProvider(Protocol):
    name: str
    def generate(self, request: SommRequest) -> SommResponse: ...
    def stream(self, request: SommRequest) -> Iterator[SommChunk]: ...
    def health(self) -> ProviderHealth: ...
    def models(self) -> list[SommModel]: ...
    def estimate_tokens(self, text: str, model: str) -> int: ...
```

Registration via entry points:
```toml
# somm-provider-groq/pyproject.toml
[project.entry-points."somm.providers"]
groq = "somm_provider_groq:GroqProvider"
```

Discovery + use:
```
pip install somm-provider-groq
somm provider add groq --model llama-3.3-70b
somm provider test groq
```

Built-in providers (ollama, openrouter, minimax, anthropic, openai) use the same protocol so they're not special. Third parties (Together, Fireworks, Groq, vLLM, LM Studio, custom internal gateways) can ship without a somm fork.

### Docs layout (committed for D6–D7)

```
README.md                     # 2-minute hello world + demo mode default
CHANGELOG.md
CONTRIBUTING.md
ARCHITECTURE.md               # somm-core + package dep graph
DESIGN.md                     # tokens, web admin anatomy (Phase 2)
docs/
  quickstart.md               # <5 min, strict flow
  errors/
    SOMM_WORKLOAD_UNREGISTERED.md
    SOMM_PORT_BUSY.md
    SOMM_PROVIDERS_EXHAUSTED.md
    SOMM_SCHEMA_STALE.md
    ... (one per exception class)
  recipes/
    json_extract.py
    parallel_workers.py
    streaming.py
    shadow_eval_optin.py
    cross_project_commons.py
    private_workload.py
    litellm_adapter.py
    migrate_from_raw_openai.py
  mcp/
    claude-code.md
    cursor.md
    windsurf.md
    codex.md
  plugins/
    provider_protocol.md     # SommProvider reference
    grader_protocol.md       # pluggable shadow-eval graders
  upgrading/
    0.1-to-0.2.md            # per-minor-version guide
examples/                    # runnable complete projects, not snippets
```

Every `docs/recipes/*.py` is a complete runnable script with imports, install command, env vars, and expected output commented at top.

### Deprecation warning format

```
SOMM_DEPRECATED_API

somm.probe() is deprecated and will be removed in 0.4.
Use somm.parallel_slots() instead.
```

Shown as runtime `DeprecationWarning` with stacklevel=2. Policy: one-minor-version overlap minimum; CHANGELOG entry required; `somm admin doctor` reports pending deprecations in use.

### CHANGELOG + version skew

- `CHANGELOG.md` ships from v0.1 following Keep-a-Changelog format.
- `somm admin doctor` reports `library version`, `service version`, `schema version`, and flags any skew.
- `somm upgrade-check` (one-shot): shows pending deprecations, pending migrations, and a "safe / needs-attention / blocked" summary.

### DX measurement (v0.1 scope)

Opt-in only (per sovereignty). If user runs `somm admin dx-report`, prints anonymous session timings:
- Time from `somm init` to first `.generate()` success
- Time from `somm serve` start to first web view
- Time from `somm mcp install` to first MCP tool call

No beacon by default. No external endpoint. The report is local-only unless user explicitly shares it (e.g., in a bug report).

## Pass Scores (before → after)

| Pass | Before | After |
|---|---|---|
| 1. Getting Started Experience | 4/10 | 9/10 — demo default + `somm init` + smart port |
| 2. API/CLI design | 5/10 | 9/10 — grouped CLI + renames + enum outcomes |
| 3. Error Messages & Debugging | 6/10 | 10/10 — 4-part format committed + 4 canonical strings |
| 4. Documentation & Learning | 3/10 | 9/10 — committed docs tree + recipes + quickstart |
| 5. Upgrade & Migration Path | 6/10 | 10/10 — explicit migrate + `upgrade-check` + CHANGELOG + dep format |
| 6. Dev Environment & Tooling | 5/10 | 10/10 — `somm init`, `somm mcp install --client=X`, `somm admin smoke` |
| 7. Community & Ecosystem | 2/10 | 8/10 — `SommProvider` protocol + entry-points + CONTRIBUTING + Discussions |
| 8. DX Measurement | 1/10 | 6/10 — opt-in `dx-report` locally; no beacon |

**Overall DX:** 4/10 → 9/10.

**TTHW:** 8–12 min → 3 min (demo path), 10 min (strict init path).

## Required outputs

### Developer Persona Card
Solo/small-team Python dev; uv + ollama; self-hosted; sovereignty-first. TTHW target <5 min for zero-config.

### Developer Empathy Narrative

*9pm Tuesday. I open somm's README. The hero example: `uv add somm; somm.llm(); .generate("Say hello in JSON")`. Runs, writes SQLite. In 90 seconds I've got telemetry. Next I try `somm show status` — shows my single call. Good. I run `somm serve`, open `localhost:7878`, see the hero status line "HEALTHY · 1 call · $0.00 · fresh 30s ago." I grin. I try `somm init --strict --with-mcp-client=claude-code` — it asks me three questions about my workload, writes somm.toml, drops the MCP config into ~/.claude/config, and runs a smoke check. I start a Claude Code session, ask "what model should I use for my claim extractor?" — the agent calls `somm_recommend`, gets an answer grounded in my real local telemetry. That's the magical moment.*

### Competitive DX benchmark
- LiteLLM 2-liner TTHW: ~1 min (somm v0.1 target: matched at ~2 min in observe mode)
- Langfuse self-hosted TTHW: ~3 min (somm v0.1 target: matched at ~3 min)
- Braintrust TTHW: ~5 min (somm v0.1 target: beaten at 3 min observe / 10 min full stack)

### Magical Moment Specification
Agent-mediated recommendation surfaced in Claude Code / Cursor / Windsurf session, grounded in user's real local telemetry, with `[apply]` copying a config diff to clipboard. This is the user's declared "10-star moment" from the reframe.

### DX Scorecard

| Dimension | Score |
|---|---|
| Getting started | 9/10 |
| API/CLI | 9/10 |
| Errors & debugging | 10/10 |
| Docs | 9/10 |
| Upgrade path | 10/10 |
| Dev env & tooling | 10/10 |
| Community & ecosystem | 8/10 |
| DX measurement | 6/10 |
| **Overall DX** | **9/10** |

### DX Implementation Checklist (ship with v0.1)

- [ ] Demo/observe mode default in library; strict opt-in via `--strict` or `mode="strict"`
- [ ] All SommError subclasses render via 4-part format; 4 canonical strings committed
- [ ] CLI grouped (show/admin/service/workload/prompt/provider/mcp); shell completion for bash/zsh/fish
- [ ] API renames: `probe → parallel_slots`; `mark()` takes `Outcome` enum
- [ ] `somm init [--strict] [--with-service] [--with-mcp-client=X]`
- [ ] `somm mcp install --client=claude-code|cursor|windsurf|codex` + `somm mcp status`
- [ ] `SommProvider` protocol + entry-points discovery + `somm provider add|list|test`
- [ ] Docs tree: README + quickstart + errors/* + recipes/ (8 files) + mcp/* + plugins/* + upgrading/
- [ ] `DeprecationWarning` format + `somm upgrade-check` + `CHANGELOG.md`
- [ ] `somm admin dx-report` local-only; no beacon
- [ ] `test_all_sommerror_subclasses_have_problem_cause_fix_docs`

**Ready for Phase 4 (Final Gate):** ✓

---

# Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale |
|---|---|---|---|---|---|
| 1 | CEO-0D | Deterministic replay with captured context | Mechanical | P1 completeness | in blast radius + <1d CC, upgrades existing call_id |
| 2 | CEO-0D | Cost/quality Pareto chart added to web admin | Mechanical | P1, P3 | one more chart in an already-scoped surface |
| 3 | CEO-0D | Budget circuit breaker (shadow-eval ROI gate) | Mechanical | P1, P3 | resolves open question #8 |
| 4 | CEO-0D | A/B routing → defer to TODOS.md | Mechanical | P3 pragmatic | >2d CC + significant new routing design |
| 5 | CEO-0D | Caching opportunity detection | Taste | P5 explicit | 3-4 files, provider-specific — surface to user |
| 6 | CEO-0D | Privacy classifier automation | Taste | P1, P5 | new config shape; needs user thought — surface |
| 7 | CEO-0D | `somm.ensemble()` → defer to TODOS.md | Mechanical | P3 | >2d CC; v0.1 doesn't need it |
| 8 | CEO-0D | Auto-evals from samples → defer to TODOS.md | Mechanical | P3 | >2d CC; builds on shadow-eval |
| 9 | CEO-S1 | Service-can-be-exposed docs in README | Mechanical | P1 completeness | docs-only addition |
| 10 | CEO-S4 | Content-addressed workload id (hash of schema) | Mechanical | P1, P4 DRY | resolves ambiguity on re-registration |
| 11 | CEO-S4 | Auto-bump prompt minor version on body change | Mechanical | P1 completeness | resolves open question #5; auto is safer default |
| 12 | CEO-S2 | Immutable `calls` rows + `call_updates` table | Mechanical | P1, P5 | shadow-eval correctness (extended in Eng) |
| 13 | CEO-S6 | Pluggable grader interface + judge opt-in not default | Taste | P5 explicit | surface to user: ship judge or defer |
| 14 | CEO-S8 | `worker_heartbeat` table + exposed in doctor | Mechanical | P1 | worker observability gap |
| 15 | CEO | USER CHALLENGE → scope held at 10-star; patches accepted | USER CHALLENGE | (user decision) | user rejected scope narrowing; rejected startup/wedge framing; accepted 7 architectural patches |
| 16 | Design-P1 | Hero status line + recs-above-charts IA | Mechanical | P1 completeness | both voices converge; payoff first |
| 17 | Design-P2 | Full states matrix with real copy per cell | Mechanical | P1 completeness | "TODO: empty state" = bug; ship copy day-one |
| 18 | Design-P3 | `[apply]` copies config diff to clipboard | Mechanical | P5 explicit | auto-writing project files too magic for v0.1 |
| 19 | Design-P4 | Explicit chart types (stacked area / heatmap / sparkline grid / Pareto scatter) | Mechanical | P5 | "three charts" = AI slop |
| 20 | Design-P5 | `tokens.css` with 12 CSS vars + DESIGN.md committed D1 | Mechanical | P1 | minimum design system |
| 21 | Design-P6 | A11y spec: buttons, focus ring, aria-live, chart alt/table, keyboard shortcuts | Mechanical | P1 | aspirational → specific |
| 22 | Design-P7 | Chart count standardized to 4 (resolves plan conflict) | Mechanical | P5 | Pareto as conditional render |
| 23 | Design-P7 | Workload color ramp strategy | Taste | P5 | surface: expand-12 vs top-5+aggregate vs label-disambig |
| 24 | Eng-A1 | `somm-core` package — shared schema + repo + config + parse + version | Mechanical | P1, P4, P5 | both voices agree; prevents version skew |
| 25 | Eng-E1 | Per-process writer queue + JSONL spool fallback on SQLITE_BUSY | Mechanical | P3, P5 | preserves zero-service hot path; disk-safe |
| 26 | Eng-W1 | Custom `jobs` table with atomic lease updates (not APScheduler) | Mechanical | P3, P5 | plan leans no-ORM |
| 27 | Eng-A2 | MCP one code path, two transports (stdio-direct + http-proxy) | Mechanical | P4 DRY | no semantic drift |
| 28 | Eng-S1 | Jinja autoescape + CSP header + `somm serve --bind 0.0.0.0` warning | Mechanical | P1 | XSS mitigation |
| 29 | Eng-S4 | SQLite files chmod 0600; `.somm/` dir 0700; doctor verifies | Mechanical | P1 | shared-machine privacy |
| 30 | Eng-S6 | `privacy_class` gate applies to recommendations + eval_results + export | Mechanical | P1 | defense in depth |
| 31 | Eng-T1 | Contract fixtures + canary job + PR-opener on drift | Mechanical | P1 | fixture rot mitigation |
| 32 | Eng-H3 | Streaming `<think>` buffered with lookahead + cap flag | Mechanical | P1 | correctness |
| 33 | Eng-M | Explicit `somm migrate` command; doctor diagnoses only | Mechanical | P5 explicit | magical migrations = bug risk |
| 34 | Eng-Deploy | `somm service install/start/stop/status/logs` + systemd/launchd day-one | Mechanical | P1 completeness | workers are part of value loop |
| 35 | Eng-Deploy | Optional Dockerfile for home-server users | Taste | P3 | ~0.5d to write; surface: ship v0.1 or defer |
| 36 | Eng-D1 | `somm-agent/` folded into `somm-service/` | Mechanical | P3, P4 | was always a service worker CLI |
| 37 | Eng | Nightly fixture-freshness CI job behind secret | Mechanical | P1 | non-required, doesn't block |
| 38 | Eng | 13 mandatory new tests (privacy, XSS, migration, streaming, …) | Mechanical | P1 | load-bearing invariants |
| 39 | DX-1 | Demo/observe mode default; strict opt-in | Mechanical | P3 pragmatic, P6 | original was open question #6, not user-stated; both DX voices strongly argue demo-default |
| 40 | DX-3 | Canonical error format (problem/cause/fix/link) + 4 strings committed | Mechanical | P1 completeness | aspirational → concrete |
| 41 | DX-2 | CLI grouped (show/admin/service/workload/prompt/provider/mcp) | Mechanical | P5 explicit | ~20 flat verbs → legible structure |
| 42 | DX-API | Renames: `probe → parallel_slots`; `mark()` takes Outcome enum | Mechanical | P5 | pre-release breaking safe |
| 43 | DX-6 | `somm init [--strict] [--with-service] [--with-mcp-client=X]` scaffold | Mechanical | P1 | TTHW collapse |
| 44 | DX-6 | `somm mcp install --client=claude-code\|cursor\|windsurf\|codex` | Mechanical | P1 | manual JSON = adoption death |
| 45 | DX-7 | `SommProvider` protocol + entry-points registration | Mechanical | P1 | no-fork extensibility |
| 46 | DX-4 | Docs tree (quickstart + errors/* + recipes/ + mcp/ + plugins/ + upgrading/) | Mechanical | P1 | findability |
| 47 | DX-5 | `DeprecationWarning` format + `somm upgrade-check` + CHANGELOG | Mechanical | P5 | fear-free upgrades |
| 48 | DX-8 | `somm admin dx-report` local-only; no beacon | Mechanical | (sovereignty) | respects user's no-telemetry-home stance |

---

# Cross-phase themes

**Theme 1: Privacy-first defaults, enforced at multiple layers.**
- CEO-S1: samples OFF by default
- Eng-S6: privacy_class gate expanded to recommendations.evidence_json + eval_results + export
- Eng-S4: SQLite file perms 0600
- DX-3: canonical error SOMM_WORKLOAD_UNREGISTERED explicitly names privacy context
Same concern, multiple enforcement points = defense in depth. High-confidence signal.

**Theme 2: Package-boundary discipline as enabler.**
- Eng-A1: `somm-core` shared package
- Eng-A2: MCP unified through core
- DX-6: `somm init` scaffold pulls everything together
- DX-6: `somm mcp install` uses somm-core as the MCP server entry
Without `somm-core` the DX patches would leak into version-skew hell.

**Theme 3: Sovereignty-first throughout.**
- CEO reframe: no commercial dependencies on hot path
- Eng: GitHub Actions as affordance, local CI authoritative
- Eng: self-hosted service lifecycle (systemd/launchd/Docker); no cloud option
- DX: no beacon; `dx-report` local-only
- Design: admin panel binds to localhost-only by default
Consistent posture across every surface.

**Theme 4: Complexity accepted, legibility enforced.**
- CEO kept the 10-star scope intact against both models' narrowing recommendation
- Eng broke the package-count-same-complexity equation by introducing somm-core (more packages, less fragility)
- DX grouped the CLI verbs (same commands, legible structure)
- Design pruned web admin to single column + 4 charts + rec list (dense but legible)
The plan chose "large-but-clear" over "small-and-simple." That choice compounds across phases.






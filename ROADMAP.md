# somm roadmap

What is next, what is deferred, and what is not planned. Items move here from
`docs/intel-sources.md` and from usage findings in adopting projects as they
mature from "idea" to "sized and ready to build." When an item ships it moves
to `CHANGELOG.md` and leaves this file; when it turns out to be a bad idea it
moves to **Parked** with a one-line reason.

Nothing here is committed to a release date.

## Where we are

0.8 → 1.0 built the sensor, the plugin loop, the workload as a versioned
unit, evals, the recommendation brain, and the public release. 0.14–0.16
added the harness layer, async clients, workload handlers, and sticky pins.
0.17 (October 2026) re-grounded against a census of the real fleet: the loop
became a command you can run, harness attempts entered the ledger, the
estate's raw-SQL rollups got a supported read API, and the provider layer
was brought onto one error ladder and the current model lineup. See the 0.17
entry in `CHANGELOG.md` for the findings that drove it.

November 2026 audited the code, the consumers, the session logs and the field
together (`VISION.md`, "Re-grounding, November 2026"). 0.20.2 fixed the
hot-path failures it reproduced. The sequence below comes from that audit and
takes precedence over the older sections that follow it; an older item that a
phase absorbs is named there.

## The November sequence

Each phase has a graduation test. Later phases do not wait on earlier ones
where they are independent, but nothing in phases 3–6 is trusted until phase 1
graduates.

### 1. A truthful record

- Persist `bad_json`, `timeout`, `rate_limit` and reasoning-budget exhaustion
  as outcomes; an "empty" row where reasoning consumed the whole output budget
  says so.
- Record the route on every row: first choice, served choice, fallback depth,
  reason, effective `max_tokens`.
- Project-qualified workload identity, with a read-side alias for old rows.
- Persist decision recalls; tag and exclude fixture rows.
- One price table; cache reads and writes priced at their own rates; an
  unknown cost is `NULL` with a stated source, never zero.
- The registry as the authority for ledgers: list, adopt, merge, retire.
- The data boundary: an ordered tier per route, a ceiling per workload,
  enforced at every path that leaves the process and recorded on every call.
- The Claude CLI provider: no session persistence by default, no silent
  default to the most expensive model, lean gold calls.
- **Graduation:** `somm doctor --global` reports no sentinel rows, no unknown
  cost without a source, one row per attempt including failures, and the three
  north-star numbers.

### 2. The kernel

- A ledger boundary: `somm-core` owns all SQL; a supervised writer; partial
  indexes; retention. A registry-driven fleet query replaces the full calls
  mirror, while workloads and decisions stay global.
- One pooled transport with connect, read, total and stall timeouts and one
  error classifier.
- One dispatch engine: admission (boundary, capabilities, budget reservation),
  a planner (pins, policy, `somm/auto`), an attempt loop with per-model health.
  Streaming is the primitive and `generate` collects it.
- Absorbs "Structure" below. `somm.__all__`, the `SommLLM` signatures,
  `somm_core.read` and the MCP and HTTP contracts stay fixed; the schema stays
  additive.
- **Graduation:** `stream()` and `generate()` share one pipeline and the
  consumers' test suites pass unchanged.

### 3. The paved road

- Structured output as the default JSON path: a typed result carrying
  provider, model and call id; shared repair; a prompt-schema fallback; a
  raised error on failure. Absorbs "Output-contract adoption".
- A route per call and per workload (`providers`, `exclusive`, `avoid_local`)
  with no private attributes; an explicit default route per project instead of
  local-first for everyone.
- Typed failure exceptions, a per-call `timeout=`, `client.health()`.
- `somm_core.read` extended to what downstream tools query by hand, and a
  versioned vocabulary export.
- `somm fleet status` and `somm fleet upgrade`: writer versions, dead
  editable paths, pins. Absorbs "Adoption on this fleet".
- **Graduation:** three consuming projects delete their wrapper modules.

### 4. Surfaces

- Done: the proxy on `SommLLM`, `/v1/responses`, streaming on chat
  completions, LiteLLM removed from `somm-service`. Left: incremental proxy
  streams (the client's `stream()` takes no `messages`/`tools`) and passing
  `stop`/`top_p`/`top_k` through, which the proxy accepts and reports as ignored.
- The CLI as a package with one flag grammar and a documented exit-code
  contract.
- `somm mcp install` and a distributed skill.
- OpenTelemetry GenAI export corrected and pinned to a convention version;
  agent and tool spans for harness attempts.
- A golden-fixture suite for harness adapters from recorded real streams. The
  ACP normalizer waits for ACP v2.

### 5. Kinds and placement

- A call has a kind (generate, decide, embed, rerank, handler, harness,
  external run), a meter unit, and a typed result with label, probability and
  abstention.
- `decide()` backed by decision-model providers.
- Calibration and label-agreement graders; abstain and escalate as workload
  policy.
- Aggregate placement observations, so decisions made by rules are counted
  without a row each.
- Cross-kind cost-of-pass and a climb report per workload. Absorbs
  "Cost-of-pass as the frontier metric".
- Rented GPUs: import the runner's receipts as cost, an amortised cost kind,
  and a lease-file recipe for a self-run endpoint as a provider. Somm does not
  launch or tear down.

### 6. A loop that closes

- Statistical guards and an expiry on recommendations; atomic grading leases;
  judge hygiene; a cheaper grader; human labels for calibration; observed
  quota truth; cascade routing with confidence-based escalation. Absorbs the
  "Recommendation quality" section and "Observed quota truth".
- **Graduation:** a recommendation is applied on a real fleet because the
  evidence said so.

### 7. The explanation

- A site that leads with the meter and the memory, defines its terms, shows
  each component with one diagram and one real number, and argues the case for
  building this instead of adopting a gateway and a tracing product — including
  where adopting them is the right answer.
- Version, tool and test counts generated from code.

## Next

### Adoption on this fleet

The 0.17 census found the ledger was being written by 0.14 and 0.15 clients.
No code change in somm fixes that; the work is in the consumers.

- Move every registered project to `somm>=0.17` so `client_version`,
  `call_site`, cost provenance, and `reasoning_tokens` stop being `NULL` on
  the busiest rows.
- `somm loop install` on the project whose in-process loop stopped on
  2026-09-19 (three opted-in workloads, 21 recommendations never actioned),
  then on the rest; watch the per-ledger lines in `somm doctor --global`.
- fab: consume `HarnessResult.model` / `retry_after_s` / the new outcomes,
  pass `record=True`, and drop its own process launch now that `start()`
  honors `env` and `start_new_session`.
- Consumers that re-read the ledger to learn which model served a parse
  should take it from the result (`extract_structured(with_result=True)`,
  `SommResult`); fleet dashboards and downstream ledgers read through
  `somm_core.read`.
- **Graduation:** the fleet section of `somm doctor --global` shows one
  client version, no sentinel rows, and a recent worker success on every
  ledger with shadow configs.

### Output-contract adoption

`workloads.output_schema_json` has existed for several schema versions and,
at the time of writing, **3 of 214 workloads fill it** — 50 of them with more
than 100 calls, including all ten of the busiest. The column is not the gap;
adoption is. Without a declared output contract nothing downstream can decide
whether a fixture could falsify a workload's result, so held-out grading,
eval promotion, and placement assessment stay guesses.

- **Graduation:** every workload above the traffic floor declares an output
  schema, and a grader can be built from the declaration rather than from a
  sample of outputs.
- **Boundary:** somm reports the gap and never fabricates a schema from
  observed responses — an inferred contract that looks declared is worse than
  an honest absence.

### Harness receipts, continued

Envelope 2 shipped the fields; what remains is conformance and memory.

- A golden-fixture suite that runs every adapter against captured real
  streams and asserts the envelope, so a CLI's weekly release cannot silently
  change a termination or a usage field.
- Model + harness outcome memory: record harness identity and version beside
  model, workload, permission profile, termination, and cost, and let the
  sommelier compare (model, harness) pairs when the workload is comparable.
  **Graduation:** at least one recommendation changes because observed
  harness fit beats a model-only prior.
- ACP (Agent Client Protocol) as a fifth normalizer: one `session/update`
  stream with `usage_update{cost}` covers Claude Code, Codex, OpenCode, Gemini
  CLI, Goose, and Pi through their adapters, and would de-risk per-harness
  schema churn. Sized L; promoted when a second runner besides fab wants it.

### Observed quota truth for metered seats

`plans.toml` is guesses with `enforce = false`. Claude Code's statusline and
SDK expose `rate_limits.{five_hour,seven_day}`; Codex's app-server exposes
`account/rateLimits/read`; Ollama cloud has `/api/usage`; OpenRouter `/key`
returns `limit_remaining`. Read the measured windows into pacing and let
`somm plans --learn` prefer them over observed-429 inference.

## Recommendation quality

### Cost-of-pass as the frontier metric

`somm frontier` ranks by score and latency. Dollars per correct answer
(cost ÷ judge pass rate, per workload) is the number a routing decision
actually turns on, and the frontier moves fast enough that recommendations
should carry an expiry.

### Judge hygiene

Pairwise judges have systematic position bias and style bias dominates
content. Run both orderings and log per-judge consistency; strip markdown
before judging; prefer a cheap multi-family panel to one expensive judge;
qualify a model on a seed set before it joins a panel.

### Active shadow sampling

Uniform sampling spends most of the gold budget on calls that teach nothing.
Sample by surrogate uncertainty (disagreement, outcome class, novelty) and
report eval spend against routing savings so the break-even is visible.

### Cascade routing mode

Learned routers converge to the same band; cascades (cheap first, escalate on
estimated quality) are the one policy that beats it. The judge panel is the
estimator; the policy is a workload setting. Sized L.

### Quality intel from redistributable sources

Arena's leaderboard dataset is CC-BY-4.0 on Hugging Face (24 subsets
including webdev and agent); Epoch AI's benchmark CSV is CC-BY. Both can
feed `capabilities_json` as a prior when shadow-eval is empty. Blocker: the
display-name ↔ model-id canonicalization that `somm alias` now gives us a
place to put. The HF Open LLM Leaderboard is retired and LiveBench's data is
stale; neither is planned.

### Honor catalog expiry

OpenRouter's `/models` carries `expiration_date`; models.dev carries `status`
and `last_updated`. The sommelier should stop recommending a model its
catalog says is going away.

### Bayesian prior weighting

Prior decisions are a decayed multiplier today. Treat each as evidence that
updates a belief per (model, workload); shadow-eval contributes real
observations. Needs a workload-similarity notion first.

### `outcome_note` → structured enum

Negative-outcome detection is keyword matching on free text. Add
`outcome_status` (`reliable`, `unreliable`, `capable_but_slow`,
`not_capable`, …) beside the note; keyword matching fires only when it is
null.

## Structure

- `somm/cli.py` (3,300 lines) into a `cli/` package along its existing
  section banners; `somm_core/repository.py` into mixins with one transaction
  helper; `sommelier.py` into advise / consult / decisions;
  `somm_service/app.py` into security / api / dashboard / otlp. Mechanical,
  and each is a PR on its own.
- `compare` exists in the CLI and the MCP server with drift between them
  (`allow_fallback`); one `somm/compare.py`.
- MCP server onto the `mcp` 2.x SDK and the 2026-07-28 stateless spec
  (`FastMCP` → `MCPServer`, `_meta` traceparent, `/server-card`).

## Deferred (sized but not scheduled)

- **A/B routing** — recommendations become live traffic splits with lift
  calculation. (~2–3d.)
- **`somm.ensemble(prompt, models=[…], aggregate=fn)`** — parallel-model call
  primitive. (~2–3d.)
- **Auto-eval generation** — a frontier model writes rubrics from sampled
  pairs. (~2d.)
- **Dedicated tool-call telemetry columns** — lift `tool_calls_count`,
  `tools_offered_count`, `stop_reason` out of `raw_json` once a query pattern
  needs them.
- **Streaming tool calls** — reassembling argument deltas matters for
  low-latency UX, which agent loops rarely have.
- **Ollama depth** — `/api/show` capabilities and thinking levels, `/api/ps`
  context length, cloud-tag pricing from models.dev.
- **Per-workload cache-efficiency report** — `cache_read / total_input` and a
  prefix-churn flag, per provider TTL; the tokens are already recorded.
- **Windows service lifecycle**; **packaged installers**; **provider-specific
  tokenizers** as `somm[tokenizers]`.
- **Web admin design** — recommendation evidence drawer, richer filtering,
  dark-mode polish.

### Principles for promoting deferred items

- If a deferred item becomes load-bearing for a real user demand, promote it.
- If it can ship as an optional extra (`somm[X]`) without bloating core,
  prefer that over blocking a release.
- If it would be a days-of-work surprise to someone building it themselves,
  promote it earlier.

## Parked

- **`SOMM_SCHEMA_STALE`** — never raised. Older clients writing into a newer
  ledger is how a fleet upgrades; the doctor reports version skew instead.
- **Proposal systemd units for shadow-eval and agent** — never installed.
  Replaced by `somm loop install`.
- **HF Open LLM Leaderboard / HF trending** — the leaderboard retired in
  March 2025; DOM scraping was fragile anyway.
- **Vendoring OpenRouter's catalog** — their 2026 terms bar copying; it is
  fetched at runtime only. models.dev (MIT) is the vendorable source.
- **Release-feed model-intel sources (RSS/Atom)** — most feeds are dead.

## What we are *not* planning

- A composite "quality score" that collapses Arena, Epoch, and shadow-eval
  into one number. Each signal measures something specific; the reasons list
  surfaces which one contributed.
- Continuous API polling. Intel is a cache; the advise path is fast and
  deterministic.
- Paid or gated sources on the default path. Anything needing a paid key
  lives behind a feature flag. Sovereignty-first.
- A hosted multi-tenant SaaS, a durable job runner, or a software-factory
  control plane. Fab and George own queues, retries, worktrees, verification,
  release, approval, and project intent.

# somm roadmap

Post-0.2.x work, in rough priority order. Items move here from
`docs/intel-sources.md` and from usage findings in adopting projects
as they mature from "idea" to "sized and ready to build."

Nothing here is committed to a release date. The goal is to keep the
shape of the future visible without letting the short list grow
unbounded.

## The 0.8 → 1.0 arc

A directed push from "proven sensor, inert brain" to a fully-realized
v1, in six phases. Three are shipped.

- **0.8 — Stop the bleeding (shipped).** Scheduler correctness, web-admin
  auth + CSP, cooldown admission control, hot-path connection reuse,
  prompt→call binding, worker-heartbeat truth, telemetry integrity, CLI
  honesty, MCP hardening.
- **0.9 — The Loop (shipped).** A named-phase hook bus (`pre_call` /
  `post_call` / `post_process`) with priorities and sync/async support;
  the `somm.providers` entry-point registry; four reference plugins
  (cache, redaction, notifier, OTel exporter); `somm plugin` CLI and
  `docs/plugins.md`.
- **0.10 — The Workload (shipped).** The workload as a versioned atomic
  unit: prompt label layer + forking, weighted-label A/B, the
  `somm prompt` CLI, `generate_structured()`, trace/cache/citation
  telemetry columns, workload config revisions, and per-workload routing
  policy. Schema v10→v15, migration engine made crash-atomic.
- **0.11 — The Proof (shipped).** Evals + the closed loop: graders shared
  between the background worker and a `somm eval run` CI gate, a durable
  datasets table, a binary-rubric LLM judge, eval→selection wiring, and a
  propose-only prompt optimizer.
- **0.12 — The Brain (shipped).** Close the intelligence loop: recommendation
  delivery + one-command apply, plan/quota learning, sommelier ranking
  quality (model-id aliasing, decision-aware scoring, score breakdown),
  and cross-project decision recall in `advise`.
- **1.0 — Table stakes + launch (shipped).** Session/trace UI, OTLP ingest, a
  supported JSON read API, an optional Postgres backend, and the public
  release.

Cross-cutting 1.0 work shipped the hot-path/import performance budget,
the threat-model doc, CI supply-chain checks, the one-shot
`somm generate` CLI, and the installable `somm-skill` package.

## After 0.14 — make the harness substrate portable and learnable

The reusable Claude/Codex/OpenCode harness layer is shipped. The next arc is to
make one attempt dependable for every outer runner without pulling Fab's
persistent supervision into Somm.

### Attempt receipt conformance

Define and fixture the fields every supported harness must return: actual
provider/model/harness and versions, session id, ordered normalized events,
final output, usage, artifacts exposed by the harness, and portable termination
reason. Preserve the raw native coordinate for debugging, but never require a
caller to parse provider-specific streams.

- **Graduation:** the same conformance suite passes for Claude, Codex, and
  OpenCode, with unavailable fields explicit rather than guessed.
- **Boundary:** retry policy, liveness, worktrees, verification, repair,
  release, and human escalation remain in Fab or another outer runner.

### Capability and permission profiles

Expose what each harness can actually do — resume, structured events, usage,
mid-run input, permission modes, and artifact references — as a versioned
capability record. Translate one caller intent into native flags and report the
profile actually granted.

- **Graduation:** callers can make a deterministic admission decision without
  harness-name conditionals.
- **Safety:** a missing or unknown permission mapping fails closed; it never
  silently widens access.

### Model + harness outcome memory

Record harness identity and version beside model, workload, permission profile,
tool count, termination, cost, and caller-supplied outcome correlation. Let the
sommelier compare model/harness pairs only when the workload and outer policy
are sufficiently comparable.

- **Graduation:** at least one recommendation changes because observed harness
  fit beats a model-only prior, with the supporting calls visible.
- **Non-claim:** a Fab retry/repair success is not automatically an attempt
  quality win; caller outcomes remain separately typed.

### Event-envelope stability

Version the normalized event envelope and publish additive-change rules so Fab
and other runners can upgrade independently. Keep the minimal stable event set
small; native detail stays in an opaque payload or raw artifact.

- **Graduation:** an older consumer can read a newer additive receipt, and
  golden fixtures catch semantic drift in termination and usage.
- **Rejection:** no Somm-owned workflow graph, approval queue, verifier, or
  operator dashboard grows out of this envelope.

## Make workloads joinable and gradeable

Schema 23 landed the first half: `calls.call_site` records which code asked for
each call, so telemetry joins to a static audit instead of describing the same
fleet in an incompatible vocabulary. `somm doctor` now reports capture coverage
and the workloads running without a declared output contract. What follows is
the adoption work that turns those from columns into answers.

### Output-contract adoption

`workloads.output_schema_json` has existed for several schema versions and, at
the time of writing, **0 of 166 workloads fill it** — 35 of them with more than
100 calls, including all ten of the busiest. The column is not the gap;
adoption is.

This is not cosmetic. Without a declared output contract nothing downstream can
decide whether a fixture could falsify a workload's result, so held-out
grading, eval promotion, and any placement assessment stay guesses. The doctor
warning is the nudge; the work is declaring contracts for the workloads that
carry real traffic, starting from the top of that list.

- **Graduation:** every workload above the traffic floor declares an output
  schema, and a downstream grader can be built from the declaration rather than
  from a sample of outputs.
- **Boundary:** somm reports the gap and never fabricates a schema by inferring
  one from observed responses — an inferred contract that looks declared is
  worse than an honest absence.

### Call-site attribution through adapters

The default capture records the innermost frame outside somm, which for code
reaching somm through an adapter is the adapter, not the logic that chose the
workload. That is enough to identify a repo and a module, and it is not enough
to identify the decision. `set_call_site_provider()` is the escape hatch; the
work is installing it where the indirection is real, and knowing where that is.

- **Graduation:** trailing-24h `call_site` coverage is high enough that a
  missing site is a finding rather than the norm, and the projects that route
  through adapters report the site they mean.
- **Boundary:** capture stays cheap (0.62 µs measured) and never raises into
  the call path. If attribution needs interprocedural analysis, it belongs in
  the auditing tool, not in the meter.

### A workload's shape as a first-class question

Placement assessment (`chip.placement`) reads somm telemetry to decide whether
a workload is a standing observation relationship or a batch job, routine
judgment or one-shot extraction, an envelope problem or a judgment problem. The
measures it needs — daily activity, peak concentration, distinct inputs per
call, dominant failure motif — are all derivable today, and a full-fleet daily
rollup over 1.1M rows takes 2.3 s on the existing indexes, so no materialized
view is warranted yet.

The open question is whether somm should expose those measures as a supported
read surface, or leave every consumer to write the same aggregation. Deferred
until a second consumer exists — one caller is not an interface.

- **Promotion trigger:** a second tool needs the same rollup.

## Recommendation quality

### Cross-provider model-id canonicalization

The same weights show up under multiple identifiers — `google/gemma-3-27b-it:free`
(OpenRouter), `gemma3:27b` (Ollama), `google/gemma-3-27b-it` (HuggingFace).
The sommelier treats these as three independent candidates today, which
both inflates the list and hides the "you already have this locally"
tradeoff. An alias layer would fix both.

- **Shape:** a `model_aliases` table `(canonical_id, alias_id, source)`,
  seeded from curated mappings plus HF's `model-index`/`base_model` links.
  Advise output merges candidates that share a canonical, surfacing
  "also available on ollama, 0 tok/s latency" as a reason.
- **Complication:** tier suffixes (`:free`, `:nitro`) actually change
  behavior (rate limits, routing). Keep them as distinct candidates but
  render as siblings of the canonical, not strangers.

### Bayesian prior weighting

Today's 0.2.2 prior-decision signals are a decayed multiplier. The next
step is treating each prior as evidence that updates a probability
estimate over "this model works for this workload." Same ranking math
as shadow-eval scores, different input.

- **Shape:** per-(model, workload, signal-source) belief state. Priors
  contribute pseudo-observations; shadow-eval contributes real ones.
- **Complication:** needs a workload-similarity notion so `critique_visual`
  priors can inform `captioning_dataviz` without being identical.

### outcome_note → structured enum

Current negative-outcome detection uses keyword substring matching on
free-text. Works today because we wrote the keywords ourselves; will
drift as more projects record decisions. Promote to a small enum
(`reliable`, `unreliable`, `capable_but_slow`, `not_capable`, …)
alongside the free text.

- **Migration:** additive column `outcome_status` + keep `outcome_note`.
  Keyword-match only fires when `outcome_status` is null (backfill path).

### Ranking taste surface

The score is a composite; callers can't tell if gemma outranks minimax
because of price, capability confidence, or a prior decision. Add an
optional `score_breakdown: dict` on `Candidate` so UIs and agents can
surface *why* without re-deriving the math.

## New intel signals

Each signal below is tracked in `docs/intel-sources.md` with coverage
notes. Roadmap entries here focus on what lands in the sommelier when
the signal is live.

### LMArena (quality Elo)

Broad-task quality signal, weaker than shadow-eval for your specific
workload but better than nothing when shadow-eval is empty.

- **Worker:** weekly scrape → `capabilities_json.lmarena.elo`.
- **Ranking:** moderate weight, ignored when shadow-eval exists.
- **Blocker:** canonicalization of display name ↔ model id (see above).

### LiveBench (per-category quality)

Critical for differentiating "coding model" from "reasoning model" from
"creative-writing model." `AdviseConstraints.workload_category` would
select the relevant LiveBench column.

- **Worker:** monthly fetch, per-category columns → `capabilities_json.livebench`.
- **Ranking:** high weight when workload declares a matching category.

### Artificial Analysis (speed + price composite)

Overlaps OpenRouter on price/context but adds tokens-per-second, which
the sommelier has no signal for today outside shadow-eval latency.

- **Worker:** daily (if free API) or manual refresh.
- **Ranking:** light — fast wins ties.

### canirun.ai (local GPU feasibility)

The sommelier will happily recommend `llama-70b` to someone on a 16GB
laptop. A feasibility signal closes that gap.

- **Shape:** per-(model, hardware-class) boolean + expected tok/s.
- **Ranking:** hard filter when the route is local; ignored for API.
- **Blocker:** user-declared hardware class (no reliable auto-detect).

### HuggingFace leaderboard scores

Open LLM Leaderboard composites (MMLU/GPQA/HellaSwag/ARC-C/TruthfulQA/Winogrande)
are orthogonal to both LMArena and LiveBench.

- **Worker:** weekly → `capabilities_json.hf.leaderboard`.
- **Ranking:** light weight; surfaces as a reason tag, not a score bump.

### SWE-bench (coding-specific)

Narrow task, slow refresh, but the single best proxy for "does this
model actually solve real GitHub issues."

- **Ranking:** high weight when workload declares `coding`; ignored
  otherwise.

## Surface improvements

### `somm advise` CLI

Today the only way to call `advise()` outside a library or the MCP is
via the MCP itself. A thin CLI command would make the ranking
inspectable from the shell.

### MCP `somm_compare`

Given two (provider, model) pairs, surface the full score breakdown
and prior-decision history side by side. Useful for "should we switch
off X to Y?" conversations.

### Prior-decision schema expansion

`recorded_by_tool` field so the sommelier can trace which agent/tool
recorded a decision. Useful when multiple assistants share the global
repo.

## Deferred (sized but not scheduled)

Larger items deliberately deferred; each is promotable when a real
user demand makes it load-bearing.

### Core product

- **A/B routing** — agent recommendations become live shadow traffic
  splits with lift calculation. Currently the agent only recommends;
  no closed loop. (~2–3d.)
- **`somm.ensemble(prompt, models=[…], aggregate=fn)`** —
  parallel-model call primitive for ensembling. (~2–3d.)
- **Auto-eval generation from production samples** — a frontier model
  writes grading rubrics from sampled call pairs; builds eval suites
  automatically. (~2d.)

### Telemetry

- **Dedicated tool-call telemetry columns** — once tool-call workloads
  accumulate real calls, decide whether to lift `tool_calls_count`,
  `tools_offered_count`, `stop_reason` out of `raw_json` into
  indexable columns. Don't migrate preemptively — wait for the query
  pattern.
- **Streaming tool calls** — providers stream tool-call arguments as
  deltas; reassembly is non-trivial and matters mainly for
  low-latency UX, which agent loops don't typically have. Open an
  issue when a project asks.

### Infrastructure

- **Postgres backend** for small-team shared deployments as an
  optional `somm[postgres]` extra. SQLite remains default.
- **Windows service lifecycle** support (Task Scheduler integration).
  Linux + macOS first.
- **HF trending model-intel source** — behind a feature flag;
  OpenRouter is the primary source. Fragile DOM scraping.
- **Release-feed model-intel sources** (RSS/Atom per-provider) — most
  are dead; feature-flagged.
- **Provider-specific tokenizers** as `somm[tokenizers]` extras
  (tiktoken, etc.). Default approximation (4 chars/token) ships today.

### DX / packaging

- **`somm plugin` command** — install/list/remove plugins (providers,
  graders, etc.) with supply-chain checks. Currently pip-based.
- **Packaged installers** (.dmg / .deb) — pipx/uv tool install works
  today.
- **Opt-in beacon telemetry for DX measurement** — local-only
  reporting is the default; any beacon stays opt-in.

### Web admin design

- Recommendation evidence detail panel (deep drawer/modal).
- Richer dashboard filtering/search beyond per-project toggle + time
  window.
- Dark-mode polish (light mode is currently first-class).

### Principles for promoting deferred items

- If a deferred item becomes load-bearing for a real user demand,
  promote it.
- If it can ship as an optional extra (`somm[X]`) without bloating
  core, prefer that over blocking a release.
- If it would be a days-of-work surprise to someone trying to build it
  themselves (plugin protocol, extensibility), promote it earlier.

## What we are *not* planning

- A composite "quality score" that collapses LMArena + LiveBench + HF
  into one number. Each signal measures something specific; the
  reasons list should surface which signal contributed.
- Continuous API polling. Intel is a cache; the advise path is fast
  and deterministic.
- Paid/gated sources on the default path. If a source requires a paid
  key, it lives behind a feature flag. Sovereignty-first.

## Keeping this file honest

When an item ships, move it to `CHANGELOG.md` and delete the roadmap
entry. When an item turns out to be a bad idea, move it to a "Parked"
section with a one-line reason — the "why we didn't do X" note is
useful to future you.

# somm — design blueprint for people building their own

This document is for anyone implementing their own version of somm —
in a different language, with a different feature set, or just a
smaller subset. It's not a porting tutorial. It's a distillation of
the design-forcing decisions: what somm *is* at its core, and what
makes each constraint load-bearing.

If you take away the wrong piece, the whole thing loses coherence.
If you take away the right piece, what's left is still useful.

Read alongside [`PLAN.md`](../PLAN.md) for the full context. This
document is the compressed version.

---

## The one-sentence version

**somm is a per-project SQLite file that every LLM call in a codebase
writes one row to, plus the smallest set of services needed to make
that file useful.**

Every other feature — routing, shadow-eval, recommendations, MCP,
sommelier — is a consumer of that single substrate. Build the
substrate first; the rest composes on top.

---

## The six forces

Every somm decision trades these off against each other. If you skip a
force, you'll end up with a subtly different product.

### 1. Sovereignty by default

A user can `pip install somm`, run one local ollama, and have a fully
functional system. No API keys, no cloud account, no account at
`somm.dev`. Commercial providers are opt-in env vars.

*Why it's load-bearing:* the product's thesis is that your telemetry
is *yours*. The moment a default path requires an external account,
the thesis breaks. Hobbyists and enterprises alike need the same
zero-config entry point.

### 2. Privacy as defense-in-depth

Prompt bodies are not stored by default. Private workloads are
filtered *in the router, in the shadow worker, and in a SQL view*.
Files are `chmod 0600`, dirs `0700`. The web admin binds `localhost`.

*Why it's load-bearing:* privacy that depends on one guard is one
mistake from a leak. Every defense should fail closed independently.
When porting, don't consolidate the guards — that's a lock you can't
rekey.

### 3. Write-once, amend-never telemetry

The `calls` table is immutable after insert. Late-arriving metadata
(outcome tags from `result.mark()`, judge scores) goes in
`call_updates`, not back into `calls`. This makes audit trivial and
guards against "the row I saw yesterday isn't the row today."

*Why it's load-bearing:* the intelligence loop (shadow eval → agent
recommendations) depends on a stable view of history. A row that
mutates mid-analysis gives non-deterministic recs.

### 4. Content-addressed entities where meaning is stable

Workloads are `hash(name + schemas)`. Prompts are `hash(body)`.
Decisions have a `question_hash` on the normalised natural-language
question. Calls are UUID4 (they're *events*, not content — same
prompt can produce different calls).

*Why it's load-bearing:* content addressing makes dedup free,
cross-project joins free, and "did this change?" trivially detectable.
UUIDs for events means you don't need coordination to write.

### 5. Every guarantee must work under load

SQLite in WAL mode + `busy_timeout = 5000` + `synchronous = NORMAL`.
When the writer queue gets `SQLITE_BUSY` too many times, it spills to
a per-process JSONL spool. `drain_spool` replays it on next
opportunity. Mirror writes never block the primary path.

*Why it's load-bearing:* a library that drops telemetry on disk
pressure is a library nobody trusts. And the spool has to be durable
(flush on every row) or it isn't a guarantee. **Never swallow a call
silently.**

### 6. Failures are first-class

Every provider error bucket has a `SOMM_*` code. Every canonical error
follows:
```
SOMM_<NAME>
Problem: one sentence
Cause:   one sentence
Fix:     concrete commands
Docs:    link
```
Transient vs fatal is encoded in the exception hierarchy (routing
falls through on transient; raises on fatal). Cooldowns survive
process restarts so an overnight-flaky provider doesn't get
re-hammered at dawn.

*Why it's load-bearing:* users debug library bugs via error messages
alone — nobody reads source first. If `SOMM_AUTH` and `SOMM_429` are
visually indistinguishable, you've taught users that errors are
fungible; they'll start catching `Exception`.

---

## The data model

Ten tables, all in one SQLite file. Everything else is a view or a
derived rollup.

### Events (immutable)

- **`calls`** — one row per `generate()`. The only mandatory write
  path. Fields: ids, timing, cost, outcome, hashes.
- **`call_updates`** — late-arriving metadata on a call (outcome marks,
  audit trail).
- **`samples`** — prompt/response bodies, *per-workload opt-in*.
  Separate table so the default path never touches prompt text.
- **`eval_results`** — shadow-eval grades on a call. Has a
  `grading_started_at` lease for crash-safe worker recovery.

### First-class entities (content-addressed)

- **`workloads`** — `hash(name + schemas)`. Carries `privacy_class`,
  `budget_cap_usd_daily`, `capabilities_required`.
- **`prompts`** — `hash(body)`. Versioned per workload.
- **`model_intel`** — pricing + context + capabilities per
  (provider, model). Refreshed from provider APIs.
- **`decisions`** — sommelier memory. Always cross-project mirrored.
- **`recommendations`** — agent-emitted advice. Has `dismissed_at` +
  `applied_at` for state.

### Operational

- **`provider_health`** — per-(provider, model) cooldown entries.
  `model=""` is the provider-wide circuit breaker.
- **`jobs`** — scheduler state (atomic lease pattern).
- **`worker_heartbeat`** — liveness for `somm doctor`.
- **`schema_version`** — applied migrations (simple monotonic int).

The critical relationships:
- `calls.workload_id → workloads.id` (nullable — demo-mode
  auto-registers)
- `eval_results.call_id → calls.id`
- `samples.call_id → calls.id`
- `decisions.workload_id → workloads.id` (nullable)

---

## The substrate loop

```
  library ──► calls table ──► model_intel
     │            │                  │
     │            ▼                  │
     │       eval_results            │
     │            │                  │
     │            ▼                  ▼
     │       recommendations ◄── agent worker
     │
     └─► decisions table (cross-project)
```

Every arrow is one-directional. The library writes calls; the service
reads calls, produces eval_results + recommendations. The sommelier
reads model_intel + decisions, writes decisions. Users never modify
eval_results or recommendations directly — they dismiss/apply them,
which is its own state transition.

**If you're porting a minimal viable somm, skip eval_results and
recommendations entirely.** You can reintroduce them later; they
compose cleanly on top of calls + model_intel.

---

## Non-obvious decisions

These are the ones that bit us during development. Copy them.

### Cooldowns in SQLite, not memory

Every implementation's first instinct is `dict[str, datetime]` for
per-provider cooldown state. Don't. Free-tier providers have outages
that span overnight; a Python process restart at 6am shouldn't
re-hammer a provider that's been rate-limited since 11pm. Persist to
SQLite with the same schema you use for everything else.

### Two-tier cooldowns

Provider-wide (`model=""`) AND per-model entries. OpenRouter's free
roster has ~5 models; when one 429s you want to try the next model,
not walk away from OpenRouter entirely. The per-model cooldown is the
adapter's concern; the provider-wide cooldown is the router's.

### Circuit breaker on the provider, not the model

After N consecutive failures on the *provider-wide* entry, cool the
whole provider for a longer window. This matters because a
misconfigured API key produces N model-level failures that *look*
transient but are actually fatal. The circuit breaker catches the
pattern even when individual errors don't.

### Empty ≠ failure

A model that returns an empty response successfully is tagged
`outcome=EMPTY` but the provider is marked `mark_ok`. The model had
nothing to say; the provider worked. Conflating these causes
spurious cooldowns on fine-tuned models that legitimately refuse to
answer off-task prompts.

### Observe vs strict mode

Default mode auto-registers unknown workloads with a warning. Strict
mode raises. You need both: observe for TTHW (time-to-hello-world) and
strict for production safety. Most projects ship `observe` in dev and
flip to `strict` via env var in prod.

### Per-workload privacy class

Projects have heterogeneous workloads. Treating privacy as a global
switch forces the user to pick either "everything captured" or
"nothing captured" — neither fits a real codebase. Per-workload
privacy lets claim-extraction egress freely while customer-data-parsing
never does.

### Spool is JSONL, not a binary format

When SQLite is down, you can still hand-inspect the spool with `cat`.
Recovery is `for row in file: write(row)` — the same code that writes
fresh calls. Don't optimise the spool format; optimise the recovery
code.

### Decisions are cross-project by default

Calls are per-project (privacy). Decisions are cross-project
(advisory memory). This is the one asymmetry in the data model and
it's deliberate: the *value* of a decision is exactly its
portability. "In project A we picked gemma-3 for vision" is useful
evidence in project B.

### Cost calculation reads from cache, not the provider

When a call returns, cost is computed from `model_intel`, not from
provider response metadata (which is inconsistent across providers
and sometimes missing). If intel is missing, cost is `$0` and a
warning fires once. The agent worker refreshes intel on a cadence.

### The agent worker emits *evidence*, not models

Recommendations carry `evidence_json` with the numbers that produced
them. The UI shows "switch to X: +45% quality, -80% latency (47
shadow calls)". A recommendation without provenance isn't actionable.

### Providers are swappable, not fixed

`SommProvider` is a Protocol, not a base class. Anyone can ship a
custom provider via entry points. This is why OpenAI / OpenRouter /
Minimax all share one HTTP base class — the abstraction is at the
wire shape, not the identity.

---

## What to stay out of

Explicit non-goals. If you're tempted to add these, read twice.

- **Hosted mode.** There is no `somm.dev`. The moment we add cloud
  sync, we become the kind of service we exist to replace.
- **Usage beaconing.** somm will never phone home. The project has
  zero visibility into installs — and that's a feature, not a bug.
- **Opinionated eval rubric.** Shadow-eval uses simple structural +
  text-similarity scorers. Judge-LLM grading is opt-in because
  rubrics are workload-specific. Don't bake in a "quality score" that
  doesn't fit someone's domain.
- **Workload-prescribed prompts.** somm versions prompts; it doesn't
  author them. The substrate is blind to whether a prompt is "good."
- **Auto-apply recommendations.** The agent emits, the user applies.
  An auto-applying recommendation engine is how you ship a regression
  at 3am.

---

## What to keep minimal if you're reimplementing

If you want a 500-line port in Go/Rust/TypeScript:

1. One table: `calls`. UUID id, ts, project, workload_name,
   provider, model, tokens, cost, outcome, prompt_hash.
2. Two adapters: OpenAI-compatible + local (ollama or nothing).
3. A `generate()` that writes one call row.
4. A `stats()` that aggregates the table.
5. A JSONL spool for writer pressure.

That's a real somm. Everything else you can defer:
- Cooldowns + circuit breaker (Routing layer)
- Prompt versioning (Prompts table)
- Shadow-eval (eval_results + worker)
- Agent recommendations (recommendations + worker)
- MCP (separate transport over the same Repository)
- Sommelier (decisions + advise/record tools)

Add them in that order as needs arise. Each layer's surface area fits
in a single file.

---

## Vocabulary

- **workload** — a task ("extract_contacts"), not a call
  ("call_anthropic"). Stable across time. Snake_case.
- **call** — one `generate()` invocation. UUID4 id; not
  content-addressed because events aren't content.
- **provenance** — the `(call_id, provider, model, workload)` tuple
  stamped on stored data so you can answer "what model wrote this?"
- **shadow-eval** — re-running a sampled call through a gold model to
  get a quality signal without the user writing a rubric.
- **gold model** — the best model you have on hand; used as a
  reference for shadow-eval grading.
- **cold-start** — recommending without shadow data. The sommelier's
  job.
- **decision** — a sommelier conversation outcome. Always
  cross-project mirrored.

---

## Schema-evolution etiquette

- Migrations are forward-only. Never write a migration that
  down-migrates — you can't trust users to have backups.
- Additive changes (new columns, new tables) are safe; renames and
  drops are a release negotiation.
- Version files live in `migrations/NNNN_<name>.sql`. `NNNN` is
  zero-padded four digits. Each file is one transaction.
- `SCHEMA_VERSION` in `version.py` is the highest migration that
  must exist on disk. Bump it in the same commit as the migration
  file.

---

## Why we're sharing this

If you build your own and it's better in some way, tell us. The
space of "self-hosted LLM telemetry" is underpopulated and somm is
one shape of the answer. Sovereignty-forward substrates benefit from
a plurality of implementations, not a monoculture.

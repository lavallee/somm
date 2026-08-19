# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
`somm` uses a single unified version across all workspace packages
(`somm`, `somm-core`, `somm-service`, `somm-mcp`, `somm-langchain`,
`somm-skill`).

## [0.16.0] — 2026-08-19

### Changed — BREAKING: a pinned `provider`/`model` no longer falls back silently

Naming a `provider` and/or `model` on a call now means **serve it with that or
fail**. Previously a pin was "try this first": when the pinned target errored,
somm rescued the call through the router chain, dropped the pinned model name,
and answered from whatever provider was healthy. The result carried the
substitute's attribution, so a caller who chose a model could get — and be
billed for, and record telemetry against — a different one, with nothing in the
return value saying the choice had been overridden.

That default was the wrong trade. Silent substitution is unexplainable in
production ("why does this workload sometimes cost 40× more?"), and it quietly
invalidates every measurement somm exists to produce: `somm bench`, `somm eval`,
the `somm_compare` and `somm_replay` MCP tools, and any panel judge grading on
named models. Pins are now sticky everywhere by default.

On a failed pin, the call returns `outcome=UPSTREAM_ERROR` with the **pinned**
`(provider, model)` preserved on both the `SommResult` and the telemetry row,
and `error_detail` names the escape hatch.

Opting back in, for callers who would rather have an answer from *some* model
than none (batch workers that must not lose a run when one provider drops):

- `generate(..., allow_fallback=True)` — also on `agenerate`,
  `generate_structured`, `agenerate_structured`, `extract_structured`,
  `aextract_structured`, and as `somm bench --allow-fallback`.
- `SOMM_PINNED_FALLBACK=1` / `Config.pinned_fallback` — process-wide, restoring
  the pre-0.16 default without touching call sites.
- `SommChatModel(somm_allow_fallback=True)` for the LangChain adapter.

Unaffected: calls with no `provider` pin route exactly as before, and an
explicit `model=` without a provider still rides unchanged across every
provider the chain tries — the chain has never substituted the model there.
Workload policy `fallback` chains are explicit configuration and keep working.

`no_fallback=True` is now the default and is kept as a deprecated alias. It can
only ever *force* stickiness: `no_fallback=False` does not re-enable rescue,
because the `no_fallback=bool(args.provider)` idiom in existing call sites meant
"don't rescue", never "please substitute". Use `allow_fallback=` for that.

## [0.15.0] — 2026-08-18

### Added

- **`reasoning_effort` on every generate path** — DeepSeek v4 accepts
  `none|minimal|low|medium|high|xhigh|max` on flash as well as pro and spends
  reasoning tokens accordingly, which somm could not express and whose
  capability map said the opposite of what the API does, classifying anything
  matching `-flash` as non-thinking by name. The budget now travels on
  `SommRequest` and reaches the OpenAI-compatible payload only when set, so
  providers that do not know the field never see it. Families whose reasoning is
  a request dial rather than a property of the name are listed separately and
  are never filtered out of a thinking workload: what decides is the effort on
  the request, not the suffix on the model. `SommLLM.generate`, `agenerate`,
  `generate_structured`, `agenerate_structured`, `extract_structured` and
  `aextract_structured` all take it, and `somm eval run --reasoning-effort`
  exposes it, which is the point — a model cannot be compared against itself at
  different budgets if the runner has no way to say which budget it wants.
- **`somm eval` (`promote-call`, `import`, `run`)** — a durable eval dataset
  built from what already happened rather than from fixtures. `promote-call`
  copies a sampled production call into a dataset, `import` takes reviewed
  JSONL prompt/expected-response pairs, and `run` replays a workload against
  the dataset and records the result as an eval run.

- **Public `call_updates` recording API (`somm_core.Repository`)** — the
  `call_updates` table (late-arriving metadata about immutable `calls` rows)
  gains its first general write surface: `record_call_update(call_id, field=,
  value=)` appends one typed update row, and
  `record_call_updates_for_correlation(correlation_id, field=, value=,
  include_children=True)` stamps an update onto every call attributed to an
  external unit of work via `correlation_id` (hierarchical
  `<id>:...`-namespaced descendants included by default, e.g. fab's
  `<job_id>:attempt:<idx>`), returning the linked call ids. The existing
  `record_outcome_update` now delegates to `record_call_update`. This is how
  downstream systems (fab's bridge, for one) link a job's terminal semantic
  outcome back to the calls that served it.
- **`somm doctor --max-exhausted-rate FRACTION`** — trailing-24h
  `SommProvidersExhausted` rate check, suitable for a cron/timer. `somm
  doctor` always prints the rate; the flag makes doctor exit nonzero above
  the threshold. Added after finding a month-long exhaustion storm (July
  2026, one adopter project: 90%+ call failure for a week) had no automated alert — see
  `docs/incidents/2026-07-provider-exhaustion.md`. A proposed (not
  installed) systemd timer for it lives in `deploy/systemd/`.

- **Workload handlers (`somm.workloads`)** — a workload is a unit of work, not
  necessarily an LLM call. Every workload now has a **kind**; the default is
  `llm` (the provider chain, unchanged), and a project may register other kinds
  and bind workloads to them:

  ```python
  from somm import workloads
  workloads.register(MyHandler())
  workloads.bind("triage", "my-kind", some="config")
  ```

  A handler implements `kind` and `serve(WorkloadRequest) -> WorkloadResult |
  None`. Returning `None` declines and the provider chain answers exactly as
  before; a handler that raises is treated as one that declined, so binding a
  handler to a live workload cannot make things worse than they already were.

  Handlers may be published from other packages through the new
  `somm.workload_handlers` entry-point group, loaded on demand. **somm depends
  on no particular kind** — the first non-LLM kind, `chip`, is published by fab.

  A handler-served call records the kind as its `provider` and source, so it sits
  in the same `calls` table as model-served calls for the same workload id and
  the two are separable by a `GROUP BY`. That is the point: "should this workload
  still be an LLM call?" becomes a query rather than an argument, and a workload
  can climb from a plain LLM call toward deterministic behavior on evidence.
  See `docs/workload-handlers.md`.

  Dispatch runs at `pre_call` priority 50 — after redaction, which must see the
  outbound text first, and after the response cache, since a cached answer is
  cheaper than any handler. That wiring is somm's own; a handler author never
  touches hooks.
- **Async embeddings**: `SommLLM.aembed()` makes the existing instrumented
  embedding path awaitable without caller-side thread wrappers, preserving
  sync telemetry and explicit model pins. Non-Ollama-first provider
  configuration now fails loudly on the async path instead of being silently
  overridden by the currently Ollama-only implementation.
- **Native async client API**: `SommLLM.agenerate()`, `astream()`,
  `agenerate_structured()`, and `aextract_structured()` let async applications
  use Somm without caller-side thread wrappers while preserving the existing
  synchronous routing, fallback, plan-governor, hooks, and telemetry path.
- **Honest local-cost provenance**: native Ollama calls now record
  `unknown/included/unknown` with source `local-included-unpriced` rather than
  presenting the backward-compatible numeric `$0` sentinel as computed
  marginal economic cost. Token telemetry remains intact for a future local
  resource-rate calculation.
- **Procedure outcome custody**: `record_procedure_outcome` writes a replay-safe
  Somm eval receipt carrying exact Milton/Chip/Spindle origin, explicit
  baseline and promoted implementation/profile/model/harness tuples, the Fab
  execution receipt, and the baseline plus post-promotion call/score. Both
  arms retain native Somm call IDs; Somm records the measurement while the
  external calibration consumer owns interpretation.
- **Endpoint-driven OTLP HTTP export**: installing `somm[otel]` and setting
  `SOMM_OTEL_ENDPOINT` to a full traces endpoint now activates the existing
  GenAI `post_process` span hook through Somm's plugin entry point. An unset
  endpoint remains a dependency- and thread-free no-op, while plugin-owned
  providers are flushed and shut down without taking ownership of manually
  supplied tracer providers.

### Changed

- **MiniMax M3 is the default hosted model and the first provider tried.**
  `gemma4:e4b` is no longer the default Ollama model in the client signature
  and MiniMax moves from priority 40 to 10, so a configuration that named no
  provider now reaches a hosted model first rather than a local one. The
  MiniMax default model moves M2.7 → M3. Projects relying on the previous
  order should set `provider_order` explicitly.

### Fixed

- Classify Codex's exact cybersecurity-filter termination as a refused harness
  attempt so supervisors can escalate it once instead of retrying an unchanged
  prompt as a generic failure.
- Reject known cross-provider model and harness combinations before launching
  a coding-agent process, while leaving unknown future model names compatible.
- Load the new outcome-custody exports lazily so importing `somm` remains
  within its startup budget, and require MCP 1.28.1 or newer to exclude known
  vulnerabilities in older releases.

## [0.14.0] — 2026-07-14

### Release status

- `0.14.0` remains beta/pre-1.0. The new harness API is public but may still
  evolve as additional task runners adopt it.

### Added

- **Reusable coding-agent harness API**: `somm.harnesses` executes one Claude
  Code, Codex, or OpenCode workspace attempt through a provider-neutral
  request, handle, capability, result, and outcome contract. It supports
  asynchronous outer supervision, synchronous one-shot execution, native
  session resume, safe-by-default permissions, JSON event normalization,
  final-text extraction, usage capture, and correlation IDs.
- **Explicit runner boundary**: harness execution remains independent of job
  queues, retries, watchdog policy, worktrees, verification, releases, and
  escalation, allowing Fab and other project runners to depend on Somm without
  introducing a reverse dependency.

- **Named key profiles** (`somm-core`): set `key_profile` in `[tool.somm]`
  (or `SOMM_KEY_PROFILE`) and provider keys resolve from
  `<NAME>_API_KEY_<PROFILE>` before falling back to `<NAME>_API_KEY`,
  AWS_PROFILE-style. Lets one environment hold several provisioned keys
  per provider (per product, team, or deploy stage) so provider
  dashboards attribute spend per profile. No profile set → behavior
  unchanged.

## [0.13.0] — 2026-07-11

### Release status

- `0.13.0` remains beta/pre-1.0. The release expands the service/proxy
  surface and serving telemetry, but does not declare stable public API, MCP
  contracts, service contracts, or migration semantics.

### Added

- **Serving performance rollups**: workload/provider/model stats now include
  p50/p95/p99 latency, p50/p95/p99 TTFT, TPOT, input/output/total token
  throughput, request throughput, prompt-cache read/write tokens,
  cache-read ratio, and goodput against workload SLOs for `max_p95_latency_ms`,
  `max_p95_ttft_ms`, and `max_tpot_ms`.
- **Benchmark CLI**: `somm bench latency` and `somm bench throughput` run
  normal instrumented calls and summarize latency, TTFT, TPOT, request
  throughput, and token throughput.
- **Prefix-cache advice**: `somm cache-advice` flags high-input-token
  workload/provider/model rows with low cache-read ratios.
- **Native structured-output hints**: `generate_structured()` now passes
  JSON Schema hints through provider requests for adapters that support
  native guided JSON generation while retaining validation and retry checks.
- **Service load metrics**: `/api/status` now includes one-minute call,
  token, failure, and active workload/provider/model counters.
- **Service proxy compatibility**: `/v1/messages` now supports Anthropic
  `stream: true` SSE responses, and `/v1/chat/completions` adds an
  OpenAI-compatible non-streaming chat gateway. Both paths use the same
  workload budget gate, timeout, body limit, auth, and `calls.sqlite`
  telemetry ledger.
- **Spend read API**: `/api/spend/today` returns current UTC-day spend by
  workload for the service project, including effective daily budget caps.

### Changed

- CI now exposes the release-version gate as a standalone `release-gate` check
  so branch rulesets can require the 0.x/1.0 guardrail directly.

## [0.12.0] — 2026-07-10

### Release status

- `0.12.0` replaces the mistakenly published `1.0.0` release. somm remains
  beta/pre-1.0: the system is usable and published, but its public API, MCP
  tools, service contracts, and migration semantics are not yet being declared
  stable.
- Added a CI and PyPI-publish release gate that blocks future `1.x` versions
  unless the repo contains an explicit `docs/release/ONE_DOT_ZERO_GO_DECISION.md`
  readiness decision.

### Added — the brain and beta launch surface

- **Recommendation inbox apply loop**: CLI, MCP, and web actions can list,
  apply, and dismiss routing recommendations. Applying a recommendation now
  writes a workload policy revision, records the decision, marks the
  recommendation applied, and mirrors the decision to the global store when
  configured.
- **Plan/quota learning**: `somm plans --learn` derives observed ceilings from
  quota errors, records learned limit metadata in `plans.toml`, and lets the
  plan governor block providers whose learned limits are exhausted.
- **Model aliasing and canonical ranking** (schema v19): `model_aliases`
  canonicalizes equivalent model IDs across providers, dedupes sommelier
  candidates, carries score breakdowns, and applies prior decisions against
  the canonical model rather than one spelling.
- **Cross-project sommelier recall**: MCP-recorded decisions from one project
  can be cited by `somm_advise` in another project through the shared decision
  store.
- **Service read API and trace ingest**: the service exposes supported JSON
  status, calls, and sessions endpoints plus authenticated OTLP trace ingest
  for `gen_ai` spans. The dashboard also gained no-JS filters, session/trace
  depth, and recent-call listings.
- **Adoption CLI**: `somm status --json` emits a stable machine-readable
  status envelope, and `somm generate` provides a one-shot generation command
  with JSON success and error envelopes.
- **Postgres packaging extra**: `somm[postgres]` installs the Postgres driver
  dependency for shared deployments.
- **Security and performance gates**: CI now runs `pip-audit`, `zizmor`, and a
  performance budget script covering sub-30ms top-level import and sub-1ms
  warmed hot-path p50. CLI executors use a private per-call temp cwd, and
  parse/spool fuzz coverage exercises malformed input paths.
- **Service guardrails**: dashboard/read APIs require bearer auth or the
  same-origin loopback dashboard path by default; `/health` no longer exposes
  local paths; proxy and OTLP ingress enforce body/span caps; proxy dispatch
  runs off the event loop with service-controlled timeouts; MCP compare has
  fanout/token caps; adaptive parameter bumps require explicit auto-heal opt-in.

### Added — evals and the closed loop (Phase 3)

- **Shared grader module**: structural JSON overlap, text-similarity, and
  judge helpers now live in `somm_core.graders`, reused by the shadow worker
  and synchronous eval paths.
- **Durable eval datasets** (schema v16): `datasets`/`dataset_items`
  promote sampled production calls into explicit golden fixtures. CLI:
  `somm eval promote-call <call_id> --dataset NAME`; MCP:
  `somm_eval_promote_call`.
- **`somm eval run`**: runs a workload against a durable dataset, records
  `eval_results` + receipts, reports pass/fail, and exits nonzero below the
  threshold so it can serve as a CI gate.
- **Binary-rubric judge tier**: opt-in `ShadowConfig.judge` supports
  per-criterion binary judging and cheap multi-judge panels before frontier
  judges; judge cost is included in budget accounting.
- **Eval receipts and pairwise grading** (schema v17): structured
  `eval_receipts` rows back dataset runs, shadow judge receipts, and
  pairwise A/B grading without overloading `judge_reason`.
- **Eval→selection wiring**: sommelier candidate ranking now learns from
  judge/dataset eval scores using the same precedence as prompt promotion.
- **`somm optimize`**: propose-only prompt optimizer that reads failing
  graded calls, asks an LLM for a complete replacement prompt, forks a new
  immutable version, and labels it `proposed` without moving production.
- **Campaign harness** (schema v18): `somm campaign run` repeats durable
  dataset evals under a metric contract, records append-only keep/revert
  JSONL-shaped events, and stops on max rounds, token budget, or plateau.

### Added — the workload as an atomic unit (Phase 2)

- **Prompt label layer + forking** (schema v11): mutable named pointers
  (`production`/`staging`/`latest`) over the immutable content-addressed
  versions. `set_label`/`get_label`/`get_prompt(label=...)`; rollback is a
  pointer move recorded in append-only `prompt_label_history`; `fork_prompt`
  records lineage via `prompts.parent_prompt_id`.
- **Weighted-label A/B** (schema v14): a label can point to a weighted
  distribution; `SommLLM.prompt(label=, bucket_key=)` buckets
  deterministically (defaulting the key to the ambient correlation id, so a
  session sees a stable variant) while traffic splits by weight. `prompt_id`
  binding records which variant served each call.
- **`somm prompt` CLI**: `list/show/register/fork/diff/label/promote/score`.
  `promote` gates a label move on N graded calls above a mean score;
  `score` reports per-version eval rollups — closing the eval→promotion loop.
- **`generate_structured(prompt, schema=...)`**: returns
  `(validated_object, SommResult)` with corrective retry. Accepts a Pydantic
  v2 model, a JSON Schema dict, or a callable validator (`somm[pydantic]` /
  `somm[jsonschema]` extras). Raises `SommStructuredError` on exhaustion
  rather than a sentinel dict. Absorbs four fleet reimplementations.
- **New telemetry columns** (schema v12): `ttft_ms` (streaming
  time-to-first-token), `session_id`/`parent_call_id` (trace hierarchy),
  `cache_tokens_in`/`cache_tokens_out` (Anthropic/OpenAI prompt-cache usage),
  and `citations_json` (grounded-search citations). Best-effort capture that
  never affects routing.
- **Workload config revisions** (schema v13): constraint/shadow/policy
  changes append a snapshot to `workload_revisions` (the live row stays the
  router's fast-read path); `workload_revision_diff` and forward-only
  `rollback_workload`.
- **Per-workload routing policy** (schema v15): a workload carries its own
  declarative fallback chain, retry/backoff/deadline, and request timeout
  (`policy_json`), consumed by `generate()` when the caller doesn't override.
  The no-policy path is byte-identical. Absorbs the fleet's hand-rolled
  `generate_resilient`/pin-registry/custom-timeout wrappers.

### Added — the plugin loop (Phase 1)

- **Named-phase hook bus**: `somm.hooks` grows three lifecycle phases —
  `pre_call` (synchronous; may rewrite the request or return a
  `ShortCircuit` to serve the call without a provider — cache / guardrail
  territory), `post_call` (observe-only, inline), and `post_process`
  (observe-only, dispatched to a background worker so graders/exporters
  add no caller latency). `register_hook(phase, fn, priority)` with
  WordPress-style integer priority; sync + async hooks; every hook is
  exception-isolated and gets its own copy of the event; events carry
  `schema_version=1`. `add_call_observer`/`notify_call_observers` keep
  working. `somm.hooks` and `somm.plugins` entry-point groups load once
  per process.
- **Hooks fire on the call path**: `generate`/`stream`/`embed` fire all
  three phases. The no-hooks path is byte-identical and allocates nothing
  extra (guarded by an O(1) `has_hooks` check). A short-circuit records a
  normal telemetry row and, because it spends nothing, is allowed even for
  a workload at its daily budget cap.
- **Provider entry-point registry**: the long-advertised `somm.providers`
  entry-point group is now real — built-ins are `ProviderSpec`s and third
  parties register a spec (skipped-with-warning on import failure,
  built-ins winning name collisions). The default chain, `SOMM_PROVIDER_ORDER`,
  and `full=True` behavior are unchanged.
- **Four reference plugins** (`somm.plugins`, opt-in via `register()`):
  a response cache (`pre_call` short-circuit + `wrap(llm)` to populate),
  outbound PII/secret redaction (`pre_call`), a Slack/webhook notifier
  (`post_process`), and an OpenTelemetry span exporter (`post_process`,
  behind the new `somm[otel]` extra).
- **`somm plugin list` / `somm plugin info`** show reference plugins,
  active hooks, and the provider picture. `docs/plugins.md` is the
  plugin-author guide (phases, priorities, event schema, registration,
  worked examples, custom providers).

### Fixed — the scheduler's clock

- **Scheduler timestamp bug**: `due_at`/`locked_until` were written in
  SQLite's space-separated format but compared against Python
  `isoformat()` strings — lexicographically, `' ' < 'T'`, so any
  same-day future job read as due *now*. Same-day jobs re-fired every
  poll tick, leases never excluded concurrent runners within a day,
  and comparisons flipped at UTC day rollover. All comparisons now run
  in SQL with `datetime()` normalization (legacy-format rows still
  parse).
- **`worker_heartbeat` is real**: the table was never written, so the
  dormant-loop warning fired even while workers ran. Every scheduler
  tick, job execution, and manual `somm-serve admin` run now beats;
  the warning adds a staleness variant; `somm doctor` shows the table.

### Security

- **Web service auth**: `POST /v1/messages` (which spends real provider
  money) and recommendation dismiss/apply now require a bearer token
  (generated `0600` at `.somm/service_token`; `SOMM_SERVICE_TOKEN`
  overrides) or same-origin + `X-Somm-Local: 1` custom-header evidence.
  Previously any web page could `fetch()` these routes cross-origin —
  localhost binding does not stop CSRF-class requests. HTML responses
  gain a Content-Security-Policy; all responses gain nosniff/referrer
  headers.
- **MCP trust boundary**: recorded prompt/response bodies returned by
  `somm_replay`/`somm_compare` are wrapped in explicit
  untrusted-content envelopes (capped, truncation-marked) so a
  poisoned recorded response can't steer the calling agent unlabeled.
- **Secret scrubbing**: `calls.error_detail` redacts common credential
  shapes (OpenAI/Anthropic `sk-*`, AWS, GitHub, Slack, Google, generic
  api-key/bearer assignments) before persisting upstream error bodies.

### Added

- **Prompt→call binding**: `generate()`/`stream()`/`extract_structured()`
  accept `Prompt` objects, and plain-string prompts hash-match against
  the workload's registered versions (prompt ids are content-addressed),
  stamping `calls.prompt_id`. Grades and costs can finally attribute to
  prompt versions.
- **Wait-with-deadline on exhaustion**: `generate(..., wait=<seconds>)`
  (or `SOMM_WAIT_ON_EXHAUSTED`) sleeps with jitter until the routed
  chain's earliest cooldown expiry instead of instantly raising
  `SommProvidersExhausted` — cooldown storms stop producing failure
  storms. Default stays fail-fast, and the router's old implicit
  sleep-up-to-300s-without-opt-in is gone.
- **`somm workload add/list/show`**: workload registration from the
  shell, with `--from-example` templates — the command strict-mode
  error hints always promised. Every CLI hint now references a command
  that exists (regression-tested against the real parsers).

### Changed

- **Cross-project mirror on by default**: the local-only replication of
  calls/workloads into `~/.somm/global.sqlite` was opt-in and
  effectively dead (and auto-registered workloads were never mirrored
  at all — fixed). Env-resolved configs now default it on;
  `SOMM_CROSS_PROJECT=0` disables. Nothing leaves the machine.
- **Telemetry integrity**: spool files auto-drain at writer startup and
  every 10 minutes (rename-claimed, safe under concurrent drainers);
  the fleet registry self-prunes dead entries and refuses pytest tmp
  paths; projects run from a fresh cwd resolve their registered DB
  instead of fragmenting telemetry into a new `.somm`.

### Performance

- **Hot path**: `Repository` reuses one SQLite connection per thread
  (PRAGMAs once, fork-safe) instead of opening ~5 fresh connections per
  call; model prices are cached in-process for 10 minutes; registry
  writes throttle to once per process. Full-suite wall time dropped
  ~40% from connection reuse alone. somm-core gains its first dedicated
  test suite, including a zero-new-connections-per-call regression test.

## [0.7.1] — 2026-07-03

### Fixed — in-process workers no longer require the web stack

`start_inprocess_scheduler` moved to `somm_service.inprocess`
(workers-only import weight); importing it via `somm_service.app`
dragged in starlette, which broke SOMM_INPROCESS_WORKERS=1 in
environments that have the workers' deps but not the web server's.
Old import path re-exported for compatibility.

## [0.7.0] — 2026-07-03

### Added — online-eval sample capture (the missing half of the loop)

Shadow grading needs prompt/response bodies, but nothing ever captured
them — candidates piled up ungradeable. The library now captures
bodies at call time for workloads that opted into shadow eval (the
documented consent for body storage): deterministic per-call sampling
at the workload's `sample_rate` (applied once, at capture — the worker
no longer re-samples), never for `privacy_class=private`, oversized
bodies (inline images) skipped rather than truncated, and capture can
never break the call path. The shadow worker now considers only
captured candidates, so historical body-less calls age out instead of
churning into "samples not captured" results.

## [0.6.1] — 2026-07-03

### Changed — first PyPI release

Published to PyPI: `pip install somm` (+ `somm-service`, `somm-mcp`,
`somm-langchain`, `somm-core`, `somm-skill`). Packaging fixes for the
occasion: exact lockstep pins on inter-package dependencies
(`somm-core==X`), absolute links in package READMEs (relative links
404 on PyPI), and per-package trusted-publishing environments
(`pypi-<package>`) in the publish workflow — PyPI requires pending
publishers to be unique per (repo, workflow, environment). README
install instructions now lead with pip.

## [0.6.0] — 2026-07-03

### Added — payg budgets, burn rates, plan value, quota drift

- **PAYG budgets**: payg plans may now declare limits — self-imposed
  spend ceilings over calendar or rolling windows (LiteLLM-style
  budget + duration), denominated in real dollars. Same pacing math,
  same governor semantics: over-pace deprioritizes the provider,
  `enforce = true` hard-stops spending at the ceiling.
- **Burn rates**: `somm plans` reports per-provider real-dollar
  velocity (1d/7d/30d, smoothed $/day, projected month-end) — for
  PAYG there is no vendor window, so rate is the number that matters.
- **Plan value**: metered plans with `price = N` show the value
  multiple — notional list-price consumed this month vs what the
  subscription costs.
- **Quota drift detection**: declared quotas are guesses/marketing
  copy and vendors reset them without notice. `somm plans` warns when
  (a) usage exceeds a declared quota while calls keep succeeding (real
  limit is higher), or (b) recent 429-derived observed ceilings
  diverge >25% from the declared quota (limit moved).
- `LimitStatus.mode`, `BurnRate`, `payg_burn_rates`,
  `recent_ok_calls`, `Plan.price_usd_month` exported from somm-core.

## [0.5.0] — 2026-07-03

### Added — plan catalog + observed ceilings (keeping quota data honest)

Research finding that shaped this: as of mid-2026 **no major vendor
publishes a complete numeric limits table** for subscription plans
(MiniMax describes quotas as "agent counts", Anthropic as multipliers,
OpenAI hides its weekly cap), and limits change roughly monthly. somm
now handles that with three layers:

- **Bundled plan catalog** (`somm_core/data/plan_catalog.toml`, browse
  with `somm plans --catalog`): curated entries for known plans
  (MiniMax Token Plan tiers, Claude Pro/Max, ChatGPT Plus/Pro Codex)
  with per-entry `source` URL, `last_verified` date, and provenance
  notes that say plainly when a number is an estimate. plans.toml can
  reference an entry (`catalog = "token-plan-max"`) and inherit its
  limits; explicit `[[limits]]` always win; unknown references fail
  loudly.
- **Staleness warnings**: `somm plans` flags referenced catalog
  entries not re-verified in 90 days; RELEASING.md now requires
  re-verifying aged entries (and regenerating the pricing bundle)
  before each release.
- **Observed ceilings**: `somm plans` infers quotas empirically from
  your own telemetry — at each quota-429, trailing-window usage across
  the fleet approximates the binding limit; the median over
  burst-collapsed events is reported per window/unit. Ground truth
  that self-updates when vendors move the goalposts.

Historical provider-name spellings (`claude_cli`) are matched when
aggregating usage.

## [0.4.0] — 2026-07-03

### Added — billing plans: PAYG vs metered, with quota pacing

Providers bill in two fundamentally different shapes, and somm now
models both. **PAYG** (API keys): `cost_usd` is real marginal dollars.
**Metered** (coding plans, CLI seats): marginal dollars are ~0 inside a
recurring quota, `cost_usd` is notional list-price, and the scarce
resource is window headroom.

- `~/.somm/plans.toml` declares each provider's mode and, for metered
  plans, limits: calendar windows (`month` + `anchor_day`) or rolling
  (`5h`/`7d`/`1w`), quotas in `requests` / `tokens_*` / `usd_equiv`,
  a `soft_target_pct`, and optional `enforce`. Undeclared providers
  default: ollama → free, `claude-cli`/`codex-cli` → metered, API
  providers → PAYG. Malformed config fails loudly (pacing disabled
  with a warning) rather than silently ignoring a declared quota.
- **Fleet-wide accounting** (`somm_core.registry`): every
  `somm.llm()` init registers its DB in `~/.somm/registry.json`, so
  quota usage aggregates across all projects sharing an account —
  pacing computed against one project alone would understate burn.
- **`somm plans`** — per-limit usage in the current window with pace
  ratio (burn vs window elapsed) and straight-line projection;
  `--json`, `--project-only`.
- **Pace-aware routing** — the router consults a TTL-cached plan
  governor: over-pace providers are deprioritized (tried only after
  in-pace providers fail); an exhausted limit with `enforce = true` is
  skipped outright, and if every provider is blocked the call raises
  loudly with remediation hints. Pacing failures never take the call
  path down.
- **Plan-aware sommelier** — `somm_advise` candidates on metered
  plans carry a reason like "metered plan 62% used, pace 1.3x", so
  workload-model fit accounts for quota scarcity, not just price.
- One-line stderr warning (once per provider per process) when a
  metered plan is exhausted or past its soft target and over pace.

## [0.3.1] — 2026-07-03

### Changed — migration 0010 drops `calls.commission_id` (schema v10)

Completes the 0009 expand-contract. The dead column (all NULL, never
read) existed only so telemetry writers running pre-0.3 code survived
a migrated database; with long-running daemons restarted onto 0.3+,
the contract phase lands. Any straggler pre-0.3 writer fails its
INSERT and spills to the JSONL spool — recover with `somm drain-spool`
after restarting it.

## [0.3.0] — 2026-07-03

### Added — bundled pricing snapshot; cost tracking now works for every provider

`somm-core` ships `data/pricing_bundle.json` (~350 models, derived from
LiteLLM's community price file by `scripts/update_pricing_bundle.py`)
and a new `sync_bundled_pricing(repo)` that upserts it into
`model_intel` on library init. Existing databases get it too: rows
sourced from the old seed or a previous bundle are refreshed; manually
set or live-refreshed rows are never touched. A crc32 fingerprint in
`PRAGMA user_version` makes repeat syncs a no-op. Previously only a
handful of hand-seeded anthropic/openai models ever cost-tracked;
gemini, deepseek, minimax, perplexity, and openrouter paid models all
logged $0 forever. `_PAID_PROVIDERS` now covers all six paid providers
so missing pricing warns loudly.

### Added — `somm backfill-costs`

Recomputes `cost_usd` for historical calls logged at $0 whose
(provider, model) now has pricing intel, from the recorded token
counts. `--dry-run` reports without writing; `--since N` limits the
window. On a real production database this recovered ~$290 of
previously invisible spend in one command.

### Added — intelligence loop without a dedicated service

- `SOMM_INPROCESS_WORKERS=1` runs the somm-service scheduler (model
  intel refresh, online-eval grading, agent recommendations) inside
  the library process — no `somm serve` needed. Singleton per DB per
  process; requires somm-service installed.
- When online eval is configured but no worker has ever run, the
  library now prints a one-line stderr warning (once per DB per
  process) instead of letting sampled calls pile up ungraded silently.

### Added — `somm.hooks` extension surface

Neutral integration point replacing the previous hard-wired
soft-integration: `set_correlation_provider()` stamps an external id
(request/trace/job id) on every `calls` row, and `add_call_observer()`
receives an event dict after every generate/stream/embed call.
Integrations can also attach via the `somm.hooks` entry-point group.
Hook failures never break the call path. Schema migration 0009 adds
`calls.correlation_id`; the old `commission_id` column stays in place
(dead, all NULL) so telemetry writers still running pre-0.3 code keep
inserting against a migrated database — a future migration drops it
once no pre-0.3 writers remain.

### Fixed

- The MCP server and the shadow-eval worker now build the exact
  provider chain `SommLLM` builds (via the new
  `somm.client.build_default_providers`) — previously both wired only
  5 of 10 providers, so `somm_compare`/`somm_replay`/shadow grading
  could never reach gemini, deepseek, perplexity, or the CLI
  executors.
- `SommPrivacyViolation` (and the other user-facing error classes) are
  now exported from `somm` — `examples/private_workload.py` previously
  crashed with `AttributeError` at its own `except` clause.
- `SommStrictMode.code` is now `SOMM_WORKLOAD_UNREGISTERED`, matching
  the operator-facing message and docs page (was `SOMM_STRICT_MODE`,
  which matched neither).
- Local ollama 503 "server busy" (queue full) now cools for 5s instead
  of 30s — a momentary queue spike no longer instantly exhausts
  single-provider projects.
- Live-ollama tests skip embedding-only models and skip (instead of
  fail) when the server is contended.
- `somm-serve admin list-intel` empty-state hint printed a command
  that doesn't exist.

### Changed — docs & packaging

README rewritten around the full current surface (10 providers, 6
packages, tool calling / streaming / embeddings, full env-var table);
PLAN.md archived out of the repo with `docs/BLUEPRINT.md` as the
canonical design doc; TODOS.md folded into ROADMAP.md; 9 missing
`SOMM_*` error pages added; CONTRIBUTING/SECURITY/CODE_OF_CONDUCT,
issue/PR templates, ruff + Python-matrix CI, PyPI trusted-publishing
workflow, `py.typed` markers, and per-package READMEs added; all
package versions unified at 0.3.0 (PyPI publication deferred).

### Added — `somm-langchain` adapter package

New workspace package: `SommChatModel(BaseChatModel)` for LangChain /
LangGraph / `deepagents` apps to treat somm as their LLM substrate.

Thin adapter: extracts `SystemMessage`(s) into somm's `system` field,
translates the rest into somm-neutral messages (Anthropic-shaped),
unwraps OpenAI-form tool schemas to somm-neutral, calls
`SommLLM.generate()`, and returns a `ChatGeneration` whose `AIMessage`
carries `tool_calls`, somm provenance in `response_metadata`
(provider, model, latency, cost, stop_reason, call_id), and standard
LangChain `usage_metadata`.

`bind_tools()` supports the LangChain convention (tool_choice =
"auto" | "any" | "none" | "required" | "<tool_name>" | dict).
Failures surface as `RuntimeError` by default (so retry / circuit-
breaker middleware engages) or as an error-flagged `AIMessage` when
`raise_on_failure=False` for callers that prefer in-band error handling.

Driven by agent orchestrators running on deepagents. This is the
unlock that lets agent substrates go through somm.

13 new tests cover message translation in both directions, tool
binding, tool_choice variants, provenance metadata, and the error
modes.

### Added — Tool-calling (Anthropic + OpenAI-compat; agent substrate)

`SommRequest` gains `tools`, `messages`, `tool_choice`; `SommResponse`
and `SommResult` gain `tool_calls` and `stop_reason`. A new `ToolCall`
dataclass (`id`, `name`, `arguments`, `arguments_raw`) lives on
`somm_core` and is exported from the package init. All additions
default-safe — existing callers are byte-identical.

`AnthropicProvider` translates somm-neutral tool schemas to Anthropic's
`input_schema`, emits native `tool_choice`, and parses `tool_use`
content blocks back into `ToolCall` entries with `stop_reason` surfaced
for agent loops.

`OpenAICompatProvider` (covers `openai`, `minimax`, `openrouter`,
`deepseek`) wraps tools as `{type:function, function:{...}}`,
translates Anthropic-shaped multi-turn `messages` into OpenAI's
`tool_calls` + `role:tool` messages, and parses tool_calls back —
malformed JSON arguments surface as `arguments={}` with
`arguments_raw` populated rather than crashing the agent loop.
`finish_reason` normalizes to somm's neutral `stop_reason` vocabulary
(`stop→end_turn`, `tool_calls→tool_use`, `length→max_tokens`,
`content_filter→content_filter`; unknown values pass through).

`SommLLM.generate(tools=..., messages=..., tool_choice=...)` threads
the new fields end-to-end and surfaces `tool_calls`/`stop_reason` on
the returned `SommResult`. The router's empty-response gate now
treats a tool-use turn (empty `text` + non-empty `tool_calls`) as
valid — it would previously have been recycled as "transient empty"
and exhausted the chain.

Driven by agent orchestrators on `deepagents`, which mandate
tool-calling. Without this somm couldn't be the substrate for any
agent project. Full spec lives in
[docs/tool-calling.md](./docs/tool-calling.md).

### Added — Tool-calling rollout: Gemini, Ollama, capability seeding

Completes the provider sweep started above.

`OllamaProvider` gains tool support on `/api/chat` (Ollama 0.4+; the
box runs 0.20.x). Tools are sent in the OpenAI-shaped `tools` field and
multi-turn `messages` translate through the shared OpenAI translator
(Ollama mirrors that message shape). Response `tool_calls` parse back —
Ollama returns `function.arguments` already as a dict, so there's no
`json.loads` and no `arguments_raw` repair path. `done_reason` maps to
somm's neutral `stop_reason` (`tool_use` when tool_calls are present,
since Ollama reports `done_reason="stop"` even on tool turns). Ollama
has no `tool_choice` knob, so passing one raises `SommBadRequest` rather
than silently dropping it — the router then falls through to a provider
that can honor it.

`GeminiProvider` inherits tool support unchanged from
`OpenAICompatProvider`: Google's OAI-compat endpoint accepts standard
OpenAI `tools`/`tool_choice` and returns `tool_calls` in OpenAI shape,
so no native `functionDeclarations` adapter is needed. The stale "use
the native endpoint for function calling" note in `gemini.py` is
corrected.

Capability seeding for the `tools` capability: `model_has_capability`
gains a name-hint `tools` branch (Claude 3+, GPT-4+/5+, Gemini 1.5+,
Llama 3.1+, Qwen 2.5+, DeepSeek-Chat, Mistral/Mixtral, o1/o3/o4) and
the seeded `model_intel` frontier rows now declare `{"tools": true}`.
Unknown models still fall through as capable (None = allow); there is
no negative case — a provider that genuinely can't serve tools raises
at call time. This makes `capabilities_required=["tools"]` workloads
(deepagents orchestrators) route positively to tool-capable models.

`SommLLM.generate()` now hashes `messages` for `prompt_hash` when a
multi-turn call is made, instead of the ignored `prompt` placeholder —
so replay/cache/dedup keys off the real conversation.

With Gemini and Ollama landed, every shipping provider supports tools;
no adapter silently drops a `tools=` request — unsupported features
raise loudly. Dedicated tool-call telemetry columns and streaming tool
calls remain deferred by design (see ROADMAP.md).

### Fixed — WriterQueue drains pending calls on normal process exit

`WriterQueue`'s background thread is daemon, so Python's interpreter
shutdown killed it without giving it a chance to finish its current
batch — calls submitted in the last ~100ms of a script's lifetime
vanished entirely, not even spilled to the JSONL fallback (spill only
fires on a *drain failure*, not on writer-thread death).

`start()` now registers an `atexit` hook that flushes + stops the
writer on normal shutdown. atexit runs while the daemon thread is
still alive, so the flush completes cleanly. Idempotent against
explicit `close()` calls; tolerates partial-teardown errors silently
so it never blocks process exit.

This was caught in a sibling project's evaluation runs where
short-lived comparison scripts were losing telemetry rows.

### Added — `no_fallback` for pinned-or-bust evaluation runs

`SommLLM.generate(..., no_fallback=True)` suppresses the normal pinned-call
rescue path. When a `provider` is pinned and the upstream fails, instead of
silently routing to the next provider in the chain, the call returns with
`outcome=UPSTREAM_ERROR` and the *pinned* (provider, model) preserved on
the `SommResult` and `calls` row.

Driven by the same sibling-project finding that exposed the
adequacy-frontier gap: when running an A/B comparison between two models on
the same workload, the rescue path makes failures invisible — you see a
result tagged with the pinned model that was actually produced by the
fallback. This invalidates the experiment.

Default behavior is unchanged: production workloads still fall through to
the chain when the preferred provider transiently fails.

### Added — adequacy frontier per workload (schema v6)

Driven by sibling-project demand (a parsing workload where
some models we have at our immediate disposal struggle and the question
"is this model performing adequately, or should we go shopping?" was
hard to answer from telemetry alone).

- **`FailureClass` classification on `Outcome`.** Splits the existing
  outcome enum into capability signals (model unfit: `bad_json`,
  `off_task`, `empty`) vs. detractors (provider/network flaky:
  `timeout`, `rate_limit`, `upstream_error`). Available as
  `Outcome.failure_class`, `.is_capability_signal`, `.is_detractor`,
  and via the SQL view `v_calls_classified` for direct queries.
- **Three new workload constraint columns** (nullable; null = no opinion):
  `max_p95_latency_ms` (timeliness ceiling), `max_capability_failure_rate`
  (model-traceable failure ceiling, 0–1), `max_cost_per_call_usd`
  (cost ceiling). Set via `register_workload(...)` or
  `Repository.set_workload_constraints(...)`.
- **`Repository.workload_frontier(workload_id)`** — per-(provider, model)
  rollup with capability vs detractor counts kept separate, p50/p95
  latency over ok calls only, mean cost per ok call, and fitness flags
  per constraint. Sorted fittest-first (capability_failure_rate asc,
  then mean cost). Cleanly distinguishes "the model can't do this work"
  from "the free tier was rate-limiting today."
- **`somm frontier --workload NAME`** CLI — read-only adequacy view that
  surfaces `UNFIT(cap)` / `UNFIT(slow)` / `UNFIT($)` flags only when
  the workload has the matching constraint set. Default 30-day window.

What this doesn't do: subjective quality scoring is intentionally out
of scope (lives in `eval_results`, populated by shadow-eval). The
frontier answers timeliness + error consistency + payload-validity,
which is enough to make "let's go shopping for a better model" a
visible state in the data.

Migration 0006 is additive (new columns nullable, new view replaces no
existing object). Pricing seed, sommelier ranking math, and existing
queries are unchanged.

### Added — empty-outcome diagnostics
- **`error_detail` + `error_kind="EmptyResponse"` on the EMPTY outcome.**
  Previously every `outcome='empty'` row in `calls` had both fields
  blank; cross-project audits had to inspect prompt/response bodies to
  distinguish the two empirical empty modes — openrouter
  `{"content": null}` (model never ran, sub-500ms, `tokens_out=0`) vs
  minimax-style all-`<think>` output (full latency, `tokens_out>0`,
  stripped to ""). Each EMPTY row now carries a hint
  (`no_content` when `tokens_out=0`, `stripped_empty` otherwise) plus
  `out_tokens`, `latency_ms`, `provider`, `model`. SQL triage works
  without joining samples:
  `SELECT error_detail FROM calls WHERE outcome='empty'`. Same payload
  flows through the `on_error` alerter event — previously
  `kind`/`detail` were both `None` for empties.

### Fixed
- Streaming path (`SommLLM.stream`) was constructing `Call(...)`
  without `error_detail`, silently dropping the field for all stream
  outcomes. Now declared and threaded through.

## [0.2.2] — 2026-04-20

### Added — sommelier quality

Driven by findings from a sibling project's captioner-selection
report.

- **Output-modality filter** — `AdviseConstraints.required_output_modalities`
  drops candidates whose output modality isn't a superset of the request.
  Excludes audio-gen models with image inputs (Lyria et al) from a
  captioning (`output=["text"]`) query. Reads from OpenRouter's
  `architecture.output_modalities`, the scalar `modality` field
  (`"text+image->text"`), or the new HF enrichment — whichever is
  populated first.
- **Meta-router exclusion** — `openrouter/auto`, `openrouter/free`, and
  `openrouter/auto-*` variants are filtered by default. These models
  pick a backend at inference time, so they're non-deterministic and
  inherit capability claims from whatever they route to. Opt in via
  `include_meta_routers=True`.
- **Inline blocklist** — `AdviseConstraints.exclude_models` accepts
  fnmatch-style patterns against `"<provider>/<model>"`, so callers
  can drop a bad candidate without waiting for a release.
- **Unknown-capability penalty** — models with unknown (not
  known-lacking) capabilities score by `×0.9` per unknown capability,
  so confirmed `vision✓` outranks unconfirmed `vision?`. Tunable via
  `AdviseConstraints.unknown_capability_penalty`; `1.0` restores pre-0.2.2
  behavior.
- **Prior-decision weighting** — candidates whose `(provider, model)`
  matches a prior decision are annotated (`prior(<project> <date>):
  chose — ×1.10`) *and* score-nudged. Positive nudge for clean priors,
  negative for priors whose `outcome_note`/rationale contains
  unreliability keywords (`unreliable`, `failed`, `struggled`, …).
  Both factors decay with age via exponential half-life (~90 days).
- **Deterministic tiebreaker** — candidates tying on score now sort
  predictably by `(shadow_score desc, last_seen desc, model asc)` so
  two adviser runs agree.
- **Smarter empty-result note** — when the candidate list is empty the
  `note` surfaces which filter ate them (e.g. *"Filtered out: 3 wrong
  output modality, 2 meta-router"*), not just a generic "loosen
  constraints" hint.

### Added — intel sources

- **HuggingFace `pipeline_tag` worker** (`HuggingFaceIntelWorker`).
  Opt-in via `somm-serve admin refresh-intel --hf` or
  `SOMM_ENABLE_HF_INTEL=1`. Fetches `pipeline_tag` + `tags` from the
  HF Hub, maps tags (`image-text-to-text`, `text-to-speech`, …) to
  input/output modalities, and merges under `capabilities_json.hf`.
  Supplements OpenRouter rows where `architecture.output_modalities`
  coverage is spotty. Non-fatal on 404s and rate limits.
- **`merge_intel_capabilities(repo, provider, model, delta)`** —
  shared helper for layering supplementary signals onto
  `capabilities_json` without clobbering the primary-source fields.
  Primary workers (OpenRouter, Ollama) keep using `write_intel`;
  enrichment workers (HF, future LMArena/LiveBench) use the merge path.

### Added — library

- **`sommelier.consult()`** — returns `ConsultResult`
  (`candidates`, `prior_decisions`, `note`) so Python callers get
  parity with the MCP `somm_advise` tool without going through MCP.
  The MCP wrapper routes through `consult()` for a single code path.
- **Keyword-fallback prior-decision recall** — `consult()` retries
  `search_decisions` per content word when the exact-substring query
  misses, so slightly-reworded questions still recall priors.

### Changed

- **`somm_advise` MCP tool** — extended with `required_output_modalities`,
  `exclude_models`, `include_meta_routers`, and
  `unknown_capability_penalty`. Default behavior changes: meta-routers
  are now excluded (opt in for the old default) and unknown-capability
  rows score lower than known-yes.
- **`capabilities.model_output_modalities(repo, provider, model)`** —
  new helper returning the modality set a model can produce, or `None`
  when we have no signal. Used by the output-modality filter.

## [0.2.1] — 2026-04-20

### Added — providers
- **Gemini provider** via the OpenAI-compatible endpoint (`GeminiProvider`).
  Activates when `GEMINI_API_KEY` is set; default model
  `gemini-2.5-pro`. Joins the default provider chain and the
  capability-aware router like any other adapter.
- **`SOMM_PROVIDER_ORDER`** env var — comma-separated override for the
  default provider chain (e.g. `"openrouter,minimax,ollama"`).
- **Ollama ergonomics** — `enable_think` (sets `"think": true` on
  reasoning-capable models) and `keep_alive` (default `30m`, pinned
  residency window to stop mid-chain cold-start outliers). Configurable
  via `SOMM_OLLAMA_THINK` + `SOMM_OLLAMA_KEEP_ALIVE`. Ollama + minimax
  both bump `num_predict` / `max_tokens` 3x (with a 1024 floor) so
  thinking-token budget doesn't eat the visible reply.

### Added — error visibility
- **Schema v5** — `calls.error_detail` column: bounded (512-char)
  operator-friendly description of non-OK outcomes
  (`{ErrorClass}: msg | http_status=X | body=…`). Written on every
  failed call; surfaced in doctor / stats / MCP `somm_search_calls`.
- **`SommLLM(on_error=callable)`** — fires inline on every non-OK
  outcome with a context dict
  (`workload / provider / model / outcome / error_kind / error_detail`).
  Default handler writes a one-line warning to stderr so failures are
  visible without log plumbing. Pass `on_error=lambda _: None` to
  suppress, or wire to logging / Slack / PagerDuty.
- **`_format_error_detail`** walks `httpx.HTTPStatusError.response` to
  capture the server's error body alongside the exception — no more
  opaque `UPSTREAM_ERROR` rows.

### Added — out-of-the-box cost tracking
- `seed_known_pricing(repo)` fires on `SommLLM` init, so non-zero
  `cost_usd` lands on the first call without a manual
  `somm-serve admin refresh-intel`. Ships with current Anthropic
  4.5–4.7 family IDs (`claude-haiku-4-5-20251001`,
  `claude-sonnet-4-6`, `claude-opus-4-7`) alongside prior snapshots.

### Fixed
- **OpenRouter pricing sentinel** — treat the `"-1"` / `-1` value
  OpenRouter uses for dynamic-priced models as unknown pricing
  (`None`) rather than ingesting it as "negative one dollar."
  model_intel + `somm_advise` now filter those entries correctly.
- **Parse resilience** — four new fallback parsers in
  `extract_json()` handle LLM output with literal C0 control bytes
  mid-string (`_strip_control_chars` + `_flatten_whitespace`).
  `extract_balanced` retries against the stripped text before giving
  up.

### Added — docs
- **`docs/intel-sources.md`** — prospective model-intel sources for
  the sommelier ranker (LMArena Elo, Artificial Analysis,
  canirun.ai, LiveBench, Open LLM Leaderboard) with stability +
  refresh notes.

## [0.2.0] — 2026-04-19

### Added — sommelier: model advisor + cross-project decision memory
- **`somm.sommelier`** — new module that ranks (provider, model)
  candidates from `model_intel` against free-form constraints:
  capability tokens (hard filter), price ceilings, provider
  whitelist, `min_context_window`, `free_only` shortcut, and optional
  `workload` hook that boosts candidates with shadow-eval evidence.
- **Schema v4** — `decisions` table + `Decision` dataclass. A
  decision captures a question, candidates considered, chosen
  (provider, model), rationale, and caller agent. `question_hash` is
  stable across whitespace + case so near-identical questions dedup.
- **Cross-project decision memory** — decisions are *always* mirrored
  to `~/.somm/global.sqlite` (not gated by `SOMM_CROSS_PROJECT`).
  Calls stay per-project for privacy; decisions cross over because
  advisory memory is useless without portability.
- **Three new MCP tools** on `somm-mcp` (now 10 tools total):
  - `somm_advise` — ranked candidates with per-factor reasoning.
  - `somm_record_decision` — persist the outcome of a sommelier
    conversation, auto-mirrored globally.
  - `somm_search_decisions` — recall by question / workload /
    provider, default scope is `global`.
- **`somm_recommend` cold-start branch** — when a workload has no
  shadow data, the tool now returns sommelier candidates + any
  prior decisions for that workload instead of an empty list.
- **`SOMMELIER.md` skill** in `somm-skill` — documents the
  recall → advise → record loop for coding agents, with guidance on
  when not to use it (hard user intent, hot loops, private
  workloads).

### Added — documentation
- **`RELEASING.md`** — canonical release checklist including the
  `docs/index.html` update step (previously a forgotten manual
  task).
- **`docs/BLUEPRINT.md`** — design blueprint for anyone building
  their own take: the six forces, the ten tables, non-obvious
  decisions, and an explicit "what to keep minimal if you're
  reimplementing" path. Intended for porters writing in other
  languages or with narrower scope.

## [0.1.1] — 2026-04-19

### Added — multimodal + capability-aware routing
- `SommRequest.prompt` accepts `str | list[dict]` — pass content blocks
  (text + image) following the Anthropic/OpenAI convention. All existing
  `prompt=str` callers keep working.
- Helpers in `somm_core.parse`:
  - `text_prompt(text)` — build a single-text-block list.
  - `image_prompt(text, image_bytes | url, media_type)` — build text +
    image content.
  - `infer_capabilities(prompt)` — scan for `"vision"` and future
    modality capabilities.
  - `prompt_preview(prompt)` — compact stringification that elides
    base64 image payloads for logs/samples.
  - `estimate_prompt_tokens(prompt, image_token_cost)` — per-image
    token addend shared across providers.
- `SommRequest.capabilities_required: list[str]` and `Workload.capabilities_required`:
  workload-level defaults + per-request overrides. The client merges
  these with auto-inferred capabilities before dispatch.
- Router filters the provider chain against
  `model_intel.capabilities_json` before any network call. Unknown
  models fall through as capable (no regression for untracked models).
- `SommNoCapableProvider` (`SOMM_NO_CAPABLE_PROVIDER`): raised when no
  provider in the chain can serve the required capabilities, carrying
  `(provider, model, reason)` skip triples for operator visibility.
- Schema v3 — `workloads.capabilities_required_json`. Additive
  migration; existing rows unaffected.

### Fixed
- Stale minimax tests aligned with live wire format: default model is
  `MiniMax-M2.7` and domain is `api.minimax.io` (prior commit had
  updated the adapter but left the assertions behind).

## [0.1.0] — 2026-04-19

Initial public release. Milestones D1–D7 as described below; see the
commit log for ordering. Not yet published to PyPI.

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

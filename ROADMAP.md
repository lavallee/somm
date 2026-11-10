# somm roadmap

What is left to build, what is deferred, and what is not planned. A shipped
item moves to `CHANGELOG.md`; a bad idea moves to **Parked** with a one-line
reason. Nothing here has a release date.

## Where we are

0.8 → 1.0 built the sensor, plugins, versioned workloads, evals, the
recommendation rules and the public release; 0.14–0.16 added harnesses, async
clients, workload handlers and sticky pins; 0.17 (October 2026) made the loop a
command. In November 2026 an audit of the code, the consumers, the session
logs and the field (`VISION.md`, "Re-grounding, November 2026") set the
seven-phase sequence below, and releases 0.20.2 to 0.26.0 built most of its
code. What code cannot do is what the graduation tests measure: use over time.
**No graduation test below is met yet, and nothing shipped in this sequence has
been verified in sustained use.**

## The November sequence

Each phase says what shipped, then what remains. Phases run in parallel where
they are independent, but nothing in phases 3–6 is trusted until phase 1
graduates.

### 1. A truthful record

**Shipped.** 0.20.2: the telemetry writer survives a bad row, and a failed
health write no longer discards a paid answer. 0.21.0–0.22.0: the data
boundary (four ordered tiers per route, a boundary per workload, enforced on
every path out of the process, `calls.data_tier` on every served call,
OpenRouter tiers deduced from its published zero-retention list). 0.23.0:
`bad_json`, `timeout`, `rate_limit` and reasoning-budget exhaustion recorded as
outcomes; the route on every row (`route_source`, first and served choice,
fallback depth, skipped candidates, `max_tokens` sent); decision recalls stored
and fixture decisions counted apart; one price table with cache reads and
writes at their own rates and unknown prices marked `unknown:no-price`; lean
`claude-cli` calls; the three north-star numbers in `somm doctor`. 0.24.0:
`somm ledgers` (list, `adopt`, `forget`) and global views credited to the
project that made each call.

**Remains.**
- A project-qualified workload id, with a read-side alias for old rows. Today
  two projects declaring one workload share a mirror row and global reports
  work around it (`docs/ledgers.md`).
- An unknown cost is stored as `cost_usd = 0.0` beside its source, not `NULL`;
  readers must check `cost_source`.
- Merging two registered ledgers.

**Graduation:** `somm doctor --global` reports no sentinel rows, no unknown
cost without a source, one row per attempt including failures, and the three
north-star numbers. *Not met:* the doctor prints each check; no real fleet has
run clean on them yet.

### 2. The kernel

**Shipped.** 0.20.2: the writer restarts itself and quarantines a row it
cannot write. 0.25.0: one reused connection pool per HTTP provider with
per-request timeouts (connect capped at 10 s), one error shape filled the same
way by every adapter, `stream()` on `generate()`'s admission, routing, fallback
and recording, and the CLI as a package. 0.25.0–0.26.0: one compare
implementation for the CLI and the MCP server. Retention is `somm prune`.

**Remains.**
- All SQL in `somm-core`: ledgers, telemetry, the shadow worker, the climb and
  GPU readers in `somm`, and `somm-service`, still open SQLite themselves.
- A stall timeout beside the connect, read and total timeouts.
- A registry-driven fleet query in place of mirroring every call row
  (workloads and decisions stay global).
- Split `somm_core/repository.py` (2,800 lines), `sommelier.py` and
  `somm_service/app.py`; mechanical, a PR each.
- Throughout: `somm.__all__`, the `SommLLM` signatures, `somm_core.read` and
  the MCP and HTTP contracts stay fixed; the schema stays additive.

**Graduation:** `stream()` and `generate()` share one pipeline and the
consumers' test suites pass unchanged. *Half met:* one pipeline since 0.25.0;
the consumers' suites have not been run against it.

### 3. The paved road

**Shipped.** 0.24.0: `generate_structured` as the one JSON call,
`raise_on_failure=True` with typed errors, per-call `timeout=`,
`client.health()`, the extended `somm_core.read` and the versioned
`vocabulary()`, and `somm fleet status`. 0.26.0: `SOMM_OPENAI_ENDPOINTS` for a
self-run OpenAI-compatible server without a plugin.

**Remains.**
- A route per call and per workload (`providers=`, `exclusive=`,
  `avoid_local=`) and an explicit default route per project.
- `somm fleet upgrade`: name the writers on an old client and the dead
  editable paths, and say what to run.
- Output contracts: **3 of 214 workloads** declared an output schema at the
  last count, none of the ten busiest among them, so grading most workloads
  measures similarity to a gold answer, not correctness. somm reports the gap
  and never infers a schema from responses.

**Graduation:** three consuming projects delete their wrapper modules. *Not
met.*

### 4. Surfaces

**Shipped.** 0.23.0: `somm mcp install`, `somm skill install`. 0.24.0: the
proxy on `SommLLM` with `/v1/responses` beside `/v1/messages` and
`/v1/chat/completions`; LiteLLM out of `somm-service`; OpenTelemetry export on
the GenAI conventions at a pinned commit, with an `invoke_agent` span per
harness attempt. 0.25.0: proxy streams forwarded as they arrive; one CLI flag
grammar and exit-code contract (`docs/cli.md`). 0.26.0: OTLP protobuf ingest
(`somm-service[otlp]`); refusals on `/v1/responses` as refusal content parts.

**Remains.**
- Apply `stop`, `top_p`, `top_k`, `parallel_tool_calls`, `truncation` and
  non-adaptive `thinking` on the proxy; today they are accepted and named in
  `x-somm-ignored`.
- gzip-encoded OTLP bodies.
- Tool-execution spans inside a harness attempt.
- Golden fixtures that run every harness adapter against recorded real
  streams, so a CLI release cannot silently change a termination or usage
  field.
- An ACP (Agent Client Protocol) normalizer; waits for ACP v2.
- The MCP server on the stateless MCP spec (it uses the 1.x SDK's `FastMCP`).

### 5. Kinds and placement

**Shipped.** 0.24.0: `somm gpu spend`, a read-only report of rented-GPU spend
from gipper receipts; cost per passing answer in `switch_model` evidence.
0.25.0: `llm.decide()`, a decision call returning a label, its probability,
every label's probability and `abstained`, recorded with `kind = 'decide'`;
`result.verdict()` and `decision_calibration` (Brier score, calibration error);
`somm climb`, a read-only report of each rung of a workload from chipsocket
receipts. 0.26.0: `decide_many`, `somm.decide.grade_decisions` (an LLM answers
the same questions as a grader) and `decision_economics` (agreement, abstain
share, cost per 1,000 inputs with and without the decision model).

**Remains.**
- Call `decide` against the live decision API; it is built from the vendor's
  documentation and tested against fakes. Support more than one decision
  provider.
- Images and `box` questions for decisions.
- Abstain-and-escalate as workload policy; today the caller escalates.
- The climb joins a rule-level receipt to a ledger row only when the receipt
  writer passes a call id; otherwise by exact workload name, and the model
  rung's cost is a floor.
- Rented-GPU cost attributed to individual calls, and an amortised cost kind.
- `somm frontier` ranked by cost per passing answer, not score and latency.

### 6. A loop that closes

**Shipped.** 0.24.0: `switch_model` needs 20 grades per arm within 30 days and
a 95% interval that excludes zero; recommendations expire after 30 days;
grading claims are atomic; an unparseable judge reply has no score; a failing
gold route marks the shadow config broken instead of using up samples. 0.25.0:
decision verdicts as human labels for calibration. 0.26.0: the shadow worker
replays the captured request (schema 31) and skips decision rows.

**Remains.**
- Active shadow sampling by surrogate uncertainty, with eval spend reported
  against routing savings.
- Judge hygiene: both orderings of a pairwise judge, markdown stripped, a cheap
  multi-family panel, judges qualified on a seed set.
- Human labels for LLM workloads, not only decisions.
- Observed quota truth for seats: Claude Code's `rate_limits`, Codex's
  `account/rateLimits/read`, Ollama cloud's `/api/usage` and OpenRouter's
  `/key` read into pacing ahead of inference from 429s.
- Cascade routing (cheap first, escalate on estimated quality) as a workload
  setting.
- Verify which upstream served a pinned OpenRouter request;
  `routing_json.upstream` records it only when the response names it.

**Graduation:** a recommendation is applied on a real fleet because the
evidence said so. *Not met.*

### 7. The explanation

**Shipped.** 0.23.0: a site that leads with the meter and the memory, a
glossary, each component in four fixed parts with its evidence or "no evidence
yet", and the case against adopting a gateway and a tracing product, including
when adopting them is right.

**Remains.** A favicon, dashboard screenshots and a seeded demo ledger to take
them from; version, tool and test counts generated from code.

## Later

- **Model and harness memory**: record harness identity and version beside
  model and outcome, and let the sommelier compare (model, harness) pairs.
  Graduation: a recommendation changes because observed harness fit beat a
  model-only prior.
- **Quality priors** from Arena's leaderboard dataset (CC-BY-4.0) and Epoch
  AI's benchmark CSV (CC-BY) when shadow-eval is empty; needs the display-name
  ↔ model-id mapping `somm alias` holds.
- **Catalog expiry**: stop recommending a model whose catalog entry
  (OpenRouter `expiration_date`, models.dev `status`) says it is going away.
- **Bayesian prior weighting** per (model, workload) instead of a decayed
  multiplier; needs a workload-similarity notion first.
- **`outcome_status`** beside `outcome_note`, so negative-outcome detection
  stops matching keywords.
- **Our own price data**: the bundled snapshot is generated from LiteLLM's
  published price file; build it from first-party and MIT sources.

## Deferred (sized but not scheduled)

- **A/B routing** (recommendations as traffic splits, ~2–3 days);
  **`somm.ensemble`** (a parallel-model call, ~2–3 days); **auto-eval
  generation** (rubrics from sampled pairs, ~2 days).
- **Tool-call telemetry columns** out of `raw_json` once a query needs them.
- **Ollama depth**: `/api/show`, `/api/ps`, cloud-tag pricing.
- **Cache efficiency**: a prefix-churn flag and per-provider TTL on
  `somm cache-advice`.
- **Tokenizers**: the `somm[tokenizers]` extra installs `tiktoken`, but token
  estimates still use a characters-per-token rule.
- **Windows service lifecycle**, **packaged installers**, **web admin design**.

Promote one when a real user needs it, when it fits an optional extra, or when
building it yourself would be a days-of-work surprise.

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

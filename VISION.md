# somm — Vision

*The durable north-star for somm. Distinct from `ROADMAP.md` (the sequenced
bets) and `CHANGELOG.md` (what shipped). Re-ground this against reality each
cycle — a vision you never re-ground rots.*

**North star:** somm is the self-hosted meter and memory that lets one
developer with a dozen projects put every model-shaped piece of work in the
right place — a frontier model, a small local one, a decision model, a rule, a
coding-agent harness, a subscription seat, a rented GPU — by recording every
attempt locally, grading production samples, and remembering every placement
decision so the next project starts where the last one left off.

**North-star metric:** cross-project decisions recalled — the number of times a
project's placement choice is answered by somm's accumulated memory instead of
a fresh guess. If somm isn't making the *next* workload faster to get right, it
isn't earning its keep.

Two companions keep that number honest, because a memory nobody acts on is
not a loop:

- **Evidence-routed share** — the fraction of calls whose route was decided by
  a recommendation, a learned override, or a workload policy rather than a
  caller's hard-coded pin.
- **Recommendations applied** — the fraction applied or dismissed within
  fourteen days of being created.

## Strategy bets

- **The record is complete and the hot path never lies.** Everything else is
  built on the ledger, so it comes first. Every attempt is recorded with its
  true outcome, the route it took, and the basis of its cost; a telemetry
  failure never breaks a call or buys it twice; and a guarantee the
  documentation makes is enforced in code or removed from the documentation.
- **Cross-project memory is the moat.** Every observability tool does
  telemetry + evals + prompts. Only somm learns which model to use from *your
  own* call history *across all your projects*, and lets the coding agent ask.
  The sommelier — pick once, remember everywhere — is the differentiated core;
  telemetry is the sensor that feeds it.
- **Sovereignty is the constraint that makes it real.** No phone-home, no cloud
  account, no commercial dependency on the hot path. Works offline with just
  `ollama`. This is not a feature list — it is the reason a one-dev-no-ops-budget
  user can adopt somm where a hosted platform can't follow.
- **The agent closes the loop.** The endgame is not a dashboard a human reads;
  it is an agent that watches telemetry, sources current model intel, and
  proposes better routing — with one-command apply. Recommendation → apply is
  the compounding flywheel.
- **Model and harness are one observed treatment.** For agentic work, raw model
  identity is not enough: permission translation, context management, event
  handling, and resume behavior change the result. Somm provides a portable
  one-attempt contract across supported harnesses and learns from the pair,
  while outer runners retain durable workflow policy.
- **The workload is the unit; what serves it is pluggable.** A workload is a unit
  of work, not necessarily an LLM call. The default kind is `llm` — the provider
  chain — but a project may register other **workload handlers** and bind
  workloads to them. Somm depends on no particular kind; handlers arrive by entry
  point.

  This is the strategic version of the same bet as cross-project memory. A
  workload can start as the cheapest thing to write — a plain LLM call — and then
  *climb*: telemetry and side-testing reveal which part of what the model is
  doing is decidable by a rule, and each such part gets evicted from the model
  into deterministic behavior, until only the irreducible judgment is still
  served by one. Because every rung records against the same workload id, the
  climb is evidence-driven rather than taste-driven, and reversible when a
  stronger model reclaims ground.

  The rungs are not all LLMs. A decision model that returns a typed label and
  a probability, an embedding with a linear head, a reranker, a rule, a chip:
  each is a kind of serving with its own unit of cost and its own notion of
  being right, and each belongs on the same ledger under the same workload id,
  compared by what it costs to get a correct answer. The same holds for where
  the work runs — a metered API, a subscription seat, a local model, or a GPU
  rented for the job.

  Somm's job in that loop is what it has always been: be the meter and the
  memory. The first climb is live: gabe's `same_event` — 700,000 calls on a
  7B local model — runs as a chipsocket socket with somm as its gateway and
  meter. It does not own the deterministic components — the first non-LLM kind,
  `chip`, is published by fab — it owns the record that tells you where each
  workload belongs.

  Owning that record has a standard to meet, and measuring it in August 2026
  showed the gap. A workload name says what kind of work a call is; until
  schema 23 nothing said *which code asked for it*, so telemetry and static
  analysis described the same fleet in two vocabularies that could not be
  joined — 26 of 29 workloads failed to match a known call site, including
  every one of the busiest. And no workload in the fleet declares an output
  contract, which means nothing downstream can decide whether a fixture could
  falsify a workload's result. A record that cannot say where a workload runs
  or what it promises can meter the climb but cannot adjudicate it.

## Re-grounding, October 2026

A census of this machine's fleet — 1.7 million calls since June — showed what
the north-star metric was hiding. The sensor was everywhere: nine projects
writing, the busiest at 300,000 calls a month. The brain was invisible from the
fleet view — zero recommendations, zero evaluations in the shared ledger — and
had in fact run for one project inside its own web process, graded 7,000
samples, left 21 recommendations unactioned, and stopped with that process
three weeks earlier. 87 sommelier decisions had been recorded, nearly all from
one MCP session. The estate had routed around the gaps — seven tools reading the
ledger with their own SQL, every production unit overriding the
sovereign-first default with an environment drop-in, coding-agent attempts
accounted for everywhere except here.

Two lessons for the strategy. First, "the agent closes the loop" needs the
loop to be runnable by a project with no resident process, and visible from
the fleet when it runs; 0.17 made it a command and a timer and taught the
doctor to look in every ledger. Second, cross-project memory is only a moat if the
projects write with a current client; the ledger now records which version
wrote each row so the skew is a finding, not a mystery. The metric stands.
The honest reading of it today is low, and that is the point of having one.

## Re-grounding, November 2026

A second look, this time at the code, the consumers, the agents' session logs
and the field at once. The differentiators held: no tool surveyed combines
cross-project decision memory, production shadow evaluation feeding routing, a
local ledger and a surface a coding agent can ask. The independent gateways
thinned over the year — acquisitions, an archived project, two compromised
releases of the most widely used client library — which is the practical case
for owning the provider layer.

What did not hold was the foundation under the differentiators.

- The north-star metric was not low; it was unrecorded. Recalls were never
  persisted, and nearly every decision in the shared ledger had been written
  by a test fixture. Fifty-four recommendations had been created and none
  applied.
- The record had holes: structured-output failures, timeouts and rate limits
  never reached the ledger as outcomes; fallback depth was not stored; a
  self-hosted endpoint was priced at zero; a workload's identity omitted its
  project.
- The hot path broke promises: one bad row killed the telemetry writer for the
  life of the process; a locked ledger could discard a paid answer and buy it
  again; and a workload marked private could be routed to a cloud provider,
  which three documents said was impossible.
- The flagship climb was mostly invisible. Rules answered more than nine in
  ten of one workload's decisions, and somm saw only the remainder that
  reached a model.
- Consumers had routed around the surface: some twenty hand-rolled JSON
  repair, a dozen overrode the default provider order, and ten read the ledger
  with their own SQL.
- The agent-facing surfaces — the MCP server and the skill — were installed in
  no harness.

Three changes to the strategy follow. Trust in the record becomes the first
bet, ahead of the moat it supports. The unit widens from "which LLM" to "what
should serve this workload", across kinds of model and places to run. And
privacy stops being a three-way label and becomes a **data boundary**: an
ordered scale of where a workload's data may go — hardware you run, a
zero-retention commitment, a provider you subscribe to, anything else — that
the router enforces and the ledger records.

The paved-road test for all of it: a consuming project needs no wrapper
module around somm.

*Note, 10 November 2026.* The two days after this re-grounding (0.20.2 to
0.26.0) put the foundation into code: the data boundary; outcomes, routes and
recalls on every row; one price table with unknown costs marked; one call path
for `generate` and `stream`; `generate_structured`, typed errors and per-call
timeouts; a read API and a vocabulary export; `somm ledgers`, `somm fleet
status` and `somm mirror sync`; `somm mcp install` and `somm skill install`;
decision calls; and `somm climb`. They did not move any of the three
north-star numbers, because those measure use, not code, and none of the
graduation tests in `ROADMAP.md` is met.

## Non-goals

- A composite "quality score" collapsing LMArena + LiveBench + HF into one
  number. Each signal measures something specific; surface which one contributed.
- Continuous API polling. Intel is a cache; the advise path stays fast and
  deterministic.
- Paid/gated sources on the default path. Anything needing a paid key lives
  behind a feature flag. Sovereignty-first.
- A hosted multi-tenant SaaS. The bet is one developer, many projects — not a
  team platform. Postgres is an optional extra, not a pivot.
- A durable job runner or software-factory control plane. Somm does not own
  queues, cross-attempt retry, worktrees, verification, release, approval, or
  project intent; Fab and George do.

## Engine map

- **library** (`somm`) — the one-line drop-in: providers, tool calling,
  streaming, embeddings, decision calls, retries/fallbacks, local telemetry
  capture, and the data boundary that decides which routes a workload may use.
- **harnesses** (`somm.harnesses`) — one provider-neutral coding-agent attempt:
  executable discovery, arguments and permissions, native-event normalization,
  session identity, final output, usage, and portable termination.
- **sommelier** — the ranking brain: cross-project model memory, prior-decision
  recall, shadow-eval + intel signals → a recommended model with reasons.
- **loop** (`somm loop`) — the workers (intel refresh, shadow grading,
  recommendations), run as one pass or on a timer with no resident process.
- **service** (`somm-service`) — the web admin (usage / cost / quality by
  project/workload/time), the HTTP read API, the provider-compatible proxy and
  OTLP ingest; it can also host the loop's scheduler.
- **MCP** (`somm-mcp`, 14 tools) — lets a coding agent query real telemetry and
  collaborate on model selection from inside the harness.
- **skill** (`somm-skill`) — steers coding agents to use somm instead of rolling
  their own wrapper, and to consult the MCP during workload definition.

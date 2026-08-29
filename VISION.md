# somm — Vision

*The durable north-star for somm. Distinct from `ROADMAP.md` (the sequenced
bets) and `CHANGELOG.md` (what shipped). Re-ground this against reality each
cycle — a vision you never re-ground rots.*

**North star:** somm is the self-hosted intelligence loop that lets one
developer with a dozen projects use LLMs well everywhere — recording every call
locally, grading production samples against a gold model, and remembering every
model decision so the next project starts where the last one left off.

**North-star metric:** cross-project decisions recalled — the number of times a
project's model choice is answered by somm's accumulated memory instead of a
fresh guess. If somm isn't making the *next* workload faster to get right, it
isn't earning its keep.

## Strategy bets

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

  Somm's job in that loop is what it has always been: be the meter and the
  memory. It does not own the deterministic components — the first non-LLM kind,
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

- **library** (`somm`) — the one-line drop-in: ten providers, tool calling,
  streaming, embeddings, retries/fallbacks, local telemetry capture.
- **harnesses** (`somm.harnesses`) — one provider-neutral coding-agent attempt:
  executable discovery, arguments and permissions, native-event normalization,
  session identity, final output, usage, and portable termination.
- **sommelier** — the ranking brain: cross-project model memory, prior-decision
  recall, shadow-eval + intel signals → a recommended model with reasons.
- **service** (`somm-service`) — scheduled workers (intel refresh, graders) +
  the web admin (usage / cost / quality by project/workload/time).
- **MCP** (`somm-mcp`, 14 tools) — lets a coding agent query real telemetry and
  collaborate on model selection from inside the harness.
- **skill** (`somm-skill`) — steers coding agents to use somm instead of rolling
  their own wrapper, and to consult the MCP during workload definition.

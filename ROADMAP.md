# somm roadmap

Post-0.2.2 work, in rough priority order. Items move here from
`docs/intel-sources.md` and from malo proposals (`../malo/docs/*.md`)
as they mature from "idea" to "sized and ready to build."

Nothing here is committed to a release date. The goal is to keep the
shape of the future sommelier visible without letting the short list
grow unbounded.

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

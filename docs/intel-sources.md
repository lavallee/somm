# Model intel sources — scratchpad for sommelier expansion

A running list of signals that could feed `model_intel` or the
sommelier ranking, with notes on coverage, refresh cadence, and
failure modes. Most are not yet wired up. Add entries; don't delete —
the "why we didn't use X" notes are useful too.

## Coverage matrix

| Source | What it's good for | Live today? | Notes |
|---|---|---|---|
| OpenRouter `/api/v1/models` | Price, context, modality, vision flags | ✅ `model_intel` worker | Primary source. Stable JSON. |
| Static pricing table | Anthropic/OpenAI/Minimax pricing | ✅ `model_intel` worker | Hand-curated; refresh on each somm release. |
| Ollama `/api/tags` | Locally installed models | ✅ `model_intel` worker | No pricing. No context window (not exposed). |
| **[LMArena](https://lmarena.ai)** | Crowd-sourced Elo rankings — the single best *quality* signal across proprietary + open models | ❌ | See notes below. |
| **[Artificial Analysis](https://artificialanalysis.ai)** | Composite quality + speed (tokens/s) + price per model | ❌ | Commercial-heavy coverage. Has an API (check terms). |
| **[canirun.ai](https://canirun.ai)** | "Can my local GPU run model X at acceptable speed?" | ❌ | Critical for ollama-first workflows — price alone is the wrong filter when hardware is the bottleneck. |
| **[LiveBench](https://livebench.ai)** | Contamination-resistant per-category benchmarks (reasoning, coding, math, data) | ❌ | Monthly refresh. Good for "which model for coding vs reasoning?" differentiation. |
| **[HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)** | MMLU / GPQA / HellaSwag / ARC-C / TruthfulQA / Winogrande, composite score | ❌ | Already in TODOs as "HF trending". DOM scraping is fragile. |
| SWE-bench / SWE-bench Verified | Coding-specific quality (real GitHub issue solve rate) | ❌ | Per-model, slow to update, narrow task. |
| `somm` internal shadow-eval | Workload-specific quality on YOUR prompts | ✅ (opt-in) | The highest-signal source for a specific workload, but only once it has coverage. |

## Why each might matter to the sommelier

### LMArena (quality as Elo)

The ranking currently uses pricing + capability + (maybe) shadow-eval.
That misses *latent* quality — a model that's never been graded on
your workload but is known to perform well generally.

- **Use as:** a per-model `quality_score` in `model_intel.capabilities_json`
  (or a dedicated column if we formalize).
- **Weight:** moderate — Elo is broad-task; your workload may differ.
  Shadow-eval should still dominate when available.
- **Refresh:** weekly is plenty. Elo changes slowly.
- **Failure mode:** the model's name on LMArena may not match the
  `model_intel.model` (e.g. `claude-opus-4-7` vs `Claude Opus 4.7`).
  Need a canonicalization layer.

### canirun.ai (hardware feasibility)

Today, `ollama` models in `model_intel` have no speed / memory
footprint data. The sommelier will happily recommend `llama-70b` to
someone on a 16GB laptop.

- **Use as:** a per-(model, hardware-class) boolean + expected tok/s.
- **Weight:** hard filter when running local, ignored when routing to
  an API provider.
- **Refresh:** on user request (`somm doctor --local`) or at install
  time — not periodic.
- **Failure mode:** user's GPU profile might be stale or absent.
  Default to permissive when unknown (same as capability lookup —
  let them try).
- **Question:** does canirun.ai have an API? If not, this may need to
  become "user declares GPU class in config."

### Artificial Analysis (price + speed + quality composite)

Overlaps partially with OpenRouter (price, context) but adds tokens-
per-second and their own composite quality score.

- **Use as:** speed supplement for the ranking (fast wins ties).
- **Weight:** light — current sommelier has no latency signal beyond
  shadow-eval's `latency_ms` average.
- **Refresh:** daily if they have an API; otherwise skip.

### LiveBench (per-category quality)

Crucial if we ever want the sommelier to differentiate "pick a coding
model" from "pick a reasoning model" from "pick a creative-writing
model." Workload metadata could declare a category hint.

- **Use as:** per-(model, category) quality score.
- **Weight:** high when the workload declares a matching category,
  ignored otherwise.
- **Refresh:** monthly.

## Integration sketch (non-binding)

When this scope gets picked up, the pattern is likely:

1. Extend `model_intel.capabilities_json` to carry nested signal dicts:
   ```json
   {
     "modality": "text+image->text",
     "quality": {"lmarena_elo": 1234, "livebench_coding": 78.2},
     "local_feasibility": {"min_vram_gb": 16, "tok_per_s_rtx4090": 42}
   }
   ```
2. Sommelier `_score()` consumes those nested fields when they exist;
   silently ignores them when they don't. **Backward-compat:** the
   current scoring keeps working when the fields are missing.
3. One refresh worker per source, each feature-flagged. Failures
   never poison the cache — last-good entries remain valid (same
   pattern as `model_intel` today).
4. Canonicalization layer: map display names ↔ API model ids. Start
   as a hand-curated JSON; promote to a learned alias table if
   entries grow.

## What to stay out of

- **No "composite quality score" we invent.** Each benchmark measures
  something specific; don't collapse them into a single number. Let
  the sommelier reasons list surface which signal contributed.
- **No continuous API polling.** Refresh on a schedule (daily/weekly),
  not on every advise() call. `model_intel` is a cache; the advise
  path is fast and deterministic.
- **No paid/gated sources on the default path.** If a source requires
  a paid key, it belongs behind a feature flag. Sovereignty-first.

## References the user has flagged

- lmarena.ai
- canirun.ai
- plus "other benchmarks you're aware of" — see table above

Add to this list as we encounter more. The file is a running
scratchpad, not a design doc.

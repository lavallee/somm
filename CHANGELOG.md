# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
`somm` uses a single unified version across all workspace packages
(`somm`, `somm-core`, `somm-service`, `somm-mcp`, `somm-skill`).

## [Unreleased]

### Added — `no_fallback` for pinned-or-bust evaluation runs

`SommLLM.generate(..., no_fallback=True)` suppresses the normal pinned-call
rescue path. When a `provider` is pinned and the upstream fails, instead of
silently routing to the next provider in the chain, the call returns with
`outcome=UPSTREAM_ERROR` and the *pinned* (provider, model) preserved on
the `SommResult` and `calls` row.

Driven by the same sibling-project (steve) finding that exposed the
adequacy-frontier gap: when running an A/B comparison between two models on
the same workload, the rescue path makes failures invisible — you see a
result tagged with the pinned model that was actually produced by the
fallback. This invalidates the experiment.

Default behavior is unchanged: production workloads still fall through to
the chain when the preferred provider transiently fails.

### Added — adequacy frontier per workload (schema v6)

Driven by sibling-project demand (steve's `parse_listing` workload, where
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

Driven by findings from malo's captioner selection
(`../malo/docs/somm-sommelier-report.md`).

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

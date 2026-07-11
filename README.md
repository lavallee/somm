# somm

**Self-hosted LLM telemetry, routing, and intelligence loop.**
Zero-config. Privacy-first. No commercial dependencies on the hot path.

[![CI](https://github.com/lavallee/somm/actions/workflows/ci.yml/badge.svg)](https://github.com/lavallee/somm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Every LLM observability pitch in 2026 says "telemetry + evals +
prompts." Here's what none of them can say: **your agents learn which
model to use from your own call history, across all your projects, and
your coding agent can ask.** somm records every call locally, grades
production samples against a gold model (online evaluation, no
platform), and remembers every model decision you make — so the next
project starts where the last one left off.

Built for **one developer with a dozen projects and no ops budget** —
the exact scope where cross-project memory pays off and a hosted
platform doesn't.

- 🟢 **Works offline** with just `ollama` running locally
- 🟢 **No phone-home**, no cloud account, no hosted service
- 🟢 **One-line drop-in** for codebases with an existing LLM wrapper
- 🟢 **Ten providers** — ollama, OpenRouter, DeepSeek, Minimax,
  Anthropic, Gemini, OpenAI, Perplexity, `claude`/`codex` CLI
  executors + any OpenAI-compatible gateway
- 🟢 **Tool calling** on every provider — neutral schema in, native
  format out, `ToolCall`s back
- 🟢 **Streaming, embeddings, multimodal** — image prompts routed only
  to capable models
- 🟢 **Sommelier** — cross-project model memory: pick once, remember
  everywhere
- 🟢 **Loud on failure** — `calls.error_detail` + inline `on_error`
  callback so crashes don't hide
- 🟢 **MCP** (14 tools) for Claude Code / Cursor / Windsurf to query
  your real telemetry

## Install

```bash
# library only:
pip install somm

# + web admin + scheduled workers + MCP server:
pip install somm somm-service somm-mcp

# optional shared-deployment driver dependency:
pip install "somm[postgres]"
```

Requires Python 3.12+. The library (`somm` + `somm-core`) is all you
need to start; `somm-service` adds the web admin + background workers,
`somm-mcp` the MCP server, `somm-langchain` the LangChain adapter.

Working from source ([uv](https://docs.astral.sh/uv/) workspace):

```bash
git clone https://github.com/lavallee/somm && cd somm
uv sync --all-packages
```

## Two-minute hello world

```python
import somm

llm = somm.llm(project="my_app")
result = llm.generate(
    prompt="Reply with exactly: pong",
    workload="ping",
)
print(result.text)             # → "pong"
print(result.provider)         # → "ollama"
print(f"${result.cost_usd}")   # → from seeded pricing, updates from model_intel
```

That call just landed a row in `./.somm/calls.sqlite`. Inspect:

```bash
somm status --project my_app --since 1
somm status --project my_app --since 1 --json
somm generate "Reply with exactly: pong" --project my_app --workload ping --json
somm serve --project my_app   # → dashboard at localhost:7878
```

## Quickstart ladder

**L0 — one call.** `import somm`, create `somm.llm(project="my_app")`,
and call `.generate(..., workload="ping")`. This creates
`./.somm/calls.sqlite` and gives you `somm status --json` immediately.

**L1 — named workloads.** Register stable workload names with
`somm workload add claim_extract` or pass `workload="claim_extract"` from
code. Workload names are the unit for spend, failures, policies, and
recommendations.

**L2 — policies and budgets.** Add per-workload constraints with the
workload CLI and configure `~/.somm/plans.toml` for PAYG budgets or
metered coding-plan quotas. The router uses those limits before calling a
provider.

**L3 — online evaluation.** Enable sample capture and shadow eval on
non-private workloads, then run `somm-serve admin run-shadow`. Graded
samples feed `somm frontier`, recommendations, and eval datasets.

**L4 — agent loop.** Install `somm-mcp`, point your coding agent at it,
and use `somm_advise` / `somm_record_decision` / `somm_inbox` so model
choices and recommendations become cross-project memory.

## Why somm

LLM-using Python projects all grow along the same axes. You end up with:

- Multiple call sites across multiple providers
- Retries + fallbacks + backoff sprinkled inline
- Ad-hoc prompt management and silent drift when you edit a string
- Swallowed errors — "UPSTREAM_ERROR" rows with no body to triage from
- No idea what you spent, which model answered, or if quality regressed
- The frontier agent pitching you models from its training data, not
  your real workload

`somm` is the shared substrate that replaces every one of those.

### What's NOT here

The self-hosted LLM-tooling space has a pattern: open-source today,
acquired or gated tomorrow. somm is structured so that can't happen to
you:

- **No `ee/` directory.** Every feature in this repo is MIT, forever.
- **No license keys, no feature flags tied to a vendor account.**
- **No beacon telemetry.** The somm project receives nothing about you.
- **No commercial dependency on the hot path.** Model pricing intel
  comes from free, keyless, redistributable sources; anything gated
  lives behind an opt-in feature flag.
- **No server between you and your data.** Everything is SQLite files
  you own, queryable with plain SQL.

### Grafts into an existing project — change one import

```python
# Before:
from myproject.llm import FooLLM
# After:
from somm.compat import GenericLLMCompat as FooLLM
```

Existing call sites don't change. Telemetry, provider fallback, and cost
tracking land on every call. If your project uses the raw OpenAI SDK,
there's an [`openai_chat_completions`](./examples/openai_swap_in.py)
shim that mirrors `openai.OpenAI().chat.completions.create()`.

### Privacy-first by default

- Prompt bodies are **not** stored unless you opt in per workload.
- `privacy_class=PRIVATE` workloads never egress. Enforced in the router,
  the online-eval worker, AND a SQL view (defense in depth).
- SQLite files are `chmod 0600`; parent dir `0700`. `somm doctor` warns
  on drift.
- Web admin binds `localhost` only by default.

### Loud on failure

Every non-OK outcome lands in `calls.error_detail` — a bounded (512
char) operator-friendly string: `{ExceptionClass}: msg | http_status=X | body=…`
parsed from `httpx.HTTPStatusError.response`. No more opaque
`UPSTREAM_ERROR` rows with nothing to triage from.

```python
# Default: one-line stderr warning on every non-OK outcome.
llm = somm.llm(project="my_app")

# Forward to logging / Slack / PagerDuty:
llm = somm.llm(
    project="my_app",
    on_error=lambda evt: logger.warning("llm fail: %s", evt["error_detail"]),
)

# Or silence entirely (noisy in CI/tests):
llm = somm.llm(project="my_app", on_error=lambda _: None)
```

### Tool calling — one neutral shape, every provider

```python
result = llm.generate(
    messages=[{"role": "user", "content": "What's the weather in Oslo?"}],
    tools=[{
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }],
    workload="agent_loop",
)
for call in result.tool_calls:
    print(call.name, call.arguments)   # → get_weather {'city': 'Oslo'}
print(result.stop_reason)              # → "tool_use"
```

Declare tools once in somm's neutral schema (`parameters` = JSON
Schema); per-provider adapters translate to and from native formats
(Anthropic `input_schema`, OpenAI `function`, Ollama `/api/chat`). Providers that can't honor a request
raise loudly instead of silently dropping it, and the router falls
through. Full design: [`docs/tool-calling.md`](./docs/tool-calling.md).

### Streaming and embeddings

```python
for chunk in llm.stream("Tell me a story", workload="narrate"):
    print(chunk, end="", flush=True)     # <think> blocks stripped

vec = llm.embed("the quick brown fox", workload="search_index")
print(vec.dim, vec.cost_usd)             # telemetry row lands like any call
```

### Multimodal prompts, capability-aware routing

```python
import somm
from somm_core.parse import image_prompt

llm = somm.llm(project="my_app")
blocks = image_prompt(
    text="What's in this photo?",
    image_bytes=open("shot.png", "rb").read(),
)
result = llm.generate(
    prompt=blocks,
    workload="photo_describe",
    # capabilities auto-inferred → router skips text-only models
)
```

Router filters `model_intel.capabilities_json` before any network call.
Unknown models fall through as capable. `SommNoCapableProvider`
surfaces upfront if no provider in the chain can serve the request —
with the list of (provider, model, reason) triples it skipped. More:
[`docs/multimodal.md`](./docs/multimodal.md).

### Sommelier — cross-project model memory

Ask the sommelier in your MCP-capable agent: _"Best free vision models
on openrouter right now?"_ Get a ranked list from `model_intel` with
capability + price reasoning. Pick one. Next week in a different
project, ask the same thing — somm remembers what you picked and why,
across every project on your machine. Decisions are the one signal
that always crosses project boundaries.

Three MCP tools (`somm_advise`, `somm_record_decision`,
`somm_search_decisions`) wrap the recall → advise → record loop. See
[`SOMMELIER.md`](./packages/somm-skill/src/somm_skill/SOMMELIER.md).

### Online evaluation — builds its own eval dataset

Opt a workload in and a background worker samples N% of production
calls, re-runs them through a gold-standard model of your choice, and
grades both with structural + text-similarity scorers — the pattern
hosted platforms call "online evaluation," running locally at zero
platform cost. The resulting data feeds the agent worker, which emits
concrete recommendations:

> **`switch_model`** — claim_extract currently on ollama/gemma4:e4b
> (score 0.4, 500ms). Shadow evals show ollama/gemma3:27b scoring 0.85
> at 100ms — +45% quality, -80% latency, same cost. Try it?

Budget-capped per workload. Skipped entirely on private workloads.

Production samples can also become durable CI fixtures. Promote a captured
call into a dataset, run it synchronously as a gate, ask somm to propose a
new prompt from failing graded calls, then run an experiment campaign that
logs keep/revert decisions without mutating production:

```bash
somm eval promote-call <call_id> --dataset golden
somm eval run --workload claim_extract --dataset golden --threshold 0.85
somm optimize --workload claim_extract --from production --label proposed
somm campaign run --workload claim_extract --dataset golden --max-rounds 5 \
  --token-budget 50000 --plateau-window 2 --log campaign.jsonl
```

### LangChain / agent frameworks

`somm-langchain` ships `SommChatModel(BaseChatModel)` so LangChain,
LangGraph, and `deepagents` apps get telemetry, routing, and model
memory without changing agent-framework call sites — including
`bind_tools()`. See [`packages/somm-langchain`](./packages/somm-langchain).

### Extension hooks

External tools can observe somm without somm knowing about them:
register a **correlation-id provider** (stamps your request/trace/job id
on every `calls` row) and/or **call observers** (an event dict after
every call) — explicitly via `somm.hooks`, or automatically through the
`somm.hooks` entry-point group. Hook failures never break the call path.

### MCP — talk to your telemetry from the agent's side

`somm-mcp` ships **14 stdio tools** any MCP-capable agent can call:

| tool | what it does |
|---|---|
| `somm_stats` | rollup by workload × provider × model |
| `somm_search_calls` | filter calls by workload / provider / model / outcome |
| `somm_recommend` | open recs + shadow-ranked models per workload |
| `somm_inbox` | list open/applied/dismissed recommendations |
| `somm_apply_recommendation` | apply a recommendation and record a decision |
| `somm_dismiss_recommendation` | dismiss a recommendation |
| `somm_register_workload` | commit a workload with privacy class + required capabilities |
| `somm_register_prompt` | commit prompt versions (minor/major/explicit) |
| `somm_compare` | run a prompt through N models side-by-side |
| `somm_replay` | replay a stored call against a different model |
| `somm_eval_promote_call` | copy a sampled call into a durable eval dataset |
| `somm_advise` | rank candidates from `model_intel` against free-form constraints |
| `somm_record_decision` | persist the outcome of a sommelier conversation (cross-project) |
| `somm_search_decisions` | recall prior decisions — globally by default |

Add to Claude Code / Cursor / Windsurf:

```json
{
  "command": "somm-mcp",
  "env": { "SOMM_PROJECT": "my_app" }
}
```

## CLI

```bash
somm status [--project P] [--since N] [--global] [--json]
                                                    # rollup (per-project / cross-project)
somm generate <prompt> [--workload W] [--json]      # one-shot LLM call through somm
somm tail [--workload NAME] [--poll-interval S]    # live call stream
somm compare <prompt> --models p/m,p/m             # side-by-side N-model comparison
somm frontier --workload NAME [--since N]          # adequacy frontier per (provider, model)
somm spend [--json]                                # today's spend vs daily budget cap
somm plans [--json] [--project-only]               # metered-plan quota usage + pacing (fleet-wide)
somm backfill-costs [--since N] [--dry-run]        # recompute $0 calls that now have pricing
somm drain-spool                                   # replay spooled telemetry into the DB
somm workload add/list/show                        # register and inspect workloads
somm prompt list/show/register/fork/diff/label     # immutable prompt versions + mutable labels
somm eval promote-call/run                         # durable datasets + synchronous eval gates
somm optimize --workload NAME                      # propose a prompt fork from failing graded calls
somm campaign run --workload NAME --dataset NAME   # repeated eval campaign with keep/revert log
somm doctor                                        # config, ollama, db, intel, workers, cooldowns
somm serve [--host H] [--port N]                   # web admin + scheduler + workers
```

JSON-mode command failures use a stable stderr envelope:

```json
{"ok": false, "error": {"type": "ValueError", "message": "...", "exit_code": 2}}
```

Exit codes are stable for JSON-mode automation: `0` success, `1`
unexpected runtime failure, `2` usage/value error, `66` missing input
file, and `77` permission error.

### Stable local read API and OTLP ingest

With `somm-service` installed, the dashboard also exposes supported JSON
read endpoints. They require the service bearer token, or the dashboard's
same-origin local header path on a loopback `Host`; set
`SOMM_SERVICE_PUBLIC_READ=1` only for deployments that deliberately expose
read-only telemetry.

```text
GET /api/status
GET /api/stats?window=7
GET /api/calls?window=7&q=gemma&limit=100
GET /api/sessions?window=7
GET /api/version
GET /api/recommendations
```

`/api/stats`, `somm status --json`, and MCP `somm_stats` include serving
rollups by workload/provider/model: latency and TTFT p50/p95/p99, TPOT,
observed token throughput, and goodput against `max_p95_latency_ms` when a
workload SLO is set.

Polyglot apps can send OpenTelemetry JSON traces without importing the
Python SDK:

```bash
curl -H "Authorization: Bearer $SOMM_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d @traces.json \
  http://127.0.0.1:7878/api/otlp/v1/traces
```

The OTLP parser is lenient about `gen_ai` semantic-convention drift and
stores accepted spans as ordinary `calls` rows. Ingest rejects oversized
JSON bodies and over-cap span batches before writing rows. SQLite remains
the default backend; the `somm[postgres]` extra installs the Postgres
driver dependency for shared-deployment experiments.

With `somm-service` installed:

```bash
somm-serve admin refresh-intel [--hf]   # refresh model pricing + context windows
somm-serve admin list-intel             # inspect the cache
somm-serve admin run-agent              # one-shot analysis pass
somm-serve admin run-shadow             # one-shot online-eval grading pass
```

### PAYG vs metered plans — cost is not always dollars

Not every provider bills the same way. API keys are usually **PAYG**
(per-token dollars; `cost_usd` is real spend), but coding plans and CLI
seats are **metered**: marginal dollars are ~0 inside a recurring
quota, `cost_usd` is notional list-price, and the scarce resource is
**window headroom**. Declare which is which — machine-wide, because a
plan's quota is shared by every project on the same account — in
`~/.somm/plans.toml`:

```toml
[minimax]
mode = "metered"
plan = "coding-pro"
soft_target_pct = 80    # deprioritize beyond this pace
enforce = false         # true: hard-skip the provider when a limit is exhausted

[[minimax.limits]]
window = "month"        # calendar month (anchor_day = billing reset day)…
anchor_day = 12
quota = 40.0
unit = "usd_equiv"      # requests | tokens_in | tokens_out | tokens_total | usd_equiv

[[minimax.limits]]
window = "5h"           # …and/or rolling windows
quota = 500
unit = "requests"

[gemini]
mode = "payg"
```

PAYG providers can carry limits too — there they're **self-imposed
budgets** (a max spend over a window, LiteLLM-style), paced with the
same math but in real dollars.

Then:

- **`somm plans`** shows every limit's usage in its current window —
  across all your projects (each `somm.llm()` registers its DB in
  `~/.somm/registry.json`) — with pace ratio and straight-line
  projection: are you on track to blow the quota before it resets?
  Plus **payg burn rates** (1d/7d/30d spend, $/day, projected month),
  a **value multiple** for metered plans (`price = 50.0` → "consumed
  $260 of list-price tokens on a $50/mo plan, ≈5.2x"), and **quota
  drift warnings** when your own telemetry contradicts a declared
  limit (usage past quota with calls still succeeding, or recent 429
  ceilings far from the declared number — vendors reset limits
  without notice).
- **The router paces automatically**: a provider past its soft target
  and burning faster than the window passes is deprioritized (tried
  only after in-pace providers fail); an exhausted limit with
  `enforce = true` is skipped outright.
- **The sommelier knows the difference**: `somm_advise` annotates
  metered candidates with plan headroom and pace, so "cheap but
  scarce" ranks differently from "cheap".

You don't have to transcribe vendor limits by hand: somm ships a
curated **plan catalog** (`somm plans --catalog` lists it, with source
URLs and last-verified dates). Reference an entry and inherit its
limits — your own `[[limits]]` always win:

```toml
[minimax]
mode = "metered"
catalog = "coding-pro"   # limits inherited from the bundled catalog
```

Plan limits are marketing copy, not an API, so every catalog entry is
dated; `somm plans` warns when one you rely on hasn't been re-verified
in 90 days. And because vendors increasingly publish *no* numbers at
all, `somm plans` also reports **observed ceilings**: at each quota-429
in your own telemetry, the trailing-window usage ≈ the real limit —
your fleet measures what the vendor won't say.

Providers you don't declare default sensibly: ollama → free,
`claude-cli`/`codex-cli` → metered, API providers → PAYG.

## Configuration

Everything works offline with just ollama running. Every commercial
provider is opt-in via its own env var.

<details>
<summary>Env var reference</summary>

| Variable | Default | Meaning |
|---|---|---|
| `SOMM_PROJECT` | `default` | project name tagged on every call |
| `SOMM_MODE` | `observe` | `observe` (auto-registers workloads) or `strict` |
| `SOMM_PROVIDER_ORDER` | sovereign-first | comma-sep chain override (e.g. `openrouter,minimax,ollama`) |
| `SOMM_OLLAMA_URL` | `http://localhost:11434` | local ollama endpoint |
| `SOMM_OLLAMA_MODEL` | `gemma4:e4b` | default ollama model |
| `SOMM_OLLAMA_THINK` | `0` | `1` sets `"think": true` on ollama requests (reasoning models) |
| `SOMM_OLLAMA_KEEP_ALIVE` | `30m` | pinned residency window; `0` opts out |
| `OPENROUTER_API_KEY` | — | enables OpenRouter |
| `SOMM_OPENROUTER_ROSTER` | built-in free roster | comma-sep model ids |
| `DEEPSEEK_API_KEY` | — | enables DeepSeek |
| `SOMM_DEEPSEEK_MODEL` | `deepseek-chat` | |
| `MINIMAX_API_KEY` | — | enables Minimax |
| `SOMM_MINIMAX_MODEL` | `MiniMax-M2.7` | |
| `ANTHROPIC_API_KEY` | — | enables Anthropic |
| `SOMM_ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | |
| `GEMINI_API_KEY` | — | enables Gemini (via OpenAI-compat endpoint) |
| `SOMM_GEMINI_MODEL` | `gemini-2.5-pro` | |
| `OPENAI_API_KEY` | — | enables OpenAI |
| `SOMM_OPENAI_MODEL` | `gpt-4o-mini` | |
| `SOMM_OPENAI_BASE_URL` | `https://api.openai.com/v1` | for OpenAI-compatible gateways |
| `PERPLEXITY_API_KEY` | — | enables Perplexity (pinned-only; never a routine fallback) |
| `SOMM_PERPLEXITY_MODEL` | `sonar` | |
| `SOMM_HTTP_TIMEOUT` | `180` | seconds, OpenAI-compat providers |
| `SOMM_BUDGET_FAIL_CLOSED` | `0` | `1` blocks calls once a workload's daily cap is reached |
| `SOMM_BUDGET_DEFAULT_CAP_USD_DAILY` | — | daily cap for workloads without an explicit one (fail-closed mode) |
| `SOMM_INPROCESS_WORKERS` | `0` | `1` runs the intelligence-loop scheduler inside your process (needs somm-service) |
| `SOMM_CROSS_PROJECT` | `0` | `1` mirrors telemetry to `~/.somm/global.sqlite` |
| `SOMM_GLOBAL_PATH` | `~/.somm/global.sqlite` | mirror file location |
| `SOMM_ENABLE_HF_INTEL` | `0` | `1` enables the opt-in HuggingFace intel worker |

The `claude` / `codex` CLI executors are auto-detected when the binary
is on PATH, but never join the default routing order — reach them via
`SOMM_PROVIDER_ORDER` or `generate(provider="claude-cli")`.

</details>

## Architecture

```
   library (sensor) ──► local store ◄── service (brain)
       ▲                    ▲                  │
       │                    │                  ├─► online-eval worker
       │                    │                  ├─► model-intel worker
       │                    │                  ├─► agent worker
       │                    │                  └─► web admin
       │                    │
       └── skill (onboarding) ─── MCP (interface for coding agents)
```

Six packages:

- **`somm-core`** — schema, migrations, repository, config, parse
  helpers (incl. multimodal content-block + capability helpers)
- **`somm`** — `SommLLM`, providers, routing, streaming, embeddings,
  tool calling, sommelier, compat shims, hooks, CLI
- **`somm-service`** — starlette web admin + HTTP API + scheduler + 3 workers
- **`somm-mcp`** — stdio MCP server with 14 tools
- **`somm-langchain`** — `SommChatModel` adapter for LangChain/LangGraph/deepagents
- **`somm-skill`** — onboarding markdown templates for coding agents

`somm-core` ships a bundled pricing snapshot (~350 models, derived from
[LiteLLM's price file](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json))
that syncs into every project database on init — cost tracking works
offline, out of the box, for every provider somm routes to. The
service tier's background workers run under `somm serve`, or inside
your own process with `SOMM_INPROCESS_WORKERS=1`.

## Docs

- 📐 [**BLUEPRINT.md**](./docs/BLUEPRINT.md) — the design doc: six
  load-bearing forces + the data model
- 🗺️ [**ROADMAP.md**](./ROADMAP.md) — what's next, what's deferred, what's not planned
- 📜 [**CHANGELOG.md**](./CHANGELOG.md) — release log
- 🔧 [**Tool calling**](./docs/tool-calling.md) — neutral schema + per-provider adapters
- [**Plugins and hooks**](./docs/plugins.md) — hook phases, reference plugins, and provider entry points
- 🖼️ [**Multimodal prompts**](./docs/multimodal.md) — image blocks +
  capability-aware routing
- 🍷 [**Sommelier skill**](./packages/somm-skill/src/somm_skill/SOMMELIER.md) —
  cross-project model advisor for coding agents
- 🚢 [**RELEASING.md**](./RELEASING.md) — canonical release checklist
- 🔥 [**Error reference**](./docs/errors/) — canonical `SOMM_*` codes
- 🧪 [**Examples**](./examples/) — drop-in, OpenAI swap, private workloads

## Contributing

```bash
uv sync --all-packages
uv run pytest packages/ tests/
```

Live-provider tests (ollama) auto-skip when unavailable or contended.
VCR-style fixtures cover provider-specific parsing quirks. The
top-level `tests/test_blocklist.py` guard fails builds that leak
internal names or personal paths.

## License

[MIT](./LICENSE) · © 2026 Marc Lavallee and contributors.

## Status

**v0.12.0 beta / pre-1.0.** somm is published for real-world dogfooding, but
the public API, MCP tools, service contracts, and migration semantics are not
yet declared stable. See [CHANGELOG](./CHANGELOG.md) for the release log and
[ROADMAP.md](./ROADMAP.md) for where things are headed.

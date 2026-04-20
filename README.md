# somm

**Self-hosted LLM telemetry, routing, and intelligence loop.**
Zero-config. Privacy-first. No commercial dependencies on the hot path.

`somm` is the substrate you wish every LLM-using Python project had: one
call wraps telemetry, provider routing, cooldowns, streaming, prompt
versioning, provenance, cost tracking, shadow-eval, multimodal
dispatch, cross-project model memory, and agent-driven recommendations
— all local, all yours.

- 🟢 **Works offline** with just `ollama` running locally
- 🟢 **No phone-home**, no cloud account, no hosted service
- 🟢 **One-line drop-in** for codebases with an existing LLM wrapper
- 🟢 **Six providers** — ollama, OpenRouter, Anthropic, OpenAI, Gemini,
  Minimax + any OpenAI-compatible gateway
- 🟢 **Multimodal** — text + image prompts routed only to capable models
- 🟢 **Sommelier** — cross-project model memory: pick once, remember
  everywhere
- 🟢 **Loud on failure** — `calls.error_detail` + inline `on_error`
  callback so crashes don't hide
- 🟢 **MCP** (10 tools) for Claude Code / Cursor / Windsurf to query your
  real telemetry

---

## Install

```bash
# library only:
pip install somm

# + web admin + scheduled workers + MCP server:
pip install somm somm-service somm-mcp
```

Requires Python 3.12+. Uses [uv](https://docs.astral.sh/uv/) in development.

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
somm serve --project my_app   # → dashboard at localhost:7878
```

## Why somm

LLM-using Python projects all grow along the same axes. You end up with:

- Multiple call sites across multiple providers (ollama, OpenRouter,
  Anthropic, OpenAI, Gemini, Minimax …)
- Retries + fallbacks + backoff sprinkled inline
- Ad-hoc prompt management and silent drift when you edit a string
- Swallowed errors — "UPSTREAM_ERROR" rows with no body to triage from
- No idea what you spent, which model answered, or if quality regressed
- The frontier agent pitching you models from its training data, not
  your real workload

`somm` is the shared substrate that replaces every one of those.

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
  the shadow-eval worker, AND a SQL view (defense in depth).
- SQLite files are `chmod 0600`; parent dir `0700`. `somm doctor` warns
  on drift.
- Web admin binds `localhost` only by default.
- Zero beacon telemetry. The `somm` project receives nothing about you.

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

### Builds its own eval dataset

Opt a workload in to shadow-eval and a background worker samples N% of
production calls, re-runs them through a gold-standard model of your
choice, and grades both with structural + text-similarity scorers. The
resulting data feeds the agent worker, which emits concrete
recommendations:

> **`switch_model`** — claim_extract currently on ollama/gemma4:e4b
> (score 0.4, 500ms). Shadow evals show ollama/gemma3:27b scoring 0.85
> at 100ms — +45% quality, -80% latency, same cost. Try it?

Budget-capped per workload. Skipped entirely on private workloads.

### MCP — talk to your telemetry from the agent's side

`somm-mcp` ships **10 stdio tools** any MCP-capable agent can call:

| tool | what it does |
|---|---|
| `somm_stats` | rollup by workload × provider × model |
| `somm_search_calls` | filter calls by workload / provider / model / outcome |
| `somm_recommend` | open recs + shadow-ranked models per workload |
| `somm_register_workload` | commit a workload with privacy class + required capabilities |
| `somm_register_prompt` | commit prompt versions (minor/major/explicit) |
| `somm_compare` | run a prompt through N models side-by-side |
| `somm_replay` | replay a stored call against a different model |
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
somm status [--since N] [--global]    # rollup (per-project / cross-project)
somm tail [--workload NAME]           # live call stream
somm compare <prompt> --models p/m,p/m   # side-by-side N-model comparison
somm doctor                           # config, ollama, db, intel, workers, cooldowns
somm serve                            # web admin + scheduler + workers
```

With `somm-service` installed:

```bash
somm-serve admin refresh-intel    # refresh model pricing + context windows
somm-serve admin list-intel       # inspect the cache
somm-serve admin run-agent        # one-shot analysis pass
somm-serve admin run-shadow       # one-shot shadow-eval grading pass
```

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
| `ANTHROPIC_API_KEY` | — | enables Anthropic |
| `SOMM_ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | |
| `OPENAI_API_KEY` | — | enables OpenAI |
| `SOMM_OPENAI_MODEL` | `gpt-4o-mini` | |
| `SOMM_OPENAI_BASE_URL` | `https://api.openai.com/v1` | for OpenAI-compatible gateways |
| `GEMINI_API_KEY` | — | enables Gemini (via OpenAI-compat endpoint) |
| `SOMM_GEMINI_MODEL` | `gemini-2.5-pro` | |
| `MINIMAX_API_KEY` | — | enables Minimax |
| `SOMM_MINIMAX_MODEL` | `MiniMax-M2.7` | |
| `SOMM_CROSS_PROJECT` | `0` | `1` mirrors telemetry to `~/.somm/global.sqlite` |
| `SOMM_GLOBAL_PATH` | `~/.somm/global.sqlite` | mirror file location |

</details>

## Architecture

```
   library (sensor) ──► local store ◄── service (brain)
       ▲                    ▲                  │
       │                    │                  ├─► shadow-eval worker
       │                    │                  ├─► model-intel worker
       │                    │                  ├─► agent worker
       │                    │                  └─► web admin
       │                    │
       └── skill (onboarding) ─── MCP (interface for coding agents)
```

Five packages:

- **`somm-core`** — schema v5, migrations, repository, config, parse
  helpers (incl. multimodal content-block + capability helpers)
- **`somm`** — `SommLLM`, providers, routing, streaming, sommelier,
  compat shims, CLI
- **`somm-service`** — starlette web admin + HTTP API + scheduler + 3 workers
- **`somm-mcp`** — stdio MCP server with 10 tools
- **`somm-skill`** — onboarding markdown templates for coding agents

`somm-core` ships known-model pricing on library init, so cost
tracking works out of the box without running `somm-serve admin
refresh-intel` first. Current Anthropic 4.5–4.7 family +
OpenAI / Gemini / ollama-local prices all land on first use.

## Docs

- 📘 [**PLAN.md**](./PLAN.md) — full design doc + 48-row decision audit trail
- 📜 [**CHANGELOG.md**](./CHANGELOG.md) — release log
- 🖼️ [**Multimodal prompts**](./docs/multimodal.md) — image blocks +
  capability-aware routing
- 🍷 [**Sommelier skill**](./packages/somm-skill/src/somm_skill/SOMMELIER.md) —
  cross-project model advisor for coding agents
- 📐 [**BLUEPRINT.md**](./docs/BLUEPRINT.md) — design guide for building your own take
- 🚢 [**RELEASING.md**](./RELEASING.md) — canonical release checklist
- 🔥 [**Error reference**](./docs/errors/) — canonical `SOMM_*` codes
- 🧪 [**Examples**](./examples/) — drop-in, OpenAI swap, private workloads

## Contributing

```bash
uv sync --all-packages
uv run pytest packages/ tests/     # 242 tests pass
```

Live-provider tests (ollama) auto-skip when unavailable. VCR-style
fixtures cover provider-specific parsing quirks. The top-level
`tests/test_blocklist.py` guard fails builds that leak internal names
or personal paths.

## License

[MIT](./LICENSE) · © 2026 Marc Lavallee and contributors.

## Status

**v0.2.1.** See [CHANGELOG](./CHANGELOG.md) for the release log and
[PLAN.md](./PLAN.md) for where things are headed.

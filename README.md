# somm

> Self-hosted LLM telemetry, routing, and intelligence loop for individuals
> and small teams. Zero-config. Privacy-first. No commercial dependencies
> on the hot path.

`somm` wraps your LLM calls with:
- **Provider-agnostic routing** (ollama, openrouter, anthropic, openai,
  minimax; any OpenAI-compatible gateway via plugin).
- **Cooldown-aware fallback** — when the free-tier roster rate-limits,
  roll to the next model; when a provider's circuit-breaks, skip it.
- **Telemetry** — every call lands in local SQLite. Call ID, tokens,
  latency, cost, outcome, prompt/response hashes, provenance.
- **Shadow-eval (opt-in)** — sample N% of calls, re-run through a gold
  model, grade via structural + text similarity. Build an eval dataset
  from production traffic without writing any evals.
- **Agent worker** — weekly analysis of calls + eval results + model
  intel; emits `switch_model` / `new_model_landed` / `chronic_cooldown`
  recommendations with evidence.
- **MCP server** — 7 tools for coding agents (Claude Code, Cursor,
  Windsurf) to query stats, compare models, replay calls, register
  workloads + prompts.
- **Web admin** at `localhost:7878` — one page, recommendations above
  charts, no login needed.

`somm` is built to be **self-hosted**. It works fully offline with just
`ollama` running locally. Every commercial provider is opt-in, per its
own env var. No hosted service, no cloud account, no phone-home
telemetry — your prompts and traces stay on your disk.

## Install

```bash
# Library only (hot-path integration):
uv add somm
pip install somm

# Library + web admin + MCP server + scheduled workers:
uv add somm somm-service somm-mcp
pip install somm somm-service somm-mcp
```

`somm` targets Python 3.12+ and uses [uv](https://docs.astral.sh/uv/) in
development.

## Two-minute hello world

```python
import somm

llm = somm.llm(project="my_project")
result = llm.generate(
    prompt="Reply with exactly: pong",
    workload="ping",
)
print(result.text)                   # "pong"
print(result.provider, result.model) # "ollama", "gemma4:e4b"
print(f"${result.cost_usd:.6f}")     # cost (from model_intel cache)
```

That call lands a row in `./.somm/calls.sqlite`. Ask for a rollup:

```bash
somm status --project my_project --since 1
```

Stand up the dashboard:

```bash
somm serve --project my_project
# open http://localhost:7878
```

## Grafting into an existing project

If your project already has an LLM wrapper, the `somm.compat` package
gives you a one-line swap:

```python
# before:
from myproject.llm import FooLLM
# after:
from somm.compat import GenericLLMCompat as FooLLM
```

Call-site code doesn't need changes. For raw-OpenAI-SDK projects:

```python
from somm.compat import openai_chat_completions as create
resp = create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
    project="my_project",
)
print(resp.choices[0].message.content)
```

See [`examples/`](./examples/) for runnable patterns
(`drop_in_wrapper.py`, `openai_swap_in.py`, `private_workload.py`).

## Configuration (env vars)

All optional; `somm` works with just ollama running locally.

| Variable                    | Default                    | Meaning |
|-----------------------------|----------------------------|---------|
| `SOMM_PROJECT`              | `default`                  | project name (tag on every call) |
| `SOMM_MODE`                 | `observe`                  | `observe` or `strict` |
| `SOMM_OLLAMA_URL`           | `http://localhost:11434`   | local ollama endpoint |
| `SOMM_OLLAMA_MODEL`         | `gemma4:e4b`               | default ollama model |
| `OPENROUTER_API_KEY`        |                            | enables openrouter in chain |
| `SOMM_OPENROUTER_ROSTER`    | *built-in free roster*     | comma-sep list of openrouter model ids |
| `ANTHROPIC_API_KEY`         |                            | enables anthropic |
| `SOMM_ANTHROPIC_MODEL`      | `claude-haiku-4-5-20251001` | |
| `OPENAI_API_KEY`            |                            | enables openai |
| `SOMM_OPENAI_MODEL`         | `gpt-4o-mini`              | |
| `SOMM_OPENAI_BASE_URL`      | `https://api.openai.com/v1` | for openai-compatible gateways |
| `MINIMAX_API_KEY`           |                            | enables minimax |
| `SOMM_MINIMAX_MODEL`        | `MiniMax-M2`               | |
| `SOMM_CROSS_PROJECT`        | `0`                        | set `1` to mirror telemetry to `~/.somm/global.sqlite` |
| `SOMM_GLOBAL_PATH`          | `~/.somm/global.sqlite`    | mirror file location |

## CLI

```bash
somm status [--since N] [--global]   # rollup (per-project or cross-project)
somm tail [--workload NAME]          # live call stream
somm compare <prompt> --models p/m,p/m  # side-by-side N-model comparison
somm doctor                          # health: config, ollama, db, intel, workers, cooldowns
somm serve                           # run the web admin + HTTP API + worker scheduler

# with somm-service installed:
somm-serve admin refresh-intel       # refresh model pricing
somm-serve admin list-intel          # show cached prices
somm-serve admin run-agent           # one-shot agent analysis
somm-serve admin run-shadow          # one-shot shadow-eval pass
```

## MCP

`somm-mcp` ships seven tools for coding agents:

- `somm_stats` — rolled-up call counts
- `somm_search_calls` — query calls by filters
- `somm_recommend` — open recommendations + shadow-eval rankings per workload
- `somm_register_workload` — commit a workload (with optional privacy_class)
- `somm_register_prompt` — commit prompt versions (minor/major/explicit)
- `somm_compare` — run a prompt through N models side-by-side
- `somm_replay` — replay a stored call against a different model

Install into Claude Code / Cursor / Windsurf:

```bash
# stdio MCP entry point — configure in your client:
command: somm-mcp
args:    []
env:     SOMM_PROJECT=my_project
```

## Privacy

`somm` is designed for workloads you'd be uncomfortable egressing.

- **`privacy_class=private`** workloads are banned from shadow-eval and
  from any upstream provider — enforced in the router AND in the
  `shadow_candidates` SQL view (defense in depth).
- **Prompt/response bodies** are NOT stored by default. The library
  keeps only hashes; full bodies land in a separate `samples` table
  only when you opt in per workload.
- **SQLite files** are created with perms `0600` and parent dir `0700`.
  `somm doctor` warns if the perms drift.
- **Web admin** binds to `localhost` only by default. Exposing via
  `--bind 0.0.0.0` prints a loud warning.
- **No beacon telemetry.** The `somm` project receives zero data
  about your installation or usage.

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

- `somm-core`: schema v2 + migrations + repository + config + parse helpers.
- `somm`: Python library (SommLLM, providers, routing, compat).
- `somm-service`: starlette web admin + HTTP API + scheduler + 3 workers.
- `somm-mcp`: stdio MCP server.
- `somm-skill`: onboarding markdown templates for coding agents.

See [`PLAN.md`](./PLAN.md) for the full design doc including CEO /
Design / Eng / DX review findings + 48-row decision audit trail.

## Contributing

Run tests:

```bash
uv sync --all-packages
uv run pytest packages/
```

Live-provider tests (ollama) are auto-skipped when unavailable. VCR-style
fixtures cover provider-specific parsing quirks.

Before opening a PR that touches provider adapters, run the privacy +
XSS + migration suites:

```bash
uv run pytest packages/ -k "privacy or xss or migration or spool"
```

## License

TBD. Until a license file is added, no license is granted. If you'd
like to use this code beyond reading, please open an issue.

## Status

v0.1.0-dev — all core functionality working; not yet published to PyPI.
See [`CHANGELOG.md`](./CHANGELOG.md) for milestones.

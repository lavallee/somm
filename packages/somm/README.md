# somm

**Self-hosted LLM telemetry, routing, and intelligence loop.**

The main library: `SommLLM` — one call wraps telemetry, provider routing
across ten providers, tool calling, streaming, embeddings, multimodal
dispatch, cost tracking, budget gates, online evaluation, durable eval
gates, prompt optimization proposals, experiment campaigns, and
cross-project model memory (the sommelier). It also exposes a neutral
one-attempt harness API for Claude Code, Codex, and OpenCode so task runners
do not reimplement CLI and event-stream adapters. Zero-config, privacy-first,
no phone-home.

```python
import somm

llm = somm.llm(project="my_app")
result = llm.generate(prompt="Reply with exactly: pong", workload="ping")
print(result.text, result.provider, result.cost_usd)
```

Async code can call `await llm.agenerate(...)`, `await llm.aembed(...)`,
`await llm.agenerate_structured(...)`, or iterate `llm.astream(...)`. These
methods use the same synchronous routing, embedding, governance, and telemetry
paths without blocking the caller's event loop.

```python
from somm import harnesses
from somm.harnesses import HarnessRequest

result = harnesses.run("codex", HarnessRequest(
    prompt="Fix the failing tests",
    cwd="~/src/my-project",
    capture_dir="./runs/task-1",
))
print(result.outcome, result.final_text)
```

Full documentation, design docs, and examples live in the
[somm repository](https://github.com/lavallee/somm).

### New-project Ollama default and text-only Claude CLI calls

The Ollama default remains `qwen3:8b` for compatibility; Somm does not
automatically substitute an installed model. If it is absent, install it with
`ollama pull qwen3:8b`, or run `ollama list` and select an installed generation
model using `SOMM_OLLAMA_MODEL`, `[tool.somm].ollama_model` in pyproject.toml,
or the per-call `--model` / `model=` option. A 404 now includes this guidance.

Claude CLI calls are lean by default: no built-in tools, MCP servers, or
setting sources, no persisted session (`--no-session-persistence`, skipped on a
CLI too old to know it), and a neutral working directory. A one-word prompt
costs about 3.6k input tokens instead of about 34k. With no model given the
provider runs `claude-haiku-5-5` (override with `SOMM_CLAUDE_CLI_MODEL` or
`claude_cli_model`), not the CLI's own default; the row records the model that
actually served. Configured `claude_cli_extra_args` / `SOMM_CLAUDE_CLI_ARGS`
replace the lean flag set.

`claude_cli_text_only` / `--claude-cli-text-only` is the same lean set, asserted
explicitly for untrusted text:

```sh
somm generate --provider claude-cli --claude-cli-text-only --json "Summarize this text"
```

```python
result = llm.generate(
    untrusted_text,
    provider="claude-cli",
    claude_cli_text_only=True,
    allow_fallback=False,
)
```

It requires a Claude CLI provider pin and disabled fallback, and is rejected
when `claude_cli_extra_args` / `SOMM_CLAUDE_CLI_ARGS` are configured. This mode
controls CLI tool access; it is not an operating-system sandbox.

`somm generate` exits 0 only for an OK result, and nonzero for failed outcomes,
including empty responses. JSON results retain `ok`, `outcome`, and
`error_kind`, and include `error_detail` for actionable diagnostics.

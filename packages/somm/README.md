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

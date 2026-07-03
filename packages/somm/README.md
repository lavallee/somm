# somm

**Self-hosted LLM telemetry, routing, and intelligence loop.**

The main library: `SommLLM` — one call wraps telemetry, provider routing
across ten providers, tool calling, streaming, embeddings, multimodal
dispatch, cost tracking, budget gates, online evaluation, and
cross-project model memory (the sommelier). Zero-config, privacy-first,
no phone-home.

```python
import somm

llm = somm.llm(project="my_app")
result = llm.generate(prompt="Reply with exactly: pong", workload="ping")
print(result.text, result.provider, result.cost_usd)
```

Full documentation, design docs, and examples live in the
[somm repository](https://github.com/lavallee/somm).

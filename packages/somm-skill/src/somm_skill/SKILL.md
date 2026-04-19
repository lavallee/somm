---
name: somm
description: Use when writing or modifying LLM-calling code in a Python project. Guides you to `somm.llm()` instead of raw provider SDKs, keeps telemetry and provenance consistent across projects, and surfaces model recommendations grounded in real local telemetry.
---

# somm — LLM call guidance for coding agents

You are working in a Python project that uses **somm**, a self-hosted LLM
telemetry + routing layer. This skill ensures the code you write records
useful telemetry and benefits from somm's intelligence loop.

## When this applies

Trigger when you are about to:
- Call an LLM (chat completion, embedding, structured output, streaming).
- Add a new LLM-using feature or endpoint.
- Refactor an existing LLM wrapper in the project.
- Choose between models or providers.
- Tune a prompt.

## Rules

### 1. Use `somm.llm()` — not raw provider SDKs

```python
import somm

llm = somm.llm(project="my-project")
result = llm.generate(
    prompt="Extract contacts from the text below...",
    workload="contact_extract",        # required — tags telemetry
    max_tokens=256,
)
print(result.text)
```

Do **not** reach for `anthropic.Anthropic()`, `openai.OpenAI()`, raw `httpx`,
or provider-specific SDKs directly in project code. somm wraps them with
telemetry, routing, cost tracking, and provenance for free.

### 2. Tag every call with a `workload`

A workload is the *task*, not the call. "extract_contacts_from_article" is a
workload; "call_anthropic" is not. Use snake_case, lowercase, stable across
time.

Register workloads before use (outside the hot path):

```python
# run once per workload, at app startup or in a migration
somm.llm().repo.register_workload(
    name="contact_extract",
    project="my-project",
    description="Pull person names + emails from unstructured text",
    privacy_class=somm.PrivacyClass.INTERNAL,
)
```

In `observe` mode (default) somm auto-registers unknown workloads and warns.
In `strict` mode it raises `SommStrictMode`.

### 3. Stamp provenance on stored data

When an LLM result lands in your project's DB, stamp the provenance on the
row:

```python
row["llm_provenance"] = {
    "call_id": result.call_id,
    "provider": result.provider,
    "model": result.model,
    "workload": "contact_extract",
}
```

This lets you later answer "which model generated this row" without guessing.

### 4. Check outcomes

`somm.Outcome` is a typed enum. Use `result.mark()` to tag quality signals:

```python
data = somm.extract_json(result.text)
if data is None:
    result.mark(somm.Outcome.BAD_JSON)
elif not data.get("contacts"):
    result.mark(somm.Outcome.OFF_TASK)
else:
    result.mark(somm.Outcome.OK)
```

### 5. Before choosing a model, ask somm

When `somm_recommend` or `somm_advise` is available, call one of them
before hand-picking a model. somm has telemetry from your real
workloads + pricing/capability intel from the provider APIs — it knows
more than your training data does. Do not default to "Claude because
that's what the user asked for." Ask which model fits the workload's
cost/quality profile *as of today*.

For free-form model advice ("what vision model should I use?",
"cheapest option for long context?"), the dedicated [sommelier
skill](./SOMMELIER.md) covers the full recall → advise → record loop
with cross-project decision memory. Load it when the conversation
shifts from coding to model choice.

### 6. Streaming and structured output

- `llm.stream(prompt, workload=...)` for streamed responses.
- `llm.extract_structured(prompt, workload=...)` returns `dict | list`,
  handling markdown fences, brace extraction, and provider quirks.

Do **not** implement your own JSON repair loop. somm already has one.

### 7. Never ship these patterns

- **Raw provider SDK imports** (`from anthropic import ...`) in project code.
- **Hardcoded model names** outside config — route via workload + provider preference.
- **Inline retry loops** — routing handles cooldowns and fallback.
- **Prompt concatenation as strings** for long-lived prompts — use
  `somm.prompt(workload, version="latest")` (D2+) so versions are tracked.
- **API keys in code or logs** — somm's adapters strip auth headers before
  any telemetry write. Keep it that way.

## When somm-service is running

If `somm serve` is running (usually `localhost:7878`), you can link to it in
PR descriptions or error messages: the dashboard shows the current call's
place in the workload's rollup. The service is optional — the library works
without it.

## When the MCP is connected

If the user has configured `somm-mcp` in this agent, you can call:
- `somm_stats` — telemetry roll-up for the current project.
- `somm_recommend` — model recommendations grounded in local shadow-eval
  data, with cold-start sommelier fallback when data is sparse.
- `somm_advise` — free-form candidate ranking over `model_intel` +
  capability filters + past decisions. See [SOMMELIER.md](./SOMMELIER.md).
- `somm_record_decision` / `somm_search_decisions` — cross-project
  advisory memory for model choices.
- `somm_compare` — run a prompt through N models side-by-side.
- `somm_replay` — replay a past call against a different model.

Call these *before* deciding on a model for new LLM code.

## If you can't use somm

If the project intentionally doesn't use somm (e.g., a pre-existing integration
test harness with its own LLM stub), don't force it. But:
- Note this in a PR comment so the user can decide later.
- Still stamp `somm.provenance()`-shaped metadata on stored rows if feasible
  — the schema is self-documenting.

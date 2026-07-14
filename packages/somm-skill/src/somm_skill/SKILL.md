---
name: somm
description: Use when writing or modifying LLM-calling code in a Python project. Guides you to `somm.llm()` instead of raw provider SDKs, keeps telemetry and provenance consistent across projects, wires new workloads into online evaluation and budget/quota pacing, and surfaces model recommendations grounded in real local telemetry.
---

# somm — LLM call guidance for coding agents

You are working in a Python project that uses **somm**, a self-hosted LLM
telemetry + routing layer. This skill ensures the code you write records
useful telemetry and benefits from somm's intelligence loop.

## When this applies

Trigger when you are about to:
- Call an LLM (chat completion, tool calling, embedding, structured
  output, streaming).
- Add a new LLM-using feature, agent loop, or endpoint.
- Refactor an existing LLM wrapper in the project.
- Choose between models or providers.
- Tune a prompt.
- Launch Claude Code, Codex, or OpenCode as an autonomous workspace agent.

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
or provider-specific SDKs directly in project code. somm wraps ten providers
(ollama, OpenRouter, DeepSeek, Minimax, Anthropic, Gemini, OpenAI,
Perplexity, `claude`/`codex` CLI seats) with telemetry, routing, cost
tracking, and provenance for free.

If you are integrating a tool that only accepts a provider-compatible HTTP
base URL, run `somm serve` and point it at the authenticated proxy instead:
`/v1/messages` for Anthropic Messages (including `stream: true` SSE) or
`/v1/chat/completions` for OpenAI Chat Completions. Still pre-register and
send `X-Somm-Workload`; the proxy uses the same budget gate and telemetry
ledger as `somm.llm()`.

### 2. Tag every call with a `workload`

A workload is the *task*, not the call. "extract_contacts_from_article" is a
workload; "call_anthropic" is not. Use snake_case, lowercase, stable across
time.

Register workloads before use (outside the hot path), with a budget cap:

```python
# run once per workload, at app startup or in a migration
somm.llm().repo.register_workload(
    name="contact_extract",
    project="my-project",
    description="Pull person names + emails from unstructured text",
    privacy_class=somm.PrivacyClass.INTERNAL,
    budget_cap_usd_daily=2.0,      # always set one for paid-provider workloads
)
```

In `observe` mode (default) somm auto-registers unknown workloads and warns.
In `strict` mode it raises `SommStrictMode`.

### 3. Opt quality-sensitive workloads into online evaluation

If output quality matters (extraction, verification, enrichment — anything
you'd hand-check), attach a shadow config so somm samples production calls,
captures bodies, and grades them against a gold model in the background:

```python
wl = llm.repo.workload_by_name("contact_extract", "my-project")
llm.repo.set_shadow_config(wl.id, {
    "gold_provider": "claude-cli",   # subscription seat = zero marginal cost gold
    "gold_model": "sonnet",
    "sample_rate": 0.03,             # capture+grade 3% of OK calls
    "budget_usd_daily": 0.5,
    "max_grades_per_run": 20,
})
```

Capture is the documented consent for body storage; `privacy_class=PRIVATE`
workloads are never captured regardless. Grading needs the workers running
(see rule 8).

### 4. Stamp provenance on stored data

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

### 5. Check outcomes

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

### 6. Before choosing a model, ask somm — and respect billing mode

When `somm_recommend` or `somm_advise` is available, call one of them
before hand-picking a model. somm has telemetry from your real
workloads + pricing/capability intel — it knows more than your training
data does.

Providers bill in two shapes, and somm models both (`somm plans`):
**PAYG** (per-token dollars — the constraint is spend rate) and
**metered** (subscription quota — the constraint is window headroom;
`cost_usd` is notional). Candidate reasons include plan headroom and
pace ("metered plan 62% used, pace 1.3x"). Don't aim a new high-volume
workload at a metered provider that's already over pace — the router
will deprioritize it anyway. The `claude-cli`/`codex-cli` seat providers
are deliberately **pinned-only**: excellent for gold grading and
quality-sensitive low-volume calls, never for hot loops.

For free-form model advice, the dedicated [sommelier
skill](./SOMMELIER.md) covers the full recall → advise → record loop
with cross-project decision memory.

### 7. Tool calling, streaming, embeddings, structured output

- **Tool calling** — one neutral schema, every provider translates:

  ```python
  result = llm.generate(
      messages=[{"role": "user", "content": "..."}],
      tools=[{"name": "get_weather", "description": "...",
              "parameters": {...}}],       # parameters = JSON Schema
      workload="agent_loop",
  )
  for call in result.tool_calls: ...       # result.stop_reason == "tool_use"
  ```

- `llm.stream(prompt, workload=...)` for streamed responses
  (`<think>` blocks stripped).
- `llm.embed(text, workload=...)` for embeddings (telemetry row like
  any call).
- `llm.extract_structured(prompt, workload=...)` returns `dict | list`,
  handling markdown fences, brace extraction, and provider quirks.
  Do **not** implement your own JSON repair loop.

### 8. Turn the intelligence loop on

A project that only records calls gets half of somm. Set
`SOMM_INPROCESS_WORKERS=1` in the project's environment (requires
`somm-service` installed) so the scheduler runs model-intel refresh,
online-eval grading, and the recommendation agent inside the app's own
processes — no dedicated service needed. Alternatively run `somm serve`.
somm warns at startup when grading is configured but no worker has ever
run.

### 9. Use `somm.harnesses` for autonomous coding-agent attempts

Do not confuse a subscription-seat provider call with a coding-agent run.
`llm.generate(provider="claude-cli")` is intentionally an isolated,
single-shot text generator. When the task needs repository access, tools, or
native session resume, use the harness API:

```python
from somm import harnesses
from somm.harnesses import HarnessRequest

result = harnesses.run("codex", HarnessRequest(
    prompt="Implement the accepted change",
    cwd=repo,
    capture_dir=run_dir,
    correlation_id=task_id,
), timeout=1800)
```

Schedulers that own cancellation or inactivity checks should use
`harnesses.start()` and `harnesses.inspect()`. Keep `allow_unsafe=False`
unless the caller has explicitly authorized and isolated the workspace. Somm
executes one attempt; queues, retries, failover, verification, and releases
belong to the outer task runner.

### 10. NEVER ship these patterns

These are guardrails, not style preferences — DO NOT weaken or remove them.

- NEVER import **raw provider SDKs** (`from anthropic import ...`) in project code.
- NEVER hardcode **model names** outside config — route via workload + provider preference.
- NEVER write **inline retry loops** — routing handles cooldowns, fallback, and
  metered-plan pacing.
- DO NOT concatenate **prompts as strings** for long-lived prompts — use
  `llm.register_prompt(...)` / `llm.prompt(workload)` so versions are tracked.
- NEVER put **API keys in code or logs** — somm's adapters strip auth headers before
  any telemetry write. Keep it that way.
- NEVER ship **unbudgeted paid workloads** — set `budget_cap_usd_daily`;
  `SOMM_BUDGET_FAIL_CLOSED=1` makes it a hard gate.

## CLI quick reference (for debugging sessions)

```bash
somm status --json           # machine-readable project/global status
somm generate "prompt" --workload ad_hoc --json
somm plans --learn           # metered quota pacing + learned ceilings
somm inbox list              # recommendation inbox
somm inbox apply <id>        # apply a recommendation + record decision
somm eval run --workload W --dataset D
somm spend                   # today's spend vs budget caps
somm doctor                  # config / db / intel / workers / cooldowns health
somm backfill-costs          # recompute $0 calls after pricing intel improves
somm drain-spool             # replay telemetry spooled during db outages
```

## When the MCP is connected

If the user has configured `somm-mcp` in this agent, you can call:
- `somm_stats` — telemetry roll-up for the current project.
- `somm_search_calls` — query the call log by filters.
- `somm_recommend` — model recommendations grounded in local online-eval
  data, with cold-start sommelier fallback when data is sparse.
- `somm_inbox` / `somm_apply_recommendation` /
  `somm_dismiss_recommendation` — inspect and action routing
  recommendations.
- `somm_advise` — free-form candidate ranking over `model_intel` +
  capability filters + plan headroom + canonicalized past decisions.
  See [SOMMELIER.md](./SOMMELIER.md).
- `somm_record_decision` / `somm_search_decisions` — cross-project
  advisory memory for model choices.
- `somm_register_workload` / `somm_register_prompt` — commit workload
  definitions and prompt versions.
- `somm_eval_promote_call` — copy a sampled call into a durable eval
  dataset.
- `somm_compare` — run a prompt through N models side-by-side (reaches
  every configured provider, CLI seats included).
- `somm_replay` — replay a past call against a different model.

Call these *before* deciding on a model for new LLM code.

## If you can't use somm

If the project intentionally doesn't use somm (e.g., a pre-existing integration
test harness with its own LLM stub), don't force it. But:
- Note this in a PR comment so the user can decide later.
- Still stamp `somm.provenance()`-shaped metadata on stored rows if feasible
  — the schema is self-documenting.

External systems that want to observe somm without coupling can attach via
`somm.hooks` (a correlation-id provider stamps your trace/request id on every
telemetry row; call observers receive an event per call).

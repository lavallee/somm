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
workloads are never captured regardless.

Sensitive data: give the workload a data boundary
(`data_boundary="local" | "zdr" | "subscribed"`, or `somm boundary set`). Routes
outside it are refused with `SommPrivacyViolation` (never retried, never a
looser fallback); a private workload is always `local`. Check
`somm boundary audit` before tightening. See docs/data-boundary.md. Grading needs the workers running
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

### 5. JSON, failures and timeouts: use somm's, write none of your own

**Need JSON? Make one call.** Do not write a parser, a fence stripper, a
brace slicer or a "return only JSON" re-ask:

```python
contacts, result = llm.generate_structured(
    prompt,
    schema=Contacts,             # Pydantic model, JSON Schema dict, or callable;
    workload="contact_extract",  # omit to use the workload's output_schema
    timeout=60,
)
# result.provider / .model / .call_id / .cost_usd say who served it;
# result.parsed is the JSON as a dict or list.
```

It sends the schema natively where the provider supports it and in the
prompt everywhere, repairs common JSON damage, marks `bad_json` on the
ledger and retries with the error fed back. When every attempt fails it
raises `somm.SommStructuredError`; `e.delivery_failure` True means the model
never answered (back off), False means it answered badly.

**Failures: pass `raise_on_failure=True` and catch types.** Never match on
error message strings or test `result.text` for emptiness:

```python
try:
    result = llm.generate(prompt, workload="brief", raise_on_failure=True, timeout=120)
except somm.SommRateLimited as e:       # e.retry_after_s
    ...
except somm.SommProviderNotConfigured:  # pinned provider has no key/binary here
    ...
except somm.SommError as e:             # everything else; e.provider, e.model, e.call_id
    ...
```

Other types: `SommTimeout`, `SommProvidersExhausted`, `SommAuthError`,
`SommModelNotFound`, `SommEmptyResponse`, `SommRefused` (the model or a
content filter declined), `SommPrivacyViolation`. Without the flag a failed
call returns a result with `result.ok == False` and `result.error` saying
why; that default exists for old callers, so new code should set the flag.

**Timeouts: `timeout=` per call.** It reaches every adapter. Do not set
`SOMM_HTTP_TIMEOUT` in a unit file for one slow workload or assign
`provider.timeout`.

**Retries: a workload policy, not a loop.**
`llm.set_workload_policy("brief", {"retry": {"max": 3, "backoff_s": 5},
"timeout_s": 120})` covers every unpinned call on that workload. Check
`llm.health()` (configured, cooled until, last error per provider; no
network call) instead of probing providers yourself.

Tag quality signals the code can judge with `result.mark()`:

```python
if not contacts.items:
    result.mark(somm.Outcome.OFF_TASK)
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
quality-sensitive low-volume calls, never for hot loops. somm finds their
binaries on PATH, then in `~/.local/bin`, `~/.npm-global/bin`,
`~/.volta/bin`, `/usr/local/bin` and `/opt/homebrew/bin`; under a
service manager with a bare PATH, set `SOMM_CLAUDE_BIN=/path/to/claude`
or `SOMM_CODEX_BIN=/path/to/codex` (or `claude_cli_bin` / `codex_cli_bin`
under `[tool.somm]`). A pin that fails with `not configured` says which
of those it searched.

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

- `llm.stream(prompt, workload=...)` for streamed responses (`<think>`
  blocks stripped). It is `generate`'s call path — same policy, pins,
  boundary, budget and arguments (`messages=`, `tools=`, …). An attempt
  falls back until its first chunk; a failure after that is recorded and
  raises, so callers that need an answer on a mid-stream failure must retry
  the whole stream themselves. `events=True` yields `somm.events`
  (`StreamStart`, `TextDelta`, `ToolCallDelta`, `ReasoningDelta`, and a final
  `StreamDone` with the recorded result and real usage when the provider
  reports it).
- `llm.embed(text, workload=...)` for embeddings (telemetry row like
  any call).
- **Structured output** is `llm.generate_structured(...)` (rule 5).
  `extract_structured` is its older, schema-less form that returns a
  `{"_somm_parse_err": True}` sentinel on failure; keep it where it is
  already used, but do not reach for it in new code.
- **Async** — `agenerate`, `astream`, `aembed`, `agenerate_structured` and
  `aextract_structured` are native. Do not wrap the sync calls in
  `asyncio.to_thread` yourself; the async path keeps the same routing,
  fallback, budget gate, hooks and telemetry.
- **Pins are sticky** — `provider=` and/or `model=` means "serve this call
  with that". If the pinned target fails, the call comes back
  `outcome=UPSTREAM_ERROR` (or raises, with `raise_on_failure=True`) with
  the pinned attribution intact; somm never
  answers from a model the caller did not name. Pass `allow_fallback=True`
  (or set `SOMM_PINNED_FALLBACK=1`) when an answer from some healthy model
  beats no answer — e.g. batch workers that must not lose a run when one
  provider drops.
  Only pin on purpose. A wrapper that forwards its own default `provider=`
  on every call turns off routing: one bad key on that provider then fails
  every call. Leave `provider=None` and let the chain decide. An unpinned
  call routes past a provider whose key is rejected and reports it through
  `on_fallback`.
  Claude CLI attribution confirms a pin against `modelUsage` keys or
  `canonicalModel`; without a match, it selects the largest prompt volume
  (including cache reads/writes), then output tokens when all prompts are
  zero. Auxiliary passes therefore do not relabel short pinned answers.
- **Images** — build them with `somm_core.parse.image_prompt()` and pass
  the list as `prompt`. Every adapter translates the block for its wire
  format. Do not hand-build OpenAI `image_url` or Ollama `images` payloads:
  somm accepts them, but the neutral block is the one every provider is
  tested with.
- **Reasoning budget** — pass `reasoning_effort=` (`none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`) on any generate call to dial how much a
  model thinks before answering. It reaches only providers that accept it.
  Measure before assuming more is better: it is workload-dependent, and a
  budget spent reasoning is a budget not spent answering — a model can burn
  its whole `max_tokens` thinking and return an empty body.

  ```python
  result = llm.generate(prompt=..., workload="triage", reasoning_effort="low")
  ```

- **Decisions (classification-shaped work)** — when the answer is a yes/no,
  one of a fixed set of labels, or a level on a scale, use `llm.decide`
  instead of prompting an LLM and parsing its text. It calls a decision
  model (TypeSafe System One, key in `TYPESAFE_API_KEY`) and returns a label
  with a probability, recorded and priced on the ledger like any call. Do
  not call the decision API directly: a direct call has no cost row, no
  boundary or budget gate, and no calibration.

  ```python
  r = llm.decide(ticket, ask="Is this about billing?", workload="ticket_billing")
  r = llm.decide(ticket, ask="Which team owns this?", labels=["billing", "auth", "other"],
                 workload="ticket_route", min_p=0.8)
  if r.abstained:  # not sure enough: escalate, same workload
      value, _ = llm.generate_structured(ticket, schema=Route, workload="ticket_route",
                                         parent_call_id=r.call_id)
  r.verdict("auth")  # when you learn the right label; feeds calibration
  ```

  See `docs/decisions.md`.

### 8. Turn the intelligence loop on

A project that only records calls gets half of somm. The loop — model-intel
refresh, online-eval grading, recommendations — is a library command:

```bash
somm loop run --project my-project       # one pass of whatever is due, then exit
somm loop install --project my-project   # systemd user timer, every 15 minutes
somm loop status --project my-project    # timer state + worker heartbeats
```

A long-lived process can instead set `SOMM_INPROCESS_WORKERS=1` and run the
scheduler inside itself; `somm serve` does the same with a dashboard. somm
warns at startup when grading is configured but no worker has ever run, and
`somm doctor` says `intelligence loop: DORMANT` until it has.

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
unless the caller has explicitly authorized and isolated the workspace. For
Codex, set scope with `sandbox_mode=` (`"read-only"` or `"workspace-write"`)
and `ephemeral=`, and pass `network_access=True` when a `workspace-write` task
needs the network (it is off by default). The Codex adapter rejects
`allow_unsafe=True` and `extra=` arguments, and every adapter rejects scope
fields it cannot enforce. A missing `cwd` raises rather than falling back to home. Somm
executes one attempt; queues, retries, failover, verification, and releases
belong to the outer task runner.

To export completed-call spans, install `somm[otel]` and set
`SOMM_OTEL_ENDPOINT` to the collector's full OTLP HTTP/protobuf traces endpoint
(including `/v1/traces` when the collector requires it). Leave it unset to keep
OpenTelemetry hooks and background processors disabled. Applications that
already own an SDK tracer provider should instead call
`somm.plugins.otel_exporter.register(tracer_provider=provider)` and retain
responsibility for that provider's shutdown.

### 10. NEVER ship these patterns

These are guardrails, not style preferences — DO NOT weaken or remove them.

- NEVER import **raw provider SDKs** (`from anthropic import ...`) in project code.
- NEVER hardcode **model names** outside config — route via workload + provider preference.
- NEVER write **inline retry loops** — routing handles cooldowns, fallback, and
  metered-plan pacing; a workload policy sets retries and backoff.
- NEVER write **your own JSON parser or repair pass** — use
  `generate_structured` (rule 5).
- NEVER detect failure by **matching error strings** — pass
  `raise_on_failure=True` and catch the typed `somm.Somm*` errors.
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
#   one JSON object on stdout: ok, text, provider, model, tokens_in/out, latency_ms,
#   cost_usd, call_id, outcome; a failed call (exit 1) adds error_kind,
#   error_message (the readable reason, <=400 chars) and error_detail.
#   A usage error exits 2 with {"ok": false, "error": {...}} on stderr.
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


## Adaptive subscription routing (0.20)

Opt in with `model="somm/auto"` and a workload `policy.adaptive` containing
`fast`, `strong`, `fallback` targets plus application-specific `fast_patterns`.
Unknown/biographical/research work selects strong; tool results and short confirmations
do not demote the task. At 90% weekly ChatGPT usage (or unknown/stale usage), only simple
work uses the non-ChatGPT fallback; strong work raises `SommDeferred` or returns proxy
HTTP 429 `SOMM_WORK_DEFERRED`. The caller must retain/resume it; Somm does not enqueue.
Explicit model pins remain explicit. Inspect `routing_json` and actual served models.

Use `somm chatgpt login`, `status`, `models`, `quota`; install `somm[chatgpt]` first.
Somm owns its OAuth file (`SOMM_CHATGPT_AUTH_FILE`), public Responses inference and
rotating refresh. Never borrow another app's credential file or use private endpoints.
Weekly metadata comes separately from supported Codex account RPC (10080-minute window,
not an assumed primary/secondary position). Only weekly percentage gates subscription
routing; API dollar caps still apply to fallback calls. Model catalog visibility is
not a successful inference test. See `docs/adaptive-routing.md` for the full contract.


### Recovery and retries (0.20.1)

ChatGPT replay repairs expanded/normalized tool batches before inference using the
same model and completed results. Inspect `routing_json.recovery`; never rerun a tool
merely to rebuild LLM context. Missing/orphan results require caller repair, not guessed
success. HTTP proxy errors preserve canonical code, retryable, retry timing, and call/
provider/model provenance. Honor `x-should-retry:false` for fatal, deferred and budget
refusals; retry only transient failures with the supplied delay. Do not downgrade a
strong-model task or bypass weekly conservation as a recovery strategy.

# OpenTelemetry export

somm can emit OpenTelemetry spans for every completed call and every recorded
harness attempt. The spans follow the OpenTelemetry GenAI semantic conventions,
so they open correctly in any OTel backend. Install the extra
(`pip install somm[otel]`); without it nothing is imported and nothing runs.

## Convention version

The GenAI conventions are still marked **Development** upstream and publish no
schema URL yet. somm therefore pins the repository commit it was written
against, in one place (`somm.plugins.otel_exporter`):

| Constant | Value |
| --- | --- |
| `GENAI_SEMCONV_REPO` | <https://github.com/open-telemetry/semantic-conventions-genai> |
| `GENAI_SEMCONV_COMMIT` | `e6658f14c9d9ea0760825f029cb82e7673f48fab` (2026-11-06) |
| `CORE_SEMCONV_VERSION` | `1.45.0` (for `error.type`) |

Sources: `docs/gen-ai/client-inference.md`, `client-invoke-agent.md` and
`docs/registry/attributes/gen-ai.md` at that commit. A test holds an allow-list
of the `gen_ai.*` names somm may emit; changing a name is a deliberate edit to
the constant, this page and that list together.

## Pointing somm at a collector

```bash
export SOMM_OTEL_ENDPOINT="http://localhost:4318/v1/traces"   # OTLP HTTP/protobuf
```

Set it before the first `SommLLM` is constructed. The endpoint is used exactly
as given. Applications that already own a tracer provider can call
`somm.plugins.otel_exporter.register(tracer_provider=...)` instead; see
[plugins.md](plugins.md) for lifecycle details. Spans nest under the caller's
active span, and start and end at the call's real start and finish times.

## LLM call span

Name `chat {model}` (just `chat` if the model is unknown), kind `CLIENT`, one per
completed call. Content is never exported: no prompts, responses, system
instructions or tool definitions.

| Attribute | Source |
| --- | --- |
| `gen_ai.operation.name` | always `chat` |
| `gen_ai.provider.name` | somm provider; `gemini` becomes `gcp.gemini`, CLI seats map to `anthropic`/`openai`, others pass through |
| `gen_ai.request.model`, `gen_ai.response.model` | the model somm recorded |
| `gen_ai.response.id` | provider response/request id when the provider returned one |
| `gen_ai.conversation.id` | somm correlation id, when set |
| `gen_ai.request.max_tokens`, `gen_ai.request.temperature` | the call's settings |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | token counts |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`, `gen_ai.usage.reasoning.output_tokens` | when reported and non-zero |
| `error.type` | on failure: somm's error kind, else the outcome (`timeout`, ...); span status is `ERROR` |
| `somm.*` | `call_id`, `outcome`, `workload`, `project`, `cost_usd`, `cost_basis`, `cost_kind`, `cost_accuracy`, `cost_source`, `pricing_version`, `observation_role`, `source_call_id`, `eval_result_id`, `provider_request_id`, `billing_id`, `origin`, `budget_eligible` |

The spec has no cost attribute, so cost lives under `somm.*`. Not emitted
because the call event does not carry them: `gen_ai.response.finish_reasons`,
and the data tier, route source and fallback depth.

The deprecated `gen_ai.system` is no longer emitted (it was replaced by
`gen_ai.provider.name`); collectors and dashboards pinned to the old name need
to read the new one.

## Harness attempt span

`record_attempt` (see the harness docs) emits one span per attempt, named
`invoke_agent {harness}`, kind `CLIENT`, starting at the attempt's start and
lasting its `duration_ms`.

| Attribute | Source |
| --- | --- |
| `gen_ai.operation.name` | `invoke_agent` |
| `gen_ai.provider.name`, `gen_ai.agent.name` | the harness name |
| `gen_ai.request.model` | the model the harness reported |
| `gen_ai.conversation.id` | the harness session id, else the correlation id |
| `gen_ai.usage.*` | as above |
| `error.type` | on failure: somm's error kind (the harness outcome, e.g. `rate_limit`) |
| `somm.call_id`, `somm.outcome`, `somm.harness_outcome`, `somm.cost_usd`, `somm.project`, `somm.origin` | ledger row and normalized result |

There are no `execute_tool` child spans: normalized harness results do not carry
per-tool events with timing.

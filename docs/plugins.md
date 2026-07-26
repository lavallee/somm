# Plugins and hooks

Somm exposes extension points around the call path. Hooks are process-wide,
best-effort callbacks: a broken hook is logged or ignored, and never prevents
the LLM call or telemetry write from completing.

## Lifecycle

One `SommLLM.generate()` call moves through these stages:

1. Resolve workload metadata and policy.
2. Run `pre_call` hooks.
3. Execute the provider route, including retries and fallback.
4. Record the immutable `calls` row.
5. Run `post_call` hooks inline.
6. Run `post_process` hooks on the background executor.

`pre_call` can rewrite the outbound request or serve a synthetic response.
`post_call` and `post_process` observe the completed call event only.

> **Serving a workload is a separate concept.** If you want a workload to be
> answered by something other than the provider chain, write a *workload
> handler*, not a hook — see [workload-handlers.md](./workload-handlers.md). A
> handler declares a workload **kind**, is bound per workload, and lands in
> telemetry as the serving source so it can be compared against the model it
> replaced. Hooks are for rewriting and observing; handlers are for serving.

## Phases

### `pre_call`

`pre_call` hooks are synchronous. Async functions are rejected at registration.
Each hook receives a mutable `PreCallContext`:

```python
workload: str
prompt: str | list[Any] | None
system: str
messages: list[Any] | None
model: str | None
provider: str | None
max_tokens: int
temperature: float
tools: list[Any]
tool_choice: Any
project: str
metadata: dict[str, Any]
```

Hooks may mutate the context in place. `project` is read-only. `metadata` is
for hook-private coordination and is not written to the call event.

A hook may also return `ShortCircuit`:

```python
text: str
provider: str = "hook"
model: str = ""
tokens_in: int = 0
tokens_out: int = 0
cost_usd: float = 0.0
raw: dict[str, Any] | None = None
tool_calls: list[Any] | None = None
source: str = ""
```

The first `ShortCircuit` wins. Later `pre_call` hooks do not run for that
request, and somm records the call as served by the hook source.

### `post_call`

`post_call` hooks receive a shallow copy of the completed call event. They run
inline after the call row is recorded. Sync and async hooks are accepted.

Use this phase for low-latency observers that need to see the event before the
caller resumes. Do not perform slow network I/O here.

### `post_process`

`post_process` hooks receive the same event shape as `post_call`, also as a
shallow copy. They run on somm's lazy single-worker background executor.

Use this phase for graders, exporters, notifiers, and other work that should
not sit on the caller's hot path. Sync and async hooks are accepted.

## Priority and isolation

Register hooks with an integer priority. Lower numbers run first. The default
priority is `100`. Equal priorities keep registration order.

Hook failures are isolated. A raising hook never breaks the call path and does
not prevent later hooks in the same phase from running. For `pre_call`, any
context mutations made before the exception are kept.

## Event schema

Call events use `HOOK_EVENT_SCHEMA_VERSION = 1`. New keys may be added, but
existing keys keep their meaning.

Current event keys:

| Key | Meaning |
|---|---|
| `schema_version` | Hook event schema version. Currently `1`. |
| `call_id` | UUID for this call event. |
| `correlation_id` | Optional ambient request id from the correlation provider. |
| `project` | Somm project name. |
| `workload` | Workload name. |
| `provider` | Provider that produced the result or final error. |
| `model` | Model name used for the call. |
| `outcome` | Final outcome string. |
| `tokens_in` | Input token count recorded for the call. |
| `tokens_out` | Output token count recorded for the call. |
| `latency_ms` | End-to-end call latency in milliseconds. |
| `cost_usd` | Rounded call cost. |
| `temperature` | Request temperature, when applicable. |
| `max_tokens` | Request token cap, when applicable. |
| `error_kind` | Exception class or normalized error kind, when present. |
| `short_circuited` | Hook source when a `pre_call` hook served the response. |

The event intentionally does not include prompt or response bodies.

## Registration

Explicit registration is direct:

```python
from somm import hooks

def observe(event):
    print(event["workload"], event["outcome"])

hooks.register_hook(hooks.POST_CALL, observe, priority=100)
```

Reference plugins are opt-in and expose `register()` functions:

```python
from somm.plugins import cache

cache.register(ttl_s=300, maxsize=512)
```

Automatic registration uses Python entry points. `somm.hooks` and
`somm.plugins` entry points must resolve to zero-argument callables that
register hooks. Somm loads both groups once, on first `SommLLM` construction.

```toml
[project.entry-points."somm.hooks"]
audit = "acme_somm.audit:register"

[project.entry-points."somm.plugins"]
redactor = "acme_somm.redactor:register"
```

## Reference plugins

The reference plugins live in `packages/somm/src/somm/plugins/`.

- `cache` (`somm.plugins.cache`): per-process in-memory cache. It installs a
  `pre_call` lookup hook and can short-circuit cache hits.
  Enable with `somm.plugins.cache.register()`.
- `redaction` (`somm.plugins.redaction`): outbound prompt/system/message
  redaction before provider calls.
  Enable with `somm.plugins.redaction.register()`.
- `notifier` (`somm.plugins.notifier`): Slack-compatible webhook notifications
  for selected events. It runs in `post_process`.
  Enable with `somm.plugins.notifier.register(webhook_url="https://...")`.
- `otel_exporter` (`somm.plugins.otel_exporter`): OpenTelemetry spans for
  completed calls. It runs in `post_process`.
  Install with `pip install somm[otel]`.

### OpenTelemetry export

Set `SOMM_OTEL_ENDPOINT` to the full OTLP HTTP/protobuf traces endpoint before
constructing the first `SommLLM`:

```bash
export SOMM_OTEL_ENDPOINT="https://collector.example.com/v1/traces"
```

Somm's built-in `somm.plugins` entry point then creates an SDK
`TracerProvider`, an OTLP HTTP/protobuf span exporter, and a
`BatchSpanProcessor`. Completed calls become `gen_ai` spans through the same
`post_process` hook used by manual registration. The endpoint is used exactly
as configured; Somm does not append `/v1/traces`.

When `SOMM_OTEL_ENDPOINT` is unset or empty, entry-point activation is a no-op:
no hook is registered, no optional OpenTelemetry module is imported, and no
batch-processing thread is started. This preserves Somm's default behavior and
dependency footprint.

Applications that already manage a tracer provider can register it directly:

```python
from somm import hooks
from somm.plugins import otel_exporter

otel_exporter.register(tracer_provider=application_tracer_provider)
```

`register()` remains idempotent and never takes ownership of a caller-supplied
provider. `otel_exporter.unregister()` is safe to call repeatedly. It removes
the hook and flushes and shuts down the provider only when Somm created that
provider from `SOMM_OTEL_ENDPOINT`; caller-supplied providers retain their
application-managed lifecycle. At interpreter shutdown, Somm drains queued
`post_process` work before applying the same cleanup to an environment-created
provider. For an explicit orderly shutdown, call
`hooks.shutdown_hooks(wait=True)` before `otel_exporter.unregister()`. If
switching configurations in a long-running process, call `unregister()` before
setting a new endpoint and registering again.

Inspect the local catalog with:

```bash
somm plugin list
somm plugin info cache
```

## Custom providers

Third-party providers register through the `somm.providers` entry-point group.
The entry point resolves to a `ProviderSpec` or a zero-argument callable
returning one. See `packages/somm/src/somm/providers/registry.py` for the full
contract.

```toml
[project.entry-points."somm.providers"]
acme = "acme_somm.provider:provider_spec"
```

The `ProviderSpec` name must not collide with a built-in provider. Its
`default_order_rank` is an integer for default routing order, or `None` for a
provider that is available only when explicitly requested.

## Minimal plugin

This `post_process` logger registers through an entry point and observes every
completed call without blocking the caller:

```python
# acme_somm/logger.py
from pathlib import Path
from somm import hooks

LOG = Path("somm-events.log")

def write_event(event):
    line = (
        f"{event['project']} {event['workload']} "
        f"{event['provider']} {event['model']} {event['outcome']}\n"
    )
    LOG.open("a", encoding="utf-8").write(line)

def register():
    hooks.register_hook(hooks.POST_PROCESS, write_event, priority=100)
```

```toml
[project.entry-points."somm.plugins"]
event_log = "acme_somm.logger:register"
```

After the package is installed in the same environment as somm, constructing
`SommLLM` loads the entry point. `somm plugin list` shows the registered
callable under `post_process`.

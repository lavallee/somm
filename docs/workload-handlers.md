# Workload handlers

A somm workload is a **unit of work**, not necessarily an LLM call.

That has quietly been true since the beginning: a workload is a named contract
with schemas, budgets, quality criteria, and SLOs, and everything somm records is
metered against it. What was never explicit is *what serves it*. The answer was
always "the provider chain." Workload handlers make that answer pluggable.

## Kinds

Every workload has a **kind**. The default is `llm`: the provider chain answers
it, exactly as before, and nothing about an unbound workload changes.

A project may register other kinds and bind workloads to them:

```python
from somm import workloads

workloads.register(MyHandler())                    # once, at startup
workloads.bind("triage", "my-kind", some="config") # per workload
```

somm knows nothing about any particular kind and takes no dependency on one.

## Writing a handler

```python
from somm.workloads import WorkloadRequest, WorkloadResult

class MyHandler:
    kind = "my-kind"

    def serve(self, request: WorkloadRequest) -> WorkloadResult | None:
        if not_my_business(request):
            return None                       # the provider chain answers
        return WorkloadResult(text=answer, cost_usd=0.0)
```

`WorkloadRequest` is a read-only view of the outbound call — workload, project,
prompt/messages (`request.content` flattens either into a string), system, model,
provider, response format, and whatever `bind()` was given as `config`. A handler
that wants to *modify* a request rather than serve it wants a `pre_call` hook
instead; see [plugins.md](./plugins.md).

**Declining is normal.** Returning `None` means "not mine, or not this time," and
the call proceeds down the provider chain exactly as it would have. A handler
that raises is treated as one that declined and the exception is logged — an
extension must never break the call path. This is what makes it safe to bind a
handler to a live workload: the worst case is the behavior you already had.

## Publishing a handler from another package

Declare a `somm.workload_handlers` entry point resolving to a zero-argument
callable that registers the handler:

```toml
[project.entry-points."somm.workload_handlers"]
my-kind = "mypkg.somm_handler:register"
```

Handlers published this way are discovered on demand, so a project that installs
your package only has to `bind()` a workload — or not, in which case nothing
changes.

## What it lands in telemetry

A handler-served call records the kind as both its `provider` and its source, so
a handler-served call and a model-served call for the **same workload id** sit in
the same `calls` table, separable by a `GROUP BY`:

```sql
select workload_id, provider, count(*), avg(latency_ms), sum(cost_usd)
from calls group by 1, 2;
```

This is the point of the whole design. "Should this workload still be an LLM
call?" stops being an argument and becomes a query over recorded cost, latency,
and outcome.

## Why: workloads climb

A workload does not have to be born knowing what it is. It can start as a plain
LLM call — the cheapest thing to write — and accumulate evidence about its own
shape. Some of what a model is doing for a workload turns out to be decidable by
a rule; some of it does not. Telemetry plus side-testing is what tells the two
apart, and a handler is where the decidable part goes once you know.

The end state is not "everything becomes deterministic." It is that every
workload ends up at the rung its evidence justifies, with the irreducible
judgment — the part a fixture cannot decide — still served by a model, and
everything else evicted from it.

The first non-LLM kind is `chip` (a portable, receipted, evaluated operational
component; see the [chip contract](https://github.com/lavallee/chip)), published
by fab. It is deliberately not special: it registers exactly the way any other
kind would.

## Ordering

somm's handler dispatch runs at `pre_call` priority 50 — after redaction hooks,
which must see the outbound text first, and after the response cache, since a
cached answer is cheaper than any handler. Handler dispatch is somm's own
internal wiring, not part of the handler contract: a handler author implements
`WorkloadHandler` and never touches hooks.

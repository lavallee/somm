# The intelligence loop

Somm has two halves. The **sensor** is the library: every call writes one row.
The **brain** is three workers that read those rows and write back something
the next call can use:

| Worker | Reads | Writes | Cadence |
|---|---|---|---|
| `model_intel` | models.dev, OpenRouter `/models`, the local Ollama tag list | `model_intel` (prices, context, capabilities), `model_aliases` | daily |
| `shadow_eval` | `samples` captured from opted-in workloads | gold and judge `calls` rows, `eval_results`, `shadow_candidates` | every pass |
| `agent` | `eval_results`, `calls`, `provider_health` | `recommendations`, `learned_param_overrides` | weekly |

Through 0.16 the brain needed a host: the web service, or a long-lived process
with `SOMM_INPROCESS_WORKERS=1`. A project that is a batch job or a CLI has
neither. The October 2026 census of this machine's fleet found the shape of
the problem: the one project with a resident process had graded 7,281 samples
and accumulated 21 recommendations nobody saw, then stopped grading when that
process stopped; every other project had nothing; and the shared ledger —
which never carries samples, evals, or recommendations — showed zero across
the board. The loop needs to be runnable without a host and visible without
opening each ledger.

## Running it

```bash
somm loop run --project my-project
```

One pass: refresh model intel if the last success is more than a day old,
grade whatever samples are waiting, emit recommendations if the last pass is
more than a week old, record a heartbeat per worker, exit. `--force` runs
every worker; `--only shadow_eval` restricts to one; `--all-projects` walks
every ledger in `~/.somm/registry.json`; `--json` prints the report.

```bash
somm loop install --project my-project [--every 15min]
somm loop status  --project my-project
somm loop uninstall --project my-project
```

`install` writes a systemd user timer that runs the command above. Nothing
else is required: no service, no resident process, no copied unit files.

Long-lived processes can still set `SOMM_INPROCESS_WORKERS=1`; `somm serve`
runs the same scheduler beside the dashboard.

## Opting a workload in

Grading is per workload and off by default, because capture stores prompt and
response bodies. `privacy_class=PRIVATE` workloads are never captured
whatever their config.

```python
wl = llm.repo.workload_by_name("claim_extract", "my-project")
llm.repo.set_shadow_config(wl.id, {
    "gold_provider": "claude-cli",   # a seat: zero marginal cost gold
    "gold_model": "claude-sonnet-5-5",
    "sample_rate": 0.03,             # capture 3% of OK calls
    "budget_usd_daily": 0.5,
    "max_grades_per_run": 20,
})
```

`sample_rate` is applied once, at capture time, against real call volume, so
candidate supply is what bounds a pass in steady state; `max_grades_per_run`
and `budget_usd_daily` are the ceilings. Each gold or judge request is an
ordinary `calls` row linked back through `source_call_id`, so the cost of
grading is visible in `somm status` like any other spend.

## Knowing whether it is alive

`somm doctor` prints the loop's state for its own ledger and for every
project ledger in `~/.somm/registry.json`; `somm doctor --global` says that
the mirror cannot see the loop and lists the project ledgers:

```
intelligence loop: not visible here — the mirror carries calls and workloads,
  not samples, evals, or recommendations; per project ledger:
  research-app   7281 evals, 21 recommendations (21 open), last worker success 2026-09-19 10:24:26
  cluster-app    idle — no evals or recommendations (0 decisions)
```

The library also warns once per process at startup when a workload is opted
in and no worker has ever run. `somm loop status` shows the timer state and
the last success per worker.

## What comes out

`somm inbox` lists recommendations; `somm inbox apply <id>` applies one and
records the decision. The MCP tools `somm_recommend`, `somm_inbox`,
`somm_apply_recommendation`, and `somm_dismiss_recommendation` do the same
from inside a coding agent. Graded samples also feed `somm frontier`, the
sommelier's ranking, and `somm eval promote-call`, which turns a graded
production call into a durable CI fixture.

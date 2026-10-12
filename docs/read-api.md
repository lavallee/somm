# Reading the ledger

Somm's ledger is SQLite and you may query it with SQL. Seven tools in one
estate did exactly that by October 2026, each with its own rollup, and the
copies disagreed about what an unregistered workload is called. This page is
the supported way to read instead.

## `somm_core.read`

Pure functions; never write. Each takes an open `sqlite3.Connection` or a
database path (opened read-only).

```python
from somm_core.read import search_calls, spend_today, fleet_summary

rows = search_calls(db_path, "my-project", since_days=7, workload="claim_extract",
                    provider="minimax", q="job-42", limit=100)
spend = spend_today(db_path, "my-project", default_cap=None)
fleet = fleet_summary(global_db_path, since_days=7)
```

- `search_calls` — filtered rows joined to their workload name. Filters:
  `since_days` or an ISO `since`, `workload` (name or id), `provider`,
  `model`, `outcome`, `origin`, and `q` (substring over id, provider, model,
  workload, session, correlation). `newest_first=False` for a tail.
- `spend_today` — today's budget-eligible spend per workload against its cap.
- `fleet_summary` — what only a shared ledger can say: rows per
  (project, client_version), whether the loop has ever produced anything,
  sentinel rows, provider spelling drift, routes without a canonical id.

**Column-stability contract.** The dict keys these functions return are
additive: a key, once published, keeps its name and meaning; new keys may
appear. Read by key, never by position. The same contract covers the `calls`
columns listed in [`BLUEPRINT.md`](./BLUEPRINT.md): a column is never renamed
or repurposed, and a row written by an older client leaves the newer columns
`NULL` — honestly unknown, not zero.

## Rollups

`Repository.stats_by_workload(project, since_days)` and
`stats_global_by_workload(since_days)` are the serving rollups behind
`somm status`, `/api/stats`, and MCP `somm_stats`: per (workload, provider,
model) counts, tokens, cache tokens, cost, latency and TTFT percentiles,
throughput, cache-read ratio, and goodput against workload SLOs. The key list
is `somm_core.repository._SERVING_STATS_KEYS`.

`somm_core.plans` holds the quota view: `limit_statuses`, `usage_in_window`,
`payg_burn_rates`, `observed_ceilings`.

## Surfaces built on these

| Surface | Needs | Notes |
|---|---|---|
| `somm status`, `somm tail`, `somm spend`, `somm doctor --global` | the library | JSON with `--json`; stable exit codes |
| MCP `somm_stats`, `somm_search_calls`, … | `somm-mcp` | for coding agents |
| `GET /api/status`, `/api/stats`, `/api/calls`, `/api/sessions`, `/api/spend/today` | `somm-service` | bearer token or loopback same-origin header |

## Identity columns a downstream ledger can join on

For Milton, cap, and other systems that reconcile somm rows with their own
records, each `calls` row carries: `id` (the gateway call id), `correlation_id`
(the caller's job/task id), `session_id` and `parent_call_id`, `provider_request_id`
and `billing_id` when the provider returned them, `prompt_hash` and
`response_hash`, `call_site` (the code that asked), `origin` (`native`,
`harness`, `foreign_imported`), `observation_role` (`production`, `shadow_gold`,
`shadow_judge`, `imported`), `client_version`, and the cost provenance quintet
(`cost_basis`, `cost_kind`, `cost_accuracy`, `cost_source`, `pricing_version`).
Harness attempts recorded through `somm.harnesses.record_attempt` carry the
harness name as `provider`, the served model, and the runner's correlation id.

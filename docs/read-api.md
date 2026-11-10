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

### Every function

Each takes an open connection or a path. Examples are two lines.

- `schema_status(db)` — call first. Says whether this ledger's schema is one
  this version understands, and which newer columns and tables it lacks.
  ```python
  s = schema_status(db)          # {"understood": True, "schema_version": 27, "missing_columns": {"calls": [...]}, ...}
  if not s["understood"]: raise SystemExit(s["reason"])
  ```
- `get_calls(db, ids)` — rows for exact call ids, shaped like `search_calls`.
  ```python
  rows = get_calls(db, ["3f2a…", "9c41…"])
  by_id = {r["id"]: r for r in rows}
  ```
- `provider_health(db, project=None, since_days=7, workload=, provider=, model=)`
  — per (provider, model): calls, ok rate, p50/p95 latency of ok calls, last
  ok time, last error time, kind and outcome. `project=None` spans the ledger.
  ```python
  for h in provider_health(db, "my-project", since_days=1): print(h["provider"], h["model"], h["ok_rate"], h["p95_ms"])
  down = [h for h in provider_health(db) if h["last_ok_ts"] is None]
  ```
- `rollup(db, project=None, by=("day",), since_days=30, …)` — totals grouped by
  any of `day`, `project`, `workload`, `provider`, `model`, `outcome`,
  `data_tier`. `cost_usd` counts only calls somm could price;
  `unknown_cost_calls` says how many it could not (their 0.0 is not "free").
  ```python
  rollup(db, "my-project", by=["day", "provider"], since_days=14)
  rollup(db, by=[], provider="minimax", since="2026-11-09T00:00:00")   # one total row
  ```
- `table_rows(db, table, since=, until=, project=, limit=)` — all columns of
  `calls`, `workloads`, `eval_results`, `eval_receipts`, `campaigns`,
  `campaign_events`, `decisions`, `recommendations` or `call_updates`, oldest first.
  ```python
  open_recs = [r for r in table_rows(db, "recommendations") if r["applied_at"] is None and r["dismissed_at"] is None]
  decisions = table_rows(db, "decisions", project="my-project")
  ```
- `across_ledgers(read_fn, ledgers=None, pass_project=False, **kwargs)` — one
  read over every project ledger in the registry; rows gain `ledger_project`;
  missing or unreadable ledgers come back in `skipped`, never silently.
  ```python
  out = across_ledgers(provider_health, since_days=1)       # {"rows": [...], "skipped": [{"project", "db_path", "reason"}]}
  out = across_ledgers(spend_today, pass_project=True, default_cap=None)
  ```
- `somm_core.vocabulary.vocabulary()` — the value sets to map against (below).

`search_calls` rows also carry `cost_known`, and `route_source` and `route`
parsed from `routing_json` (the 0.23.0 fields; `data_tier` was already there).
For which provider and model served a workload last:
`search_calls(db, project, workload=w, outcome="ok", limit=1)`.

Quota and plan burn are not duplicated here: use `somm_core.plans`
(`limit_statuses`, `usage_in_window`, `payg_burn_rates`).

### Speed

Windowed reads with the default `role="production"` use `idx_calls_role_ts`,
and with a `project` use `idx_calls_project_ts`. Two shapes scan the whole
`calls` table: `role=None` with no `project`, and `table_rows("calls",
since=…)` with no `project`. Pass a `project` on a large ledger.

### Older ledgers

Reads never crash on a ledger written by an older somm. A `calls` column the
ledger predates reads `None` (a filter on it matches nothing); a table it
predates gives no rows. `schema_status` lists exactly what is missing. A ledger
newer than the library, or without the base `calls` columns, is reported as not
understood; upgrade somm or skip it.

### Vocabulary

```python
from somm_core.vocabulary import vocabulary
v = vocabulary()
missing = set(v["outcomes"]) - set(MY_OUTCOME_MAP)       # fail at start-up, not on the first new row
```

Keys: `vocabulary_version`, `somm_version`, `schema_version`; `outcomes` (each
with `failure_class`, `capability_signal`, `detractor`); `error_kinds`,
`cost_sources` (both `open: True` with `known` names; `cost_sources` has
`patterns` and `unknown_prefix`); `data_tiers` (strictest first);
`route_sources` (value to "chosen by evidence"); `cost_basis`, `cost_kind`,
`cost_accuracy`; `observation_roles`; `origins`; `meter_units` (`call` and
`plan_limit`); `plan_modes`. `vocabulary_version` changes only when a key is
removed or a meaning changes; new values and keys are additive. An `open` set
is one somm does not enumerate (an exception class name, say): map the known
values and keep a default.

### Replace this SQL with that call

| Hand-written SQL | Call |
|---|---|
| `SELECT … FROM calls WHERE project=? AND ts>=…` | `search_calls(db, project, since_days=N)` |
| `… FROM calls WHERE id IN (…)` (with the workload name) | `get_calls(db, ids)` |
| `SELECT provider, COUNT(*), SUM(outcome='ok')/COUNT(*) … GROUP BY provider` | `provider_health(db, project)` |
| latency sorted, then a percentile in Python | `provider_health(...)` (`p50_ms`, `p95_ms`) |
| `SELECT date(ts), COUNT(*), SUM(cost_usd) … GROUP BY date(ts)` | `rollup(db, by=["day"])` |
| `… GROUP BY model, provider` / `… GROUP BY workload_id` | `rollup(db, by=["model", "provider"])` / `by=["workload"]` |
| `SUM(CASE WHEN outcome='empty' …) … WHERE provider='x' AND ts>=?` | `rollup(db, by=["outcome"], provider="x", since=…)` |
| `SELECT … FROM eval_results / recommendations / decisions / campaigns …` | `table_rows(db, "<table>")` |
| a loop opening each `calls.sqlite` in the registry | `across_ledgers(fn, **kwargs)` |
| `PROVIDER_OUTCOME[call["outcome"]]` hard-coded | `vocabulary()["outcomes"]` checked at start-up |

Still hand-written: joins across tables beyond the workload name, free-text
searches on `error_detail`, and anything over `call_updates` beyond the rows.

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

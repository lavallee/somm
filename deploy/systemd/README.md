# somm-service systemd units — PROPOSAL, not installed

These are template (`@.service`/`@.timer`) systemd **user** units for running
somm's service-tier admin commands (`somm-serve admin run-shadow`,
`somm-serve admin run-agent`) and the new exhaustion regression check
(`somm doctor --max-exhausted-rate`) on a schedule, per-project, without
needing a long-running host process to embed the in-process scheduler in
(`SOMM_INPROCESS_WORKERS=1` — see `packages/somm-service/src/somm_service/inprocess.py`).

**Nothing here is installed or enabled.** No file has been copied into
`~/.config/systemd/user/`, and no `systemctl --user enable` has been run.
This is a reviewable proposal for the operator to install by hand.

## Why a standalone-timer path, given one adopter already runs this in-process

Investigating step 2 of this work found that the shadow-eval/agent/model-intel
workers are *already* running continuously in production — just embedded
inside that adopter's own long-running web-server process via
`SOMM_INPROCESS_WORKERS=1`, writing to its project-local
`.somm/calls.sqlite` rather than the machine-wide `~/.somm/global.sqlite`
(which only mirrors `calls`/`workloads`, not `samples`/`eval_results`/
`recommendations`/`worker_heartbeat` — see `cross_project_enabled` in
`packages/somm-core/src/somm_core/config.py`). That's a valid, cheaper
deployment shape for a project that already has a long-lived process. These
units exist for the *other* case: a somm-instrumented project with no such
process (a batch job, a short-lived CLI tool, a project that only runs
on-demand) that still wants shadow-eval/agent/exhaustion-alarm coverage.

## Install (manual — operator does this, not automated)

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/somm-shadow-eval@.service deploy/systemd/somm-shadow-eval@.timer \
   deploy/systemd/somm-agent@.service        deploy/systemd/somm-agent@.timer \
   deploy/systemd/somm-exhaustion-alarm@.service deploy/systemd/somm-exhaustion-alarm@.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload

# Per project you want covered, e.g. myproject:
systemctl --user enable --now somm-shadow-eval@myproject.timer
systemctl --user enable --now somm-agent@myproject.timer
systemctl --user enable --now somm-exhaustion-alarm@myproject.timer
```

Each unit assumes `somm`/`somm-serve` are on `PATH` for the user session
(installed via `uv tool install` or a venv `bin/` on `PATH`) and that the
project is discoverable either via `~/.somm/registry.json` (auto-registered
the first time the project's somm client runs) or an explicit
`SOMM_DB_DIR=` line uncommented in the `.service` file.

## Cost summary (see each unit file's comments for detail)

- **shadow-eval**: bounded per run by `max_grades_per_run` (workload config,
  default 20) × number of workloads with `shadow_config_json` set, further
  throttled upstream by each workload's `sample_rate` against real call
  volume. For one adopter's current 3 configured workloads (`hg_verify`,
  `artifact_extract`, `enrichment`; gold model `claude-cli`/`sonnet`,
  a metered/notional-cost subscription seat, NOT a per-token API charge):
  worst-case 60 graded calls/cycle, observed real-world ~100-270 graded
  calls/**day** across 96 cycles/day (candidate supply, not the per-run cap,
  is the binding constraint in steady state). Dollar cost recorded is $0.00
  (see the incident/cost note in `somm-shadow-eval@.service` — this is a
  gap in `budget_usd_daily` enforcement for metered/notional providers,
  worth fixing separately if a workload's gold model is ever a PAYG API).
- **agent**: no provider spend — reads existing telemetry/shadow-eval scores.
- **exhaustion-alarm**: no provider spend — two local `COUNT(*)` queries.

## Files

| File | Purpose |
|---|---|
| `somm-shadow-eval@.service` / `.timer` | One bounded `somm-serve admin run-shadow --project %i` pass every 15 min |
| `somm-agent@.service` / `.timer` | One `somm-serve admin run-agent --project %i` pass every 24h |
| `somm-exhaustion-alarm@.service` / `.timer` | `somm doctor --project %i --max-exhausted-rate 0.30` every hour — see `docs/incidents/2026-07-provider-exhaustion.md` |

# systemd user units

Two things a somm project may want to run on a schedule, and how.

## The intelligence loop: `somm loop install`

Nothing in this directory is needed for the loop. From a project:

```bash
somm loop install --project myproject            # every 15 minutes
somm loop install --project myproject --every 1h
somm loop status --project myproject
somm loop uninstall --project myproject
```

`install` writes `~/.config/systemd/user/somm-loop@.service` and
`somm-loop@.timer` and enables `somm-loop@myproject.timer`. Each firing runs
`somm loop run --project myproject`: one pass that refreshes model intel when
it is more than a day old, grades sampled production calls against their gold
model, and emits recommendations when the last pass is more than a week old,
then exits. Cost is bounded by each workload's `shadow_config_json`
(`max_grades_per_run`, `budget_usd_daily`) and by `sample_rate`, which is
applied once at capture time against real call volume.

Long-lived processes can instead set `SOMM_INPROCESS_WORKERS=1` and run the
scheduler inside themselves; `somm serve` does the same with a dashboard.

Through 0.16 this directory held proposal units for `somm-serve admin
run-shadow` and `run-agent`. They were never installed, and that was the
finding: a loop that needs the web-service package and a hand-copied unit
does not get run. The loop is now a library command.

## The exhaustion alarm: `somm-exhaustion-alarm@`

`somm doctor --project %i --max-exhausted-rate 0.30` every hour — exits
nonzero when more than 30% of the trailing day's calls ended in
`SommProvidersExhausted`. The July 2026 incident
(`docs/incidents/2026-07-provider-exhaustion.md`) ran for weeks because no
cheap trailing-window check existed. No provider spend; two local
`COUNT(*)` queries.

```bash
cp deploy/systemd/somm-exhaustion-alarm@.service deploy/systemd/somm-exhaustion-alarm@.timer \
   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now somm-exhaustion-alarm@myproject.timer
```

Both unit kinds assume `somm` is on `PATH` for the user session (`uv tool
install somm`, or a venv `bin/` on `PATH`) and that the project is
discoverable via `~/.somm/registry.json` (auto-registered the first time the
project's somm client runs) or an explicit `Environment=SOMM_DB_DIR=` line.

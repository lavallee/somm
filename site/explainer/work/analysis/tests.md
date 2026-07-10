## Purpose

This test area protects Somm’s zero-config behavior, cost accounting, provider-plan governance, and release hygiene. It verifies that configuration resolves predictably, pricing works offline, usage limits influence routing safely, and private project details do not leak into the public repository.

## How it works

Configuration tests clear all relevant environment variables, call `somm_core.config.load()`, and verify defaults plus precedence among explicit environment settings, `pyproject.toml`, an existing project-root `.somm`, the project registry, and a new local `.somm` directory (`tests/test_config_load.py:44`, `tests/test_config_load.py:78`, `tests/test_config_load.py:89`, `tests/test_config_load.py:102`, `tests/test_config_load.py:135`). Plan tests load TOML into provider plans containing modes, quotas, units, windows, and soft utilization targets. They exercise calendar and rolling-window math, aggregate requests, tokens, or spend across registered SQLite databases, classify usage as `ok`, `over_pace`, or `exhausted`, and confirm the router keeps normal providers first, defers over-paced ones, and removes blocked ones (`tests/test_plans.py:37`, `tests/test_plans.py:98`, `tests/test_plans.py:139`, `tests/test_plans.py:172`, `tests/test_plans.py:262`).

Pricing tests verify that a packaged JSON snapshot covers every paid routed provider and can populate model intelligence without network access (`tests/test_pricing_bundle.py:38`, `tests/test_pricing_bundle.py:47`). Bundle synchronization replaces stale seeded data, preserves manually entered prices, and skips repeat work using a fingerprint (`tests/test_pricing_bundle.py:66`, `tests/test_pricing_bundle.py:84`). Missing prices produce a zero cost plus a once-per-provider/model warning for paid providers, while free providers remain silent (`tests/test_pricing_safety.py:87`, `tests/test_pricing_safety.py:101`, `tests/test_pricing_safety.py:115`). Separately, a repository-wide scanner rejects internal names and personal filesystem paths from selected source, documentation, example, test, and CI files (`tests/test_blocklist.py:57`, `tests/test_blocklist.py:117`, `tests/test_blocklist.py:162`).

## Key surfaces

`somm_core.config.load(cwd=...)` — resolves runtime configuration and database location (`tests/test_config_load.py:49`).

`somm_core.plans.load_plans()` / `load_catalog()` — parse provider plans and reusable catalog entries (`tests/test_plans.py:61`, `tests/test_plans.py:314`).

`PlanLimit.bounds()` and `LimitStatus.state` — calculate quota windows and pacing state (`tests/test_plans.py:98`, `tests/test_plans.py:139`).

`usage_in_window()` / `limit_statuses()` / `payg_burn_rates()` — aggregate fleet telemetry into usage and spending intelligence (`tests/test_plans.py:172`, `tests/test_plans.py:233`, `tests/test_plans.py:406`).

`PlanGovernor.decision()` and `Router._apply_plan_governor()` — turn plan status into routing decisions (`tests/test_plans.py:390`, `tests/test_plans.py:262`).

`seed_known_pricing()` / `sync_bundled_pricing()` / `cost_for_call()` — initialize, refresh, and apply offline model pricing (`tests/test_pricing_safety.py:38`, `tests/test_pricing_bundle.py:47`).

## Design decisions

- Local project state wins over registry reuse, while an explicit `SOMM_DB_DIR` wins over everything tested; registry reuse is announced on stderr to avoid silently sharing telemetry (`tests/test_config_load.py:78`, `tests/test_config_load.py:102`, `tests/test_config_load.py:135`).
- Calendar limits require both exceeding the soft target and running ahead of elapsed time; rolling limits use direct utilization because they lack a fixed calendar progression (`tests/test_plans.py:139`, `tests/test_plans.py:149`).
- A broken or absent plan governor fails open to the original provider chain, preserving availability (`tests/test_plans.py:272`, `tests/test_plans.py:281`).
- Manual pricing is authoritative over bundled data, and missing paid-provider pricing is visible but does not prevent calls (`tests/test_pricing_bundle.py:66`, `tests/test_pricing_safety.py:87`).

## One-liner

These tests ensure Somm remains private, zero-config, offline-capable, cost-aware, and resilient when governing provider usage.
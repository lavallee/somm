## Purpose

These scripts enforce release readiness, performance limits, skill quality, and offline pricing coverage. Together they keep Somm’s packaging consistent and its hot path fast, while preparing intelligence data without introducing runtime network or commercial dependencies.

## How it works

The performance gate measures clean `import somm` latency in fresh subprocesses, then exercises a warmed `SommLLM.generate()` path 500 times against a fake provider and temporary repository. It reports p50 and p95 but fails only when p50 exceeds configurable import or hot-path budgets (`scripts/check_perf_budget.py:18`, `scripts/check_perf_budget.py:52`, `scripts/check_perf_budget.py:69`, `scripts/check_perf_budget.py:96`).

The release gate loads every workspace package’s `pyproject.toml`, requires a single shared version, verifies `somm_core.VERSION`, and checks exact internal `somm`/`somm-core` dependency pins. Releases with major version 1 or higher additionally require a committed approval marker (`scripts/check_release_gate.py:23`, `scripts/check_release_gate.py:54`, `scripts/check_release_gate.py:71`, `scripts/check_release_gate.py:82`).

The skill scorer grades a candidate `SKILL.md` using deterministic behavioral string and regex checks split into training and held-out cases. Only the held-out aggregate is emitted to stdout for `ivy optimize`; diagnostics and training results go to stderr (`scripts/score_skill.py:39`, `scripts/score_skill.py:110`, `scripts/score_skill.py:127`, `scripts/score_skill.py:133`). Separately, the pricing updater fetches or reads LiteLLM pricing, retains supported providers and modes, normalizes prices to per-million-token values, adds curated missing rows, sorts the result, and writes the bundled JSON snapshot (`scripts/update_pricing_bundle.py:40`, `scripts/update_pricing_bundle.py:52`, `scripts/update_pricing_bundle.py:71`, `scripts/update_pricing_bundle.py:110`).

## Key surfaces

- `python scripts/check_perf_budget.py` — runs import and warmed-call performance gates (`scripts/check_perf_budget.py:96`).
- `measure_import_ms()` — measures isolated import latency (`scripts/check_perf_budget.py:52`).
- `measure_hot_path_ms()` — measures warmed generation overhead (`scripts/check_perf_budget.py:69`).
- `python scripts/check_release_gate.py` — validates workspace release invariants (`scripts/check_release_gate.py:54`).
- `python scripts/score_skill.py [SKILL.md]` — emits the held-out skill score (`scripts/score_skill.py:133`).
- `grade()` — evaluates a case mapping and returns its score and per-case results (`scripts/score_skill.py:127`).
- `python scripts/update_pricing_bundle.py [--source FILE]` — regenerates the offline pricing snapshot (`scripts/update_pricing_bundle.py:110`).
- `build_bundle()` — filters and normalizes raw LiteLLM model metadata (`scripts/update_pricing_bundle.py:71`).

## Design decisions

- Performance tests use a fake provider and warmed temporary database, isolating Somm overhead from network and provider latency (`scripts/check_perf_budget.py:22`, `scripts/check_perf_budget.py:71`).
- P95 is reported for visibility, but only median latency gates CI, reducing sensitivity to noisy outliers (`scripts/check_perf_budget.py:100`, `scripts/check_perf_budget.py:108`).
- The release gate uses only the standard library so it can run before dependency installation or builds (`scripts/check_release_gate.py:1`).
- Skill scoring separates machine-readable stdout from human diagnostics and keeps held-out cases distinct from editable training guidance (`scripts/score_skill.py:8`, `scripts/score_skill.py:147`).
- Pricing is pruned to routes Somm supports and bundled into the wheel, preserving offline cost tracking (`scripts/update_pricing_bundle.py:7`, `scripts/update_pricing_bundle.py:40`).

## One-liner

These scripts turn Somm’s performance, release consistency, agent guidance, and offline pricing assumptions into executable maintenance gates.